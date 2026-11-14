"""Backend services: stream status, cached inference, physics verification,
alerting and mitigation.

The :class:`MonitorService` runs in a background thread inside the API
process.  It polls SQLite (indexed query on the newest step - never a CSV),
and for every new snapshot of the active run it computes, exactly once:

    AI prediction (Random Forest two-stage pipeline)
    physics verification (WLS + chi-square + normalised residuals)
    combined decision and alert state

and caches the result in the ``predictions`` table (key: run_id, step) and in
memory.  API requests only read the cache, so models are never re-run for an
unchanged snapshot.  If it falls behind by more than ``max_catch_up_steps``
it jumps to the newest snapshot and counts the skipped ones as *dropped*.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, OrderedDict

import numpy as np

from config.settings import resolve_path
from ml.common import ModelLoadError
from physics.mitigation import MitigationAdvisor
from physics.state_estimation import WLSStateEstimator
from physics.verification import verify_snapshot, combine
from simulator import schema as S
from simulator.grid_model import GridModel
from storage.database import Database
from storage.repository import Repository

log = logging.getLogger("gridsentinel.backend")

WINDOW_COLUMNS = ["run_id", "step"] + S.ML_INPUT_COLUMNS


def ground_truth_of(row: dict) -> dict:
    return {
        "label": "SIMULATOR GROUND TRUTH",
        "attack_active": int(row["attack_active"]),
        "attack_id": row["attack_id"] or None,
        "attack_type": row["attack_type"],
        "attacked_buses": [b for b in range(1, S.N_BUS + 1) if row.get(f"attack_bus_{b}") == 1],
        "attack_severity": row["attack_severity"],
        "attack_step_in_event": row["attack_step_in_event"],
        "attack_duration_steps": row["attack_duration_steps"],
        "attack_channels": row["attack_channels"],
    }


class LRU(OrderedDict):
    def __init__(self, maxsize=512):
        super().__init__()
        self.maxsize = maxsize

    def put(self, key, value):
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.maxsize:
            self.popitem(last=False)


class MonitorService:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.db_path = resolve_path(cfg, "database")
        self.db = Database(self.db_path)           # API-request connection
        self.repo = Repository(self.db)
        self.grid = GridModel()
        self.se = WLSStateEstimator(self.grid, cfg["physics"]["max_iterations"], cfg["physics"]["tolerance"])
        self.advisor = MitigationAdvisor(self.se, cfg)
        self.predictor = None
        self.model_error = None
        self.load_models()
        self.cache = LRU(2048)
        self.mitigation_cache = LRU(64)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.state: dict = {"run_id": None, "processed": 0, "dropped": 0, "last_step": None,
                            "latencies": [], "open_alert": None}
        self.lock = threading.RLock()
        self._noise_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------ models
    def load_models(self) -> None:
        from ml.inference import GridSentinelPredictor
        try:
            self.predictor = GridSentinelPredictor(self.cfg)
            self.model_error = None
            log.info("Models loaded: %s", self.predictor.info)
        except ModelLoadError as exc:
            self.predictor = None
            self.model_error = str(exc)
            log.warning("Models unavailable: %s", exc)

    def run_noise(self, repo: Repository, run_id: str) -> dict:
        """Sensor-accuracy spec of a run (replays use their source run's spec)."""
        if run_id in self._noise_cache:
            return self._noise_cache[run_id]
        noise = self.cfg["noise"]
        run = repo.get_run(run_id)
        if run and run.get("source_run_id"):
            run = repo.get_run(run["source_run_id"]) or run
        if run and run.get("config_json"):
            try:
                noise = json.loads(run["config_json"])["noise"]
            except (ValueError, KeyError):
                pass
        self._noise_cache[run_id] = noise
        return noise

    # --------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.repo.mark_stale_runs(older_than_s=max(60.0, 10 * self.cfg["simulation"]["stale_after_s"]))
        self.thread = threading.Thread(target=self._loop, name="gridsentinel-monitor", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        self.db.close()

    # ------------------------------------------------------------- stream status
    def active_run(self, run_id: str | None = None) -> dict | None:
        if run_id:
            return self.repo.get_run(run_id)
        return self.repo.latest_run()

    def stream_status(self, run: dict | None) -> dict:
        if run is None:
            return {"state": "no_data", "live": False,
                    "message": "No live or replay run found. Start the simulator: python cli.py simulate"}
        age = time.time() - (run["last_heartbeat"] or 0)
        stale_after = max(self.cfg["simulation"]["stale_after_s"], 3 * (run.get("interval_s") or 1.0))
        if run["status"] == "running" and age <= stale_after:
            state, live = "running", True
            msg = "Simulator is producing new measurements."
        elif run["status"] == "running":
            state, live = "stale", False
            msg = f"No new measurement for {age:.0f}s - the simulator may have frozen or crashed."
        else:
            state, live = run["status"], False
            msg = f"Run {run['status']}: {run.get('message') or ''}".strip()
        return {"state": state, "live": live, "message": msg, "run_id": run["run_id"], "mode": run["mode"],
                "last_step": run["last_step"], "seconds_since_last_measurement": round(age, 2),
                "stale_after_s": stale_after, "interval_s": run.get("interval_s")}

    # -------------------------------------------------------------- the loop
    def _loop(self) -> None:
        db = Database(self.db_path)
        repo = Repository(db)
        poll = float(self.cfg["api"]["monitor_poll_s"])
        max_backlog = int(self.cfg["api"]["max_catch_up_steps"])
        while not self.stop_event.is_set():
            try:
                run = repo.latest_run()
                if run is None:
                    self.stop_event.wait(poll)
                    continue
                rid = run["run_id"]
                if rid != self.state["run_id"]:
                    stats = repo.monitor_stats(rid) or {}
                    self.state.update(run_id=rid, processed=stats.get("processed", 0),
                                      dropped=stats.get("dropped", 0),
                                      last_step=stats.get("last_processed_step"), latencies=[], open_alert=None)
                latest = repo.latest_step(rid)
                if latest is None or (self.state["last_step"] is not None and latest <= self.state["last_step"]):
                    self.stop_event.wait(poll)
                    continue
                start = latest if self.state["last_step"] is None else self.state["last_step"] + 1
                if latest - start + 1 > max_backlog:
                    self.state["dropped"] += latest - max_backlog + 1 - start
                    start = latest - max_backlog + 1
                # Process one step, then re-check the newest step: if the simulator has run ahead by more
                # than ``max_catch_up_steps`` the monitor jumps forward, so the live AI result always belongs
                # to (almost) the newest snapshot instead of trailing further and further behind.
                self.process_step(repo, rid, start)
                self.state["last_step"] = start
                lat = self.state["latencies"][-200:]
                repo.update_monitor_stats(rid, self.state["processed"], self.state["dropped"],
                                          self.state["last_step"], float(np.mean(lat)) if lat else 0.0)
            except Exception:
                log.exception("Monitor loop error (will retry)")
                self.stop_event.wait(2.0)
        db.close()

    def process_step(self, repo: Repository, run_id: str, step: int) -> dict | None:
        cached = repo.get_prediction(run_id, step)
        if cached is not None:
            self.cache.put((run_id, step), cached)
            return cached
        t0 = time.perf_counter()
        row = repo.get_row(run_id, step)
        if row is None:
            return None
        payload = {"run_id": run_id, "step": step, "timestamp": row["timestamp"],
                   "ground_truth": ground_truth_of(row)}
        noise = self.run_noise(repo, run_id)
        if self.predictor is not None:
            n = self.predictor.history_steps
            window = repo.get_rows(run_id, step - n + 1, step, WINDOW_COLUMNS)
            try:
                payload["ai"] = self.predictor.predict(window, noise)
            except Exception as exc:   # a model failure must not stall the monitor
                log.exception("AI inference failed at %s step %d", run_id, step)
                payload["ai"] = None
                payload["ai_error"] = f"inference failed: {type(exc).__name__}: {exc}"
        else:
            payload["ai"] = None
            payload["ai_error"] = self.model_error
        phys = verify_snapshot(self.se, row, dict(self.cfg, noise=noise), identify=True)
        payload["physics"] = {k: v for k, v in phys.items()
                              if k not in ("cleaned_estimated_vm", "cleaned_estimated_va_deg")}
        prob = payload["ai"]["attack_probability"] if payload["ai"] else None
        thr = payload["ai"]["detection_threshold"] if payload["ai"] else 0.5
        payload["combined"] = combine(prob, thr, phys)
        payload["processing_ms"] = (time.perf_counter() - t0) * 1000
        repo.save_prediction(run_id, step, payload)
        self.cache.put((run_id, step), payload)
        self.state["processed"] += 1
        self.state["latencies"].append(payload["processing_ms"])
        self.state["latencies"] = self.state["latencies"][-500:]
        self._update_alerts(repo, run_id, step, row, payload)
        return payload

    # ------------------------------------------------------------------ alerts
    def _update_alerts(self, repo, run_id, step, row, payload) -> None:
        dec = payload["combined"]["decision"]
        oa = self.state["open_alert"]
        ai = payload["ai"] or {}
        gt_id = payload["ground_truth"]["attack_id"]
        if dec == "ATTACK":
            buses = ai.get("predicted_compromised_buses") or payload["physics"].get("suspicious_buses", [])
            if oa is None or step - oa["end_step"] > 3:
                if oa is not None:
                    repo.update_alert(oa["alert_id"], {"status": "closed"})
                action = self._recommended_action(row, buses)
                data = {"predicted_type": ai.get("predicted_attack_type") or "unknown",
                        "predicted_buses": sorted(buses), "probability": ai.get("attack_probability", 0.0),
                        "est_severity": ai.get("estimated_severity"),
                        "physics_status": payload["combined"]["agreement"], "recommended_action": action,
                        "gt_attack_ids": [gt_id] if gt_id else []}
                aid = repo.open_alert(run_id, step, row["timestamp"], data)
                self.state["open_alert"] = {"alert_id": aid, "end_step": step, "types": Counter([data["predicted_type"]]),
                                            "buses": set(buses), "prob": data["probability"],
                                            "gt": set(data["gt_attack_ids"]), "sev": data["est_severity"] or 0.0,
                                            "agree": Counter([data["physics_status"]])}
            else:
                oa["end_step"] = step
                oa["types"][ai.get("predicted_attack_type") or "unknown"] += 1
                oa["buses"] |= set(buses)
                oa["prob"] = max(oa["prob"], ai.get("attack_probability", 0.0))
                oa["sev"] = max(oa["sev"], ai.get("estimated_severity") or 0.0)
                oa["agree"][payload["combined"]["agreement"]] += 1
                if gt_id:
                    oa["gt"].add(gt_id)
                repo.update_alert(oa["alert_id"], {
                    "end_step": step, "end_time": row["timestamp"],
                    "predicted_type": oa["types"].most_common(1)[0][0],
                    "predicted_buses": sorted(oa["buses"]), "max_probability": oa["prob"],
                    "est_severity": oa["sev"], "physics_status": oa["agree"].most_common(1)[0][0],
                    "ground_truth_attack_ids": sorted(oa["gt"])})
        elif oa is not None and step - oa["end_step"] > 3:
            repo.update_alert(oa["alert_id"], {"status": "closed"})
            self.state["open_alert"] = None

    def _recommended_action(self, row, buses) -> str:
        try:
            rec = self.advisor.recommend(row, buses, include_truth=False)
            return " | ".join(a["text"] for a in rec.get("recommendations", [])[:2])
        except Exception as exc:   # recommendation must never break alerting
            log.warning("mitigation advisor failed: %s", exc)
            return "Verify the flagged sensors and run state estimation with the suspicious channels excluded."

    # --------------------------------------------------------- read helpers
    def prediction(self, run_id: str, step: int) -> dict | None:
        key = (run_id, step)
        if key in self.cache:
            return self.cache[key]
        p = self.repo.get_prediction(run_id, step)
        if p is not None:
            self.cache.put(key, p)
        return p

    def compute_on_demand(self, run_id: str, step: int) -> dict | None:
        """Compute (and cache) a prediction for a step the monitor has not processed."""
        with self.lock:
            return self.process_step(self.repo, run_id, step) if self.prediction(run_id, step) is None \
                else self.prediction(run_id, step)

    def mitigation(self, run_id: str, step: int) -> dict:
        key = (run_id, step)
        if key in self.mitigation_cache:
            return self.mitigation_cache[key]
        row = self.repo.get_row(run_id, step)
        pred = self.prediction(run_id, step) or self.compute_on_demand(run_id, step) or {}
        buses = (pred.get("ai") or {}).get("predicted_compromised_buses", [])
        self.advisor.cfg = dict(self.cfg, noise=self.run_noise(self.repo, run_id))
        out = self.advisor.recommend(row, buses, include_truth=True)
        out.update({"run_id": run_id, "step": step, "ml_predicted_buses": buses,
                    "combined_decision": (pred.get("combined") or {}).get("decision")})
        self.mitigation_cache.put(key, out)
        return out

    def live_performance(self, run_id: str, limit: int = 300) -> dict:
        """Rolling comparison of live AI predictions with simulator ground truth."""
        hist = self.repo.prediction_history(run_id, limit)
        hist = [h for h in hist if h.get("ai")]
        if not hist:
            return {"label": "live predictions vs simulator ground truth", "n": 0}
        y = np.array([h["ground_truth"]["attack_active"] for h in hist])
        p = np.array([h["ai"]["prediction"] == "ATTACK" for h in hist]).astype(int)
        c = np.array([h["combined"]["decision"] == "ATTACK" for h in hist]).astype(int)
        from ml.common import binary_metrics
        Yb = np.array([[int(b in h["ground_truth"]["attacked_buses"]) for b in range(1, 31)] for h in hist])
        Pb = np.array([[int(b in h["ai"]["predicted_compromised_buses"]) for b in range(1, 31)] for h in hist])
        tp = int((Yb & Pb).sum())
        prec = tp / max(Pb.sum(), 1)
        rec = tp / max(Yb.sum(), 1)
        type_ok = [h["ai"]["predicted_attack_type"] == h["ground_truth"]["attack_type"]
                   for h in hist if h["ground_truth"]["attack_active"] and h["ai"]["prediction"] == "ATTACK"]
        return {"label": "LIVE rolling comparison: AI predictions vs simulator ground truth",
                "window_steps": len(hist),
                "detection": binary_metrics(y, p), "combined_decision": binary_metrics(y, c),
                "localization_micro_precision": prec, "localization_micro_recall": rec,
                "localization_micro_f1": 2 * prec * rec / max(prec + rec, 1e-12) if tp else 0.0,
                "attack_type_accuracy_on_detected": float(np.mean(type_ok)) if type_ok else None,
                "mean_processing_ms": float(np.mean([h.get("processing_ms", 0) for h in hist]))}
