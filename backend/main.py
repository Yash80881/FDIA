"""GridSentinel AI FastAPI backend.

    python -m backend.main            (or: python cli.py api)
    interactive docs: http://127.0.0.1:8000/docs

Endpoints: /health /latest /history /network /buses /lines /alerts
/attack-statistics /predictions /physics-verification /mitigation
/model-metrics /runs /docs
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from backend.schemas import Health, BusState, LineState, StreamStatus
from backend.services import MonitorService
from config.settings import load_settings, resolve_path
from simulator import schema as S
from simulator.network_visualization import load_or_create_layout

log = logging.getLogger("gridsentinel.api")


class APIError(Exception):
    def __init__(self, status: int, error: str, detail: str, hint: str | None = None):
        self.status, self.error, self.detail, self.hint = status, error, detail, hint


def create_app(cfg: dict | None = None, start_monitor: bool = True) -> FastAPI:
    cfg = cfg or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc = MonitorService(cfg)
        app.state.svc = svc
        app.state.layout = load_or_create_layout(svc.grid, resolve_path(cfg, "processed") / "network_layout.json")
        if start_monitor:
            svc.start()
        log.info("API ready (database %s)", svc.db_path)
        yield
        svc.stop()
        log.info("API stopped cleanly")

    app = FastAPI(title="GridSentinel AI API", version="2.0",
                  description="Smart-grid FDIA detection, localisation and mitigation platform "
                              "(PYPOWER IEEE 30-bus). Ground truth, AI predictions, physics verification and "
                              "mitigation are reported in separate fields.", lifespan=lifespan)

    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError):
        return JSONResponse(status_code=exc.status,
                            content={"error": exc.error, "detail": exc.detail, "hint": exc.hint})

    def svc(request: Request) -> MonitorService:
        return request.app.state.svc

    def resolve_run(s: MonitorService, run_id: Optional[str]) -> dict:
        run = s.active_run(run_id)
        if run is None:
            if run_id:
                raise APIError(404, "run_not_found", f"Run '{run_id}' does not exist.", "GET /runs lists runs.")
            raise APIError(404, "no_data", "No live or replay simulation run found.",
                           "Start the simulator: python cli.py simulate")
        return run

    def latest_step(s: MonitorService, run: dict) -> int:
        step = s.repo.latest_step(run["run_id"])
        if step is None:
            raise APIError(404, "no_measurements", f"Run {run['run_id']} has no measurements yet.",
                           "Wait for the simulator to produce the first snapshot.")
        return step

    def get_pred(s: MonitorService, run_id: str, step: int, compute: bool = True) -> dict | None:
        p = s.prediction(run_id, step)
        if p is None and compute:
            p = s.compute_on_demand(run_id, step)
        return p

    # ------------------------------------------------------------------ health
    @app.get("/health", response_model=Health, tags=["status"])
    def health(request: Request):
        s = svc(request)
        try:
            s.repo.db.query("SELECT 1")
            db_state = "ok"
        except Exception as exc:
            db_state = f"error: {exc}"
        run = s.active_run()
        mon = dict(s.state)
        mon.pop("latencies", None)
        mon.pop("open_alert", None)
        lat = s.state["latencies"][-200:]
        mon["mean_processing_ms"] = float(np.mean(lat)) if lat else None
        mon["monitor_thread_alive"] = bool(s.thread and s.thread.is_alive())
        return Health(api="ok", database=db_state, database_path=str(s.db_path),
                      models_loaded=s.predictor is not None, model_error=s.model_error,
                      model_info=s.predictor.info if s.predictor else None,
                      stream=StreamStatus(**s.stream_status(run)), monitor=mon, server_time=time.time())

    @app.get("/runs", tags=["status"])
    def runs(request: Request, purpose: Optional[str] = None):
        return {"runs": svc(request).repo.list_runs(purpose)}

    # ------------------------------------------------------------------ latest
    @app.get("/latest", tags=["measurements"])
    def latest(request: Request, run_id: Optional[str] = None):
        s = svc(request)
        run = resolve_run(s, run_id)
        step = latest_step(s, run)
        row = s.repo.get_row(run["run_id"], step)
        pred = get_pred(s, run["run_id"], step)
        stats = s.repo.attack_statistics(run["run_id"])
        gens = [{"gen": g, "bus": s.grid.generator_buses[g - 1], "p_mw": row[f"Pg_true_gen_{g}"],
                 "q_mvar": row[f"Qg_true_gen_{g}"]} for g in range(1, S.N_GEN + 1)]
        return {
            "stream": s.stream_status(run),
            "run_id": run["run_id"], "step": step, "timestamp": row["timestamp"],
            "network": {"buses_monitored": S.N_BUS, "lines_monitored": S.N_LINE, "pf_converged": bool(row["pf_converged"]),
                        "scenario": row["scenario"]},
            "system": {"total_load_p_mw": row["total_load_p_mw"], "total_load_q_mvar": row["total_load_q_mvar"],
                       "total_gen_p_mw": row["total_gen_p_mw"], "total_gen_q_mvar": row["total_gen_q_mvar"],
                       "losses_p_mw": row["losses_p_mw"], "generators": gens,
                       "generator_limit_violations": row["gen_limit_violations"] or None},
            "received": {"V_pu": [row[c] for c in S.bus_cols("V_meas")],
                         "Ang_deg": [row[c] for c in S.bus_cols("Ang_meas")]},
            "ground_truth": {**(pred or {}).get("ground_truth", {}),
                             "total_attack_events": stats["total_events"]},
            "ai": (pred or {}).get("ai"),
            "ai_error": (pred or {}).get("ai_error") or s.model_error,
            "physics": _physics_summary((pred or {}).get("physics")),
            "combined": (pred or {}).get("combined"),
        }

    # ----------------------------------------------------------------- history
    HISTORY_GROUPS = {
        "system": ["total_load_p_mw", "total_load_q_mvar", "total_gen_p_mw", "losses_p_mw", "attack_active"],
        "voltage": S.bus_cols("V_meas"), "voltage_true": S.bus_cols("V_true"),
        "angle": S.bus_cols("Ang_meas"), "power": S.bus_cols("P_meas") + S.bus_cols("Q_meas"),
        "loading": S.line_cols("loading"), "line_p": S.line_cols("P_from") + S.line_cols("P_to"),
        "line_q": S.line_cols("Q_from") + S.line_cols("Q_to"),
        "current": S.line_cols("I_from") + S.line_cols("I_to"),
        "generators": S.gen_cols("Pg_true") + S.gen_cols("Qg_true"),
        "labels": ["attack_active", "attack_type", "attack_id", "attack_severity"] + S.ATTACK_BUS_COLUMNS,
    }

    @app.get("/history", tags=["measurements"])
    def history(request: Request, run_id: Optional[str] = None,
                groups: str = Query("system,voltage", description=f"comma list of {sorted(HISTORY_GROUPS)}"),
                limit: int = Query(300, ge=1, le=5000), include_predictions: bool = True):
        s = svc(request)
        run = resolve_run(s, run_id)
        cols = []
        for g in groups.split(","):
            g = g.strip()
            if g not in HISTORY_GROUPS:
                raise APIError(400, "bad_group", f"Unknown history group '{g}'", f"Valid: {sorted(HISTORY_GROUPS)}")
            cols += HISTORY_GROUPS[g]
        limit = min(limit, cfg["api"]["history_limit"] * 10)
        df = s.repo.history(run["run_id"], list(dict.fromkeys(cols)), limit)
        out = {"run_id": run["run_id"], "n": len(df), "stream": s.stream_status(run),
               "data": _columns(df)}
        if include_predictions and len(df):
            preds = {p["step"]: p for p in s.repo.prediction_history(run["run_id"], limit)}
            out["ai"] = {
                "step": [int(x) for x in df["step"]],
                "attack_probability": [(preds.get(int(st), {}).get("ai") or {}).get("attack_probability")
                                       for st in df["step"]],
                "prediction": [(preds.get(int(st), {}).get("ai") or {}).get("prediction") for st in df["step"]],
                "predicted_attack_type": [(preds.get(int(st), {}).get("ai") or {}).get("predicted_attack_type")
                                          for st in df["step"]],
                "combined": [(preds.get(int(st), {}).get("combined") or {}).get("decision") for st in df["step"]],
                "physics_chi2": [(preds.get(int(st), {}).get("physics") or {}).get("chi2_statistic")
                                 for st in df["step"]],
            }
        return out

    # ----------------------------------------------------------------- network
    @app.get("/network", tags=["network"])
    def network(request: Request, run_id: Optional[str] = None):
        s = svc(request)
        g = s.grid
        layout = request.app.state.layout
        out = {"case": g.summary(), "layout_note": "schematic layout from connectivity (not geographic)",
               "buses": [{"bus": b, "type": g.bus_types[b], "is_generator": b in g.generator_buses,
                          "is_reference": b == g.ref_bus, "x": layout[b][0], "y": layout[b][1],
                          "vmin": g.voltage_limits[b][0], "vmax": g.voltage_limits[b][1]} for b in g.bus_numbers],
               "branches": [{"line": br.index, "from_bus": br.from_bus, "to_bus": br.to_bus, "r_pu": br.r_pu,
                             "x_pu": br.x_pu, "b_pu": br.b_pu, "rating_mva": br.rate_a_mva} for br in g.branches]}
        run = s.active_run(run_id)
        if run is not None and s.repo.latest_step(run["run_id"]) is not None:
            step = s.repo.latest_step(run["run_id"])
            row = s.repo.get_row(run["run_id"], step)
            out["flows"] = {"step": step, "p_from_mw": [row[c] for c in S.line_cols("P_from")],
                            "loading_pct": [row[c] for c in S.line_cols("loading")]}
        return out

    # ------------------------------------------------------------------- buses
    @app.get("/buses", response_model=list[BusState], tags=["network"])
    def buses(request: Request, run_id: Optional[str] = None, step: Optional[int] = None):
        s = svc(request)
        run = resolve_run(s, run_id)
        step = latest_step(s, run) if step is None else step
        row = s.repo.get_row(run["run_id"], step)
        if row is None:
            raise APIError(404, "step_not_found", f"Step {step} not found in run {run['run_id']}")
        pred = get_pred(s, run["run_id"], step)
        ai = (pred or {}).get("ai")
        out = []
        for b in s.grid.bus_numbers:
            lo, hi = s.grid.voltage_limits[b]
            v = row[f"V_meas_bus_{b}"]
            out.append(BusState(
                bus=b, type=s.grid.bus_types[b], is_generator=b in s.grid.generator_buses,
                received={"V_pu": v, "Ang_deg": row[f"Ang_meas_bus_{b}"], "P_mw": row[f"P_meas_bus_{b}"],
                          "Q_mvar": row[f"Q_meas_bus_{b}"]},
                ground_truth={"label": "SIMULATOR GROUND TRUTH", "V_pu": row[f"V_true_bus_{b}"],
                              "Ang_deg": row[f"Ang_true_bus_{b}"], "P_mw": row[f"P_true_bus_{b}"],
                              "Q_mvar": row[f"Q_true_bus_{b}"], "Pd_mw": row[f"Pd_true_bus_{b}"],
                              "Qd_mvar": row[f"Qd_true_bus_{b}"], "attacked": bool(row[f"attack_bus_{b}"])},
                ai=None if ai is None else {
                    "attack_probability": ai["bus_probabilities"][str(b)],
                    "threshold": ai["bus_thresholds"][str(b)],
                    "predicted_compromised": b in ai["predicted_compromised_buses"]},
                voltage_limits=[lo, hi], voltage_violation=bool(v < lo or v > hi)))
        return out

    # ------------------------------------------------------------------- lines
    @app.get("/lines", response_model=list[LineState], tags=["network"])
    def lines(request: Request, run_id: Optional[str] = None, step: Optional[int] = None):
        s = svc(request)
        run = resolve_run(s, run_id)
        step = latest_step(s, run) if step is None else step
        row = s.repo.get_row(run["run_id"], step)
        if row is None:
            raise APIError(404, "step_not_found", f"Step {step} not found in run {run['run_id']}")
        out = []
        for br in s.grid.branches:
            k = br.index
            rec = {q: row[f"{q}_line_{k}"] for q in S.LINE_QUANTITIES}
            tru = {q: row[f"{q}_true_line_{k}"] for q in S.LINE_QUANTITIES}
            out.append(LineState(line=k, from_bus=br.from_bus, to_bus=br.to_bus, rating_mva=br.rate_a_mva,
                                 received=rec, ground_truth=tru,
                                 overloaded=None if rec["loading"] is None else bool(rec["loading"] >= 100)))
        return out

    # ------------------------------------------------------------------ alerts
    @app.get("/alerts", tags=["security"])
    def alerts(request: Request, run_id: Optional[str] = None, limit: int = Query(100, ge=1, le=1000)):
        s = svc(request)
        run = s.active_run(run_id)
        return {"run_id": run["run_id"] if run else None,
                "alerts": s.repo.alerts(run["run_id"] if run else None, limit),
                "note": "Alerts are raised from AI + physics decisions; 'ground_truth_attack_ids' are simulator "
                        "labels shown for evaluation only."}

    @app.get("/attack-statistics", tags=["security"])
    def attack_statistics(request: Request, run_id: Optional[str] = None):
        s = svc(request)
        run = resolve_run(s, run_id)
        rid = run["run_id"]
        gt = s.repo.attack_statistics(rid)
        events = s.repo.attack_events(rid, 500)
        hist = s.repo.prediction_history(rid, 100000)
        pred_types = {}
        for h in hist:
            t = (h.get("ai") or {}).get("predicted_attack_type")
            if t:
                pred_types[t] = pred_types.get(t, 0) + 1
        return {"run_id": rid, "ground_truth": {"label": "SIMULATOR GROUND TRUTH", **gt, "events": events},
                "ai": {"label": "AI predictions", "predicted_attack_steps_by_type": pred_types,
                       "processed_steps": len(hist)}}

    # ------------------------------------------------------------- predictions
    @app.get("/predictions", tags=["security"])
    def predictions(request: Request, run_id: Optional[str] = None, step: Optional[int] = None):
        s = svc(request)
        run = resolve_run(s, run_id)
        step = latest_step(s, run) if step is None else step
        p = get_pred(s, run["run_id"], step)
        if p is None:
            raise APIError(404, "step_not_found", f"Step {step} not found in run {run['run_id']}")
        if p.get("ai") is None:
            raise APIError(503, "models_unavailable", p.get("ai_error") or s.model_error or "Models not loaded",
                           "Train the models: python cli.py train")
        return {"run_id": run["run_id"], "step": step, "stream": s.stream_status(run), "ai": p["ai"],
                "ground_truth": p["ground_truth"], "combined": p["combined"],
                "live_performance": s.live_performance(run["run_id"])}

    @app.get("/physics-verification", tags=["security"])
    def physics(request: Request, run_id: Optional[str] = None, step: Optional[int] = None):
        s = svc(request)
        run = resolve_run(s, run_id)
        step = latest_step(s, run) if step is None else step
        p = get_pred(s, run["run_id"], step)
        if p is None:
            raise APIError(404, "step_not_found", f"Step {step} not found in run {run['run_id']}")
        return {"run_id": run["run_id"], "step": step, "physics": p["physics"],
                "ai_detection": None if p.get("ai") is None else {
                    "prediction": p["ai"]["prediction"], "attack_probability": p["ai"]["attack_probability"],
                    "predicted_compromised_buses": p["ai"]["predicted_compromised_buses"]},
                "combined": p["combined"], "ground_truth": p["ground_truth"],
                "limitations": "Residual tests detect perturbations inconsistent with the network model; a "
                               "stealthy attack of the form a = Hc is not detectable by residuals alone."}

    @app.get("/mitigation", tags=["security"])
    def mitigation(request: Request, run_id: Optional[str] = None, step: Optional[int] = None):
        s = svc(request)
        run = resolve_run(s, run_id)
        step = latest_step(s, run) if step is None else step
        if s.repo.get_row(run["run_id"], step) is None:
            raise APIError(404, "step_not_found", f"Step {step} not found in run {run['run_id']}")
        return s.mitigation(run["run_id"], step)

    @app.get("/model-metrics", tags=["models"])
    def model_metrics(request: Request):
        d = resolve_path(cfg, "models") / "metrics"
        out = {"label": "OFFLINE evaluation on independent test runs (not live data)"}
        for name in ("summary", "detection", "localization", "attack_type", "severity", "data_distribution"):
            p = d / f"{name}.json"
            out[name] = json.loads(p.read_text()) if p.exists() else None
        if out["summary"] is None:
            raise APIError(404, "no_metrics", "No offline metrics found.", "Train the models: python cli.py train")
        return out

    return app


def _columns(df) -> dict:
    """DataFrame -> {column: [values]} with NaN as null (via pandas' JSON writer)."""
    d = json.loads(df.to_json(orient="split"))
    return {c: [row[i] for row in d["data"]] for i, c in enumerate(d["columns"])}


def _physics_summary(p: dict | None) -> dict | None:
    if not p:
        return None
    keys = ("status", "converged", "observable", "chi2_statistic", "chi2_threshold", "chi2_alarm",
            "max_normalized_residual", "lnr_alarm", "bad_data_detected", "suspicious_buses", "latency_ms")
    return {k: p.get(k) for k in keys}


def main() -> int:
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser(description="GridSentinel AI API")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_settings(None, args.set)
    from simulator.continuous_simulator import setup_logging
    setup_logging(cfg, "api")
    uvicorn.run(create_app(cfg), host=cfg["api"]["host"], port=int(cfg["api"]["port"]), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
