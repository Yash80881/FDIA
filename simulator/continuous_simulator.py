"""Continuous IEEE 30-bus measurement simulator (live and dataset-replay modes).

Per step (live mode):
 1. update demand (load profile model)          5. check convergence (never use a failed solution)
 2. load variations                             6. extract true measurements
 3. generator re-dispatch within limits         7. add sensor noise -> received measurements
 4. AC power flow with PYPOWER runpf            8. optional FDIA on received layer + labels
 9. store snapshot + attack metadata in SQLite (WAL), heartbeat
10. the backend monitor picks the new snapshot up (publish via the database)

Run ``python -m simulator.continuous_simulator --help`` for options.  Stop with
Ctrl+C, ``python cli.py stop`` or by creating ``data/STOP_SIMULATOR``.
"""
from __future__ import annotations

import argparse
import json
import logging
import secrets
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import load_settings, resolve_path, ensure_directories, config_hash
from simulator.grid_model import GridModel
from simulator.power_flow import run_ac_power_flow, PowerFlowError, PowerFlowResult
from simulator.load_profiles import LoadProfileModel
from simulator.dispatch import GeneratorDispatcher, check_generator_limits
from simulator.measurement_generator import MeasurementGenerator, Snapshot
from simulator import schema as S
from attacks.attack_scheduler import AttackScheduler
from storage.database import Database
from storage.repository import Repository, utc_now_iso
from pypower import idx_bus as IB
from pypower import idx_gen as IG

log = logging.getLogger("gridsentinel.simulator")


def make_run_id(mode: str, seed: int, tag: str = "") -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [mode, tag, f"s{seed}", stamp, secrets.token_hex(2)]
    return "-".join(p for p in parts if p)


class SimulationEngine:
    """Pure simulation logic (no I/O): produces one labelled row per step."""

    def __init__(self, cfg: dict, run_id: str, seed: int | None = None):
        self.cfg = cfg
        self.run_id = run_id
        self.seed = int(cfg["project"]["random_seed"] if seed is None else seed)
        self.grid = GridModel()
        # independent random streams -> changing noise does not change the attack sequence
        self.rng_load = np.random.default_rng(self.seed)
        self.rng_noise = np.random.default_rng(self.seed + cfg["noise"]["seed_offset"])
        self.rng_attack = np.random.default_rng(self.seed + cfg["attacks"]["seed_offset"])
        self.loads = LoadProfileModel(self.grid, cfg, self.rng_load)
        self.dispatcher = GeneratorDispatcher(self.grid, cfg, self.rng_load)
        self.meas = MeasurementGenerator(self.grid, cfg, self.rng_noise)
        self.attacks = AttackScheduler(self.grid, cfg, self.rng_attack, run_id, self.meas)
        self.enforce_q = bool(cfg["simulation"]["enforce_q_limits"])
        self.pf_failures_total = 0
        self.consecutive_failures = 0

    def _solve(self, pd, qd, pg) -> PowerFlowResult:
        case = self.grid.new_case()
        case["bus"][:, IB.PD] = pd
        case["bus"][:, IB.QD] = qd
        case["gen"][:, IG.PG] = pg
        return run_ac_power_flow(case, self.enforce_q)

    def step(self, t: int, timestamp: str, sim_time_s: float) -> tuple[dict | None, Snapshot | None]:
        load = self.loads.step(t)
        pg = self.dispatcher.dispatch(load.total_p_mw)
        try:
            pf = self._solve(load.pd_mw, load.qd_mvar, pg)
        except PowerFlowError as first_err:
            # recovery attempt: fall back to the base dispatch at the same demand
            log.warning("Step %d: power flow failed (%s); retrying with base dispatch", t, first_err)
            try:
                pf = self._solve(load.pd_mw, load.qd_mvar, self.dispatcher.base_pg)
            except PowerFlowError as err:
                self.pf_failures_total += 1
                self.consecutive_failures += 1
                log.error("Step %d: power flow failed again - no measurements produced (%s)", t, err)
                return None, None
        self.consecutive_failures = 0
        snap = self.meas.snapshot(t, pf)
        labels = self.attacks.process(snap)
        violations = check_generator_limits(pf.pg_mw, pf.qg_mvar, self.grid,
                                            self.cfg["generation"]["slack_margin_mw"])
        row = self.build_row(t, timestamp, sim_time_s, snap, pf, load.scenario, labels, violations)
        return row, snap

    def build_row(self, t, timestamp, sim_time_s, snap: Snapshot, pf: PowerFlowResult,
                  scenario: str, labels: dict, violations: list[str]) -> dict:
        row: dict = {
            "run_id": self.run_id, "step": t, "timestamp": timestamp, "sim_time_s": sim_time_s,
            "pf_converged": 1, "pf_iterations": pf.iterations, "scenario": scenario,
            "total_load_p_mw": float(pf.pd_mw.sum()), "total_load_q_mvar": float(pf.qd_mvar.sum()),
            "total_gen_p_mw": float(pf.pg_mw.sum()), "total_gen_q_mvar": float(pf.qg_mvar.sum()),
            "losses_p_mw": float(pf.pg_mw.sum() - pf.pd_mw.sum()),
            "gen_limit_violations": "; ".join(violations),
        }
        for prefix, arr in (("V_true", snap.true_bus.vm), ("Ang_true", snap.true_bus.va),
                            ("P_true", snap.true_bus.p), ("Q_true", snap.true_bus.q),
                            ("V_meas", snap.meas_bus.vm), ("Ang_meas", snap.meas_bus.va),
                            ("P_meas", snap.meas_bus.p), ("Q_meas", snap.meas_bus.q),
                            ("Pd_true", pf.pd_mw), ("Qd_true", pf.qd_mvar)):
            row.update(zip(S.bus_cols(prefix), arr.tolist()))
        for q, arr in snap.meas_line.as_dict().items():
            row.update(zip(S.line_cols(q), arr.tolist()))
        for q, arr in snap.true_line.as_dict().items():
            row.update(zip(S.line_cols(q, true=True), arr.tolist()))
        row.update(zip(S.gen_cols("Pg_true"), pf.pg_mw.tolist()))
        row.update(zip(S.gen_cols("Qg_true"), pf.qg_mvar.tolist()))
        row.update(labels)
        return row


class ContinuousSimulator:
    """Runs a :class:`SimulationEngine` continuously and persists every step."""

    def __init__(self, cfg: dict, mode: str = "live", purpose: str = "live", seed: int | None = None,
                 fast: bool = False, max_steps: int | None = None, tag: str = ""):
        self.cfg = cfg
        self.mode = mode
        self.purpose = purpose
        self.seed = int(cfg["project"]["random_seed"] if seed is None else seed)
        self.fast = fast
        self.max_steps = max_steps
        self.interval = float(cfg["simulation"]["interval_s"])
        self.run_id = make_run_id(mode if mode != "live" or purpose == "live" else "ds", self.seed, tag)
        self.stop_requested = False
        self.stop_flag = resolve_path(cfg, "stop_flag")
        ensure_directories(cfg)
        self.db = Database(resolve_path(cfg, "database"))
        self.repo = Repository(self.db)
        self._event_sev: dict[str, list[float]] = {}

    # ---------------------------------------------------------- signal handling
    def request_stop(self, *_):
        if not self.stop_requested:
            log.info("Stop requested - finishing the current step and closing the database safely")
        self.stop_requested = True

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)
        if hasattr(signal, "SIGBREAK"):  # Windows Ctrl+Break
            signal.signal(signal.SIGBREAK, self.request_stop)

    def _check_stop_flag(self) -> bool:
        if self.stop_flag.exists():
            log.info("Stop flag file found (%s)", self.stop_flag)
            try:
                self.stop_flag.unlink()
            except OSError:
                pass
            return True
        return False

    # ----------------------------------------------------------------- live
    def run(self) -> str:
        engine = SimulationEngine(self.cfg, self.run_id, self.seed)
        self.repo.create_run(self.run_id, self.mode, self.purpose, self.seed, self.cfg,
                             config_hash(self.cfg), self.interval)
        log.info("Run %s started (%s, seed %d, interval %.2fs%s)", self.run_id, self.mode, self.seed,
                 self.interval, ", fast" if self.fast else "")
        commit_every = 50 if self.fast else max(int(self.cfg["simulation"]["commit_every_steps"]), 1)
        retention = self.cfg["retention"]
        sim_start = datetime.now(timezone.utc)
        status, message = "stopped", "stopped by user"
        t = 0
        next_tick = time.perf_counter()
        try:
            while not self.stop_requested:
                if self.max_steps is not None and t >= self.max_steps:
                    status, message = "completed", f"completed {t} steps"
                    break
                if t % 10 == 0 and self._check_stop_flag():
                    break
                sim_time = t * self.interval
                ts = ((sim_start + timedelta(seconds=sim_time)).isoformat(timespec="milliseconds")
                      if self.fast else utc_now_iso())
                row, _ = engine.step(t, ts, sim_time)
                if row is not None:
                    commit = (t + 1) % commit_every == 0
                    self.repo.insert_measurement(row, commit=False)
                    self._record_event(engine, row, ts, commit=False)
                    if commit or not self.fast:
                        self.repo.heartbeat(self.run_id, t, engine.pf_failures_total, commit=True)
                elif engine.consecutive_failures >= self.cfg["simulation"]["max_consecutive_pf_failures"]:
                    status, message = "failed", f"{engine.consecutive_failures} consecutive power-flow failures"
                    log.error(message)
                    break
                if (self.purpose == "live" and t > 0 and t % retention["prune_every_steps"] == 0):
                    n = self.repo.prune_run(self.run_id, retention["live_max_steps"])
                    if n:
                        log.info("Retention: pruned %d old rows", n)
                t += 1
                if not self.fast:
                    next_tick += self.interval
                    delay = next_tick - time.perf_counter()
                    if delay > 0:
                        self._sleep(delay)
                    else:
                        if -delay > self.interval:
                            log.warning("Simulator is running %.2fs behind schedule", -delay)
                        next_tick = time.perf_counter()
        except Exception as exc:
            status, message = "failed", f"{type(exc).__name__}: {exc}"
            log.exception("Simulator crashed")
            raise
        finally:
            self.repo.heartbeat(self.run_id, t - 1, engine.pf_failures_total, commit=True)
            self.repo.finish_run(self.run_id, status, message)
            self.db.close()
            log.info("Run %s %s after %d steps (%d power-flow failures)", self.run_id, status, t,
                     engine.pf_failures_total)
        return self.run_id

    def _sleep(self, seconds: float) -> None:
        end = time.perf_counter() + seconds
        while not self.stop_requested:
            remaining = end - time.perf_counter()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.2))

    def _record_event(self, engine: "SimulationEngine | None", row: dict, ts: str, commit: bool) -> None:
        """Create/update the attack_events record of the attack active in ``row``."""
        aid = row["attack_id"]
        if not aid:
            return
        state = self._event_sev.setdefault(aid, {"sev": [], "start": row["step"], "start_ts": ts})
        state["sev"].append(row["attack_severity"])
        buses = [b for b in range(1, S.N_BUS + 1) if row.get(f"attack_bus_{b}")]
        rec = {"attack_id": aid, "attack_type": row["attack_type"], "start_step": state["start"],
               "end_step": row["step"], "duration_steps": row["step"] - state["start"] + 1,
               "buses": buses, "channels": json.loads(row.get("attack_channels") or "[]"),
               "params": {}, "bus_channels": {}, "line_offsets": {}}
        if engine is not None:
            event = next((e for e in reversed(engine.attacks.events) if e.attack_id == aid), None)
            if event is not None:
                full = event.to_record()
                rec.update(params=full["params"], bus_channels=full["bus_channels"],
                           line_offsets=full["line_offsets"], buses=full["buses"])
        self.repo.upsert_attack_event(self.run_id, rec, state["start_ts"], float(np.mean(state["sev"])),
                                      float(np.max(state["sev"])), commit=commit)

    # --------------------------------------------------------------- replay
    def replay(self, source_run_id: str, rate: float | None = None, loop: bool = False) -> str:
        """Re-publish a recorded run as a new run at a configurable playback rate."""
        rate = float(rate or self.cfg["simulation"]["replay_rate"])
        src = self.repo.get_run(source_run_id)
        if src is None:
            raise ValueError(f"Source run '{source_run_id}' not found")
        last = self.repo.latest_step(source_run_id)
        if last is None:
            raise ValueError(f"Source run '{source_run_id}' has no measurements")
        self.run_id = make_run_id("replay", self.seed)
        interval = (src["interval_s"] or self.interval) / max(rate, 1e-6)
        self.repo.create_run(self.run_id, "replay", "live", self.seed, self.cfg, config_hash(self.cfg),
                             interval, source_run_id=source_run_id)
        log.info("Replaying %s (%d steps) as %s at %.2fx", source_run_id, last + 1, self.run_id, rate)
        out_step, status, message = 0, "stopped", "stopped by user"
        next_tick = time.perf_counter()
        pass_no = 0
        try:
            while not self.stop_requested:
                pass_no += 1
                for chunk_start in range(0, last + 1, 200):
                    df = self.repo.get_rows(source_run_id, chunk_start, chunk_start + 199)
                    for rec in df.to_dict("records"):
                        if self.stop_requested or (out_step % 10 == 0 and self._check_stop_flag()):
                            self.stop_requested = True
                            break
                        if self.max_steps is not None and out_step >= self.max_steps:
                            self.stop_requested = True
                            status, message = "completed", f"replayed {out_step} steps"
                            break
                        rec["run_id"] = self.run_id
                        rec["step"] = out_step
                        rec["timestamp"] = utc_now_iso()
                        if rec.get("attack_id"):
                            # keep replayed attack IDs unique per replay pass
                            rec["attack_id"] = f"{self.run_id}:p{pass_no}:{rec['attack_id']}"
                        self.repo.insert_measurement(rec, commit=False)
                        self._record_event(None, rec, rec["timestamp"], commit=False)
                        self.repo.heartbeat(self.run_id, out_step, 0, commit=True)
                        out_step += 1
                        if not self.fast:
                            next_tick += interval
                            d = next_tick - time.perf_counter()
                            if d > 0:
                                self._sleep(d)
                            else:
                                next_tick = time.perf_counter()
                    if self.stop_requested:
                        break
                if not loop:
                    if not self.stop_requested:
                        status, message = "completed", f"replayed {out_step} steps"
                    break
        finally:
            self.repo.finish_run(self.run_id, status, message)
            self.db.close()
        return self.run_id


def setup_logging(cfg: dict, name: str) -> None:
    log_dir = resolve_path(cfg, "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GridSentinel AI continuous IEEE 30-bus simulator")
    ap.add_argument("--mode", choices=["live", "replay"], default="live")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--interval", type=float, default=None, help="sampling interval in seconds")
    ap.add_argument("--steps", type=int, default=None, help="stop after N steps (default: run until stopped)")
    ap.add_argument("--fast", action="store_true", help="no real-time pacing (dataset generation)")
    ap.add_argument("--source-run", help="run ID to replay (replay mode)")
    ap.add_argument("--rate", type=float, default=None, help="replay playback rate")
    ap.add_argument("--loop", action="store_true", help="loop the replay")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)

    overrides = list(args.set)
    if args.interval is not None:
        overrides.append(f"simulation.interval_s={args.interval}")
    cfg = load_settings(args.config, overrides)
    setup_logging(cfg, "simulator")
    sim = ContinuousSimulator(cfg, mode=args.mode, purpose="live", seed=args.seed, fast=args.fast,
                              max_steps=args.steps)
    sim.install_signal_handlers()
    if args.mode == "replay":
        if not args.source_run:
            ap.error("--source-run is required in replay mode")
        sim.replay(args.source_run, args.rate, args.loop)
    else:
        sim.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
