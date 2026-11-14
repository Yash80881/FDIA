"""Latency / throughput benchmarks.

* per-snapshot inference latency (feature engineering + 4 models) on stored test snapshots
* physics verification latency (WLS + tests, with/without LNR identification)
* simulator step throughput (power flow + measurements + attacks, no I/O)
* end-to-end: run the simulator for N steps in real time with a monitor service,
  and count processed / dropped snapshots
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger("gridsentinel.exp.latency")


def _stats(ms: list[float]) -> dict:
    a = np.array(ms)
    return {"n": len(a), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}


def inference_latency(cfg, n: int = 200) -> dict:
    from ml.inference import GridSentinelPredictor
    from ml.dataset import runs_for
    from storage.database import Database
    from storage.repository import Repository
    from config.settings import resolve_path
    from backend.services import WINDOW_COLUMNS
    from physics.state_estimation import WLSStateEstimator
    from physics.verification import verify_snapshot
    pred = GridSentinelPredictor(cfg)
    repo = Repository(Database(resolve_path(cfg, "database")))
    rid = runs_for(cfg, "test")[0]
    se = WLSStateEstimator()
    lat, phys, phys_id, fetch = [], [], [], []
    for step in range(50, 50 + n):
        t0 = time.perf_counter()
        w = repo.get_rows(rid, step - pred.history_steps + 1, step, WINDOW_COLUMNS)
        fetch.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        pred.predict(w)
        lat.append((time.perf_counter() - t0) * 1000)
        row = repo.get_row(rid, step)
        t0 = time.perf_counter()
        verify_snapshot(se, row, cfg, identify=False)
        phys.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        r = verify_snapshot(se, row, cfg, identify=True)
        if r.get("bad_data_detected"):
            phys_id.append((time.perf_counter() - t0) * 1000)
    return {"db_window_fetch": _stats(fetch), "ai_inference": _stats(lat), "physics_wls": _stats(phys),
            "physics_wls_with_lnr_identification": _stats(phys_id) if phys_id else None}


def simulator_throughput(cfg, steps: int = 300) -> dict:
    from simulator.continuous_simulator import SimulationEngine
    eng = SimulationEngine(cfg, "bench", 99)
    t0 = time.perf_counter()
    for t in range(steps):
        eng.step(t, "", float(t))
    dt = time.perf_counter() - t0
    return {"steps": steps, "steps_per_second": steps / dt, "ms_per_step": dt / steps * 1000}


def end_to_end(cfg, tmp_db, steps: int = 40, interval: float = 0.25) -> dict:
    """Real-time simulator + monitor service on a scratch database."""
    from config.settings import apply_overrides
    from simulator.continuous_simulator import ContinuousSimulator
    from backend.services import MonitorService
    c = apply_overrides(cfg, {"paths.database": str(tmp_db), "simulation.interval_s": interval,
                              "api.monitor_poll_s": 0.1})
    svc = MonitorService(c)
    svc.start()
    sim = ContinuousSimulator(c, seed=777, max_steps=steps)
    th = threading.Thread(target=sim.run)
    t0 = time.perf_counter()
    th.start()
    th.join()
    deadline = time.time() + 30
    while time.time() < deadline and (svc.state["last_step"] or -1) < steps - 1:
        time.sleep(0.2)
    wall = time.perf_counter() - t0
    out = {"steps": steps, "interval_s": interval, "processed": svc.state["processed"],
           "dropped": svc.state["dropped"], "last_processed_step": svc.state["last_step"],
           "mean_processing_ms": float(np.mean(svc.state["latencies"])) if svc.state["latencies"] else None,
           "wall_seconds": wall}
    svc.stop()
    return out
