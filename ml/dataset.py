"""Research dataset generation and loading (independent simulation runs).

Each dataset run is an independent simulation (own seed -> own load
trajectory, noise and attack sequence) stored in SQLite with a ``purpose``
tag.  Splits are by *run*, never by random rows, so adjacent correlated
snapshots of one attack can never sit on both sides of a split.

    python -m ml.dataset generate            # train/validation/test + stress runs
    python -m ml.dataset list
    python -m ml.dataset export --purpose test
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import load_settings, resolve_path, apply_overrides, config_hash
from simulator import schema as S

log = logging.getLogger("gridsentinel.dataset")

FEATURE_VERSION = 2          # 2 = base features; physics features add 'p' suffix. bump when feature engineering changes (invalidates the cache)
SPLIT_SEED_BASE = {"train": 1000, "validation": 2000, "test": 3000}
STRESS_SEED_BASE = 4000

LABEL_AND_META = (["run_id", "step", "attack_active", "attack_id", "attack_type", "attack_severity",
                   "attacked_bus_count", "attack_step_in_event", "scenario", "total_load_p_mw"]
                  + S.ATTACK_BUS_COLUMNS)


def dataset_plan(cfg: dict) -> list[dict]:
    d = cfg["dataset"]
    plan = []
    for purpose, key in (("train", "train_runs"), ("validation", "validation_runs"), ("test", "test_runs")):
        for i in range(d[key]):
            plan.append({"purpose": purpose, "seed": SPLIT_SEED_BASE[purpose] + i + 1, "overrides": {}})
    for j, (name, ov) in enumerate((d.get("stress_test") or {}).items()):
        plan.append({"purpose": f"stress_{name}", "seed": STRESS_SEED_BASE + j + 1, "overrides": dict(ov)})
    return plan


def _generate_one(cfg_path: str | None, base_overrides: list[str], item: dict, steps: int) -> str:
    from simulator.continuous_simulator import ContinuousSimulator, setup_logging
    cfg = load_settings(cfg_path, base_overrides)
    cfg = apply_overrides(cfg, item["overrides"])
    logging.basicConfig(level=logging.WARNING)
    sim = ContinuousSimulator(cfg, mode="live", purpose=item["purpose"], seed=item["seed"], fast=True,
                              max_steps=steps, tag=item["purpose"])
    return sim.run()


def generate(cfg: dict, base_overrides: list[str] | None = None, workers: int | None = None,
             steps: int | None = None, only: list[str] | None = None) -> list[str]:
    steps = int(steps or cfg["dataset"]["steps_per_run"])
    plan = [p for p in dataset_plan(cfg) if not only or p["purpose"] in only]
    workers = workers or min(4, len(plan))
    log.info("Generating %d independent runs x %d steps with %d workers", len(plan), steps, workers)
    ids = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_generate_one, cfg.get("_config_path"), base_overrides or [], p, steps): p
                for p in plan}
        for f in as_completed(futs):
            rid = f.result()
            log.info("finished %s (%s)", rid, futs[f]["purpose"])
            ids.append(rid)
    return ids


# ---------------------------------------------------------------- loading
def _repo(cfg):
    from storage.database import Database
    from storage.repository import Repository
    return Repository(Database(resolve_path(cfg, "database")))


def runs_for(cfg: dict, purpose: str) -> list[str]:
    return [r["run_id"] for r in _repo(cfg).list_runs(purpose) if r["status"] == "completed"]


def _features_for_run(db_path: str, run_id: str, noise: dict, window: int, cache: str,
                      physics: bool = False) -> str:
    """Compute and cache the features of one run (runs in a worker process)."""
    from ml.feature_engineering import FeatureBuilder
    from storage.database import Database
    from storage.repository import Repository
    repo = Repository(Database(db_path))
    df = repo.load_run(run_id, S.ML_INPUT_COLUMNS + LABEL_AND_META)
    repo.db.close()
    fs = FeatureBuilder(rolling_window=window, noise=noise, physics=physics).build(df)
    joblib.dump((fs.X.astype(np.float32), df[LABEL_AND_META].copy(), fs.groups, fs.bus_local), cache)
    return cache


def load_features(cfg: dict, run_ids: list[str], use_cache: bool = True, workers: int | None = None):
    """Return (X float32 DataFrame, labels DataFrame, FeatureSet metadata) for runs.

    Each run's features use the sensor-accuracy specification (``noise``
    section) the run was recorded with - the information an operator has
    about the installed instrument classes, exactly as in the WLS R matrix.
    """
    import hashlib
    import json
    from ml.feature_engineering import FeatureBuilder
    window = cfg["features"]["rolling_window"]
    physics = bool(cfg["features"].get("physics_features", False))
    cache_dir = resolve_path(cfg, "processed") / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo = _repo(cfg)
    jobs = []
    for rid in run_ids:
        run = repo.get_run(rid)
        noise = json.loads(run["config_json"])["noise"] if run and run.get("config_json") else cfg["noise"]
        nkey = hashlib.sha1(json.dumps(noise, sort_keys=True).encode()).hexdigest()[:8]
        cache = cache_dir / f"{rid}_w{window}_v{S.SCHEMA_VERSION}_f{FEATURE_VERSION}{'p' if physics else ''}_{nkey}.joblib"
        jobs.append((rid, noise, cache))
    todo = [j for j in jobs if not (use_cache and j[2].exists())]
    if todo:
        log.info("Computing features for %d runs (physics features: %s)", len(todo), physics)
        db_path = str(resolve_path(cfg, "database"))
        workers = max(1, min(len(todo), workers or os.cpu_count() or 1, 8))
        if workers == 1:
            for rid, noise, cache in todo:
                _features_for_run(db_path, rid, noise, window, str(cache), physics)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_features_for_run, [db_path] * len(todo), [j[0] for j in todo],
                            [j[1] for j in todo], [window] * len(todo), [str(j[2]) for j in todo],
                            [physics] * len(todo)))
    xs, ys, meta = [], [], None
    for rid, noise, cache in jobs:
        X, y, groups, local = joblib.load(cache)
        xs.append(X.reset_index(drop=True))
        ys.append(y.reset_index(drop=True))
        meta = (groups, local)
    if not xs:
        raise RuntimeError("No completed dataset runs found. Run: python -m ml.dataset generate")
    X = pd.concat(xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    return X, y, meta[0], meta[1]


def chronological_split(n_rows: int, test_fraction: float = 0.25, gap: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Single-run fallback: train on the past, test on the future, with a gap
    larger than the longest attack so no attack event spans both sides."""
    cut = int(n_rows * (1 - test_fraction))
    train = np.arange(0, max(cut - gap, 0))
    test = np.arange(cut, n_rows)
    return train, test


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GridSentinel dataset management")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--workers", type=int, default=None)
    g.add_argument("--steps", type=int, default=None)
    g.add_argument("--only", nargs="*", help="purposes to generate (e.g. train test)")
    sub.add_parser("list")
    e = sub.add_parser("export")
    e.add_argument("--purpose", required=True)
    for p in (g, e):
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = load_settings(None, getattr(args, "set", []))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "generate":
        ids = generate(cfg, getattr(args, "set", []), args.workers, args.steps, args.only)
        print("\n".join(sorted(ids)))
    elif args.cmd == "list":
        for r in _repo(cfg).list_runs():
            print(f"{r['run_id']:<55} {r['purpose']:<22} {r['status']:<11} steps={r['last_step'] + 1}")
    elif args.cmd == "export":
        repo = _repo(cfg)
        for rid in runs_for(cfg, args.purpose):
            print(repo.export_run_csv(rid, resolve_path(cfg, "exports")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
