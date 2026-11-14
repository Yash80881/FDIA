"""Tests 20, 21, 31, 32: storage, dataset validation, missing values, stop handling."""
import sqlite3
import threading
import time

import numpy as np
import pandas as pd
import pytest

from config.settings import resolve_path
from simulator.continuous_simulator import ContinuousSimulator
from storage.database import Database, SchemaMismatchError
from storage.repository import Repository
from storage.dataset_validation import validate_dataframe
from simulator import schema as S


def _sim(cfg, steps=300, seed=3):
    sim = ContinuousSimulator(cfg, seed=seed, fast=True, max_steps=steps)
    return sim.run()


def test_store_and_validate(cfg):
    run_id = _sim(cfg)
    repo = Repository(Database(resolve_path(cfg, "database")))
    df = repo.load_run(run_id)
    assert len(df) == 300
    assert list(df.columns) == S.ALL_COLUMNS
    rep = validate_dataframe(df, cfg["noise"])
    assert rep["passed"], [c for c in rep["checks"] if not c[1]]
    run = repo.get_run(run_id)
    assert run["status"] == "completed" and run["last_step"] == 299
    assert repo.attack_events(run_id)


def test_wal_and_indexes(cfg):
    _sim(cfg, steps=20)
    conn = sqlite3.connect(resolve_path(cfg, "database"))
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    idx = {r[1] for r in conn.execute("PRAGMA index_list(measurements)")}
    assert "idx_meas_run_step" in idx and "idx_meas_timestamp" in idx


def test_multiple_runs_kept(cfg):
    a = _sim(cfg, steps=30, seed=1)
    b = _sim(cfg, steps=30, seed=2)
    repo = Repository(Database(resolve_path(cfg, "database")))
    assert {a, b} <= {r["run_id"] for r in repo.list_runs()}
    assert len(repo.load_run(a)) == 30 and len(repo.load_run(b)) == 30


def test_missing_values_detected(cfg):
    run_id = _sim(cfg, steps=40)
    df = Repository(Database(resolve_path(cfg, "database"))).load_run(run_id)
    df.loc[5, "V_meas_bus_3"] = np.nan
    rep = validate_dataframe(df, cfg["noise"])
    assert not rep["passed"]
    assert any(name == "missing values" and not ok for name, ok, _ in rep["checks"])


def test_inconsistent_labels_detected(cfg):
    run_id = _sim(cfg, steps=40)
    df = Repository(Database(resolve_path(cfg, "database"))).load_run(run_id)
    df.loc[3, "attack_bus_4"] = 1
    df.loc[3, "attack_active"] = 0
    assert not validate_dataframe(df, cfg["noise"])["passed"]


def test_csv_export_never_overwrites(cfg):
    run_id = _sim(cfg, steps=25)
    repo = Repository(Database(resolve_path(cfg, "database")))
    p1 = repo.export_run_csv(run_id, resolve_path(cfg, "exports"))
    time.sleep(1.1)
    p2 = repo.export_run_csv(run_id, resolve_path(cfg, "exports"))
    assert p1 != p2 and p1.exists() and p2.exists()
    assert len(pd.read_csv(p1)) == 25


def test_schema_mismatch_refused(cfg):
    path = resolve_path(cfg, "database")
    Database(path).close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(SchemaMismatchError):
        Database(path)


def test_graceful_stop_via_flag_and_signal(cfg):
    sim = ContinuousSimulator(cfg, seed=1, fast=False)
    sim.interval = 0.05
    t = threading.Thread(target=sim.run)
    t.start()
    time.sleep(1.0)
    resolve_path(cfg, "stop_flag").write_text("stop")
    t.join(timeout=10)
    assert not t.is_alive()
    repo = Repository(Database(resolve_path(cfg, "database")))
    run = repo.get_run(sim.run_id)
    assert run["status"] == "stopped"
    n = len(repo.load_run(sim.run_id))
    assert n > 3 and run["last_step"] == n - 1     # every produced step was committed

    sim2 = ContinuousSimulator(cfg, seed=2, fast=False)
    sim2.interval = 0.05
    t2 = threading.Thread(target=sim2.run)
    t2.start()
    time.sleep(0.6)
    sim2.request_stop()          # what the Ctrl+C / SIGTERM handler calls
    t2.join(timeout=10)
    assert not t2.is_alive()
    assert repo.get_run(sim2.run_id)["status"] == "stopped"


def test_replay_mode(cfg):
    src = _sim(cfg, steps=60)
    sim = ContinuousSimulator(cfg, seed=1, fast=True, max_steps=50)
    rid = sim.replay(src, rate=100)
    repo = Repository(Database(resolve_path(cfg, "database")))
    a, b = repo.load_run(src), repo.load_run(rid)
    assert len(b) == 50
    assert np.allclose(a["V_meas_bus_7"].iloc[:50], b["V_meas_bus_7"])
    assert (a["attack_active"].iloc[:50].to_numpy() == b["attack_active"].to_numpy()).all()
    assert repo.get_run(rid)["source_run_id"] == src
