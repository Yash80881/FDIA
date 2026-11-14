"""Data-access functions on top of :mod:`storage.database`.

All queries are indexed by (run_id, step); nothing ever loads a full CSV.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from storage.database import Database
from simulator.schema import ALL_COLUMNS, SCHEMA_VERSION, ATTACK_BUS_COLUMNS

_INSERT_SQL = "INSERT INTO measurements ({}) VALUES ({})".format(
    ", ".join(f'"{c}"' for c in ALL_COLUMNS), ", ".join("?" for _ in ALL_COLUMNS))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _py(v):
    if isinstance(v, (np.floating,)):
        v = float(v)
    elif isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None   # NaN (e.g. loading of an unrated line) -> SQL NULL
    return v


class Repository:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------ runs
    def create_run(self, run_id: str, mode: str, purpose: str, seed: int, cfg: dict,
                   cfg_hash: str, interval_s: float, source_run_id: str | None = None) -> None:
        clean_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        with self.db.transaction():
            self.db.conn.execute(
                "INSERT INTO runs(run_id, mode, purpose, seed, config_json, config_hash, schema_version,"
                " started_at, status, last_step, last_heartbeat, interval_s, source_run_id)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, mode, purpose, seed, json.dumps(clean_cfg, default=str), cfg_hash,
                 SCHEMA_VERSION, utc_now_iso(), "running", -1, time.time(), interval_s, source_run_id))

    def heartbeat(self, run_id: str, last_step: int, pf_failures: int = 0, commit: bool = True) -> None:
        with self.db.lock:
            self.db.begin()
            self.db.conn.execute("UPDATE runs SET last_step=?, last_heartbeat=?, pf_failures=? WHERE run_id=?",
                                 (last_step, time.time(), pf_failures, run_id))
            if commit:
                self.db.commit()

    def finish_run(self, run_id: str, status: str, message: str = "") -> None:
        with self.db.transaction():
            self.db.conn.execute("UPDATE runs SET status=?, ended_at=?, message=?, last_heartbeat=? WHERE run_id=?",
                                 (status, utc_now_iso(), message, time.time(), run_id))

    def get_run(self, run_id: str) -> dict | None:
        rows = self.db.query("SELECT * FROM runs WHERE run_id=?", (run_id,))
        return dict(rows[0]) if rows else None

    def list_runs(self, purpose: str | None = None) -> list[dict]:
        cols = ("run_id, mode, purpose, seed, config_hash, schema_version, started_at, ended_at, status,"
                " last_step, last_heartbeat, interval_s, source_run_id, pf_failures, message")
        if purpose:
            rows = self.db.query(f"SELECT {cols} FROM runs WHERE purpose=? ORDER BY started_at", (purpose,))
        else:
            rows = self.db.query(f"SELECT {cols} FROM runs ORDER BY started_at")
        return [dict(r) for r in rows]

    def latest_run(self, modes: tuple[str, ...] = ("live", "replay")) -> dict | None:
        q = ",".join("?" for _ in modes)
        rows = self.db.query(
            f"SELECT run_id, mode, purpose, status, last_step, last_heartbeat, interval_s, started_at, ended_at,"
            f" message, pf_failures FROM runs WHERE mode IN ({q}) ORDER BY last_heartbeat DESC LIMIT 1", modes)
        return dict(rows[0]) if rows else None

    def mark_stale_runs(self, older_than_s: float) -> int:
        """Runs still marked 'running' without heartbeat (e.g. killed process)."""
        with self.db.transaction():
            cur = self.db.conn.execute(
                "UPDATE runs SET status='interrupted', message='no heartbeat (process ended unexpectedly)'"
                " WHERE status='running' AND last_heartbeat < ?", (time.time() - older_than_s,))
            return cur.rowcount

    # ---------------------------------------------------------- measurements
    def insert_measurement(self, row: dict, commit: bool = True) -> None:
        values = [_py(row.get(c)) for c in ALL_COLUMNS]
        with self.db.lock:
            self.db.begin()
            self.db.conn.execute(_INSERT_SQL, values)
            if commit:
                self.db.commit()

    def commit(self) -> None:
        self.db.commit()

    def upsert_attack_event(self, run_id: str, record: dict, start_time: str,
                            mean_sev: float, max_sev: float, commit: bool = True) -> None:
        with self.db.lock:
            self.db.begin()
            self.db.conn.execute(
                "INSERT INTO attack_events(attack_id, run_id, attack_type, start_step, end_step, duration_steps,"
                " start_time, buses, channels, params, mean_severity, max_severity) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(attack_id) DO UPDATE SET end_step=excluded.end_step,"
                " duration_steps=excluded.duration_steps, mean_severity=excluded.mean_severity,"
                " max_severity=excluded.max_severity",
                (record["attack_id"], run_id, record["attack_type"], record["start_step"], record["end_step"],
                 record["duration_steps"], start_time, json.dumps(record["buses"]),
                 json.dumps(record["channels"]),
                 json.dumps({"params": record["params"], "bus_channels": record["bus_channels"],
                             "line_offsets": record["line_offsets"]}),
                 mean_sev, max_sev))
            if commit:
                self.db.commit()

    def latest_step(self, run_id: str) -> int | None:
        rows = self.db.query("SELECT MAX(step) AS s FROM measurements WHERE run_id=?", (run_id,))
        return rows[0]["s"] if rows and rows[0]["s"] is not None else None

    def get_rows(self, run_id: str, start_step: int, end_step: int,
                 columns: list[str] | None = None) -> pd.DataFrame:
        cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        with self.db.lock:
            return pd.read_sql_query(
                f"SELECT {cols} FROM measurements WHERE run_id=? AND step BETWEEN ? AND ? ORDER BY step",
                self.db.conn, params=(run_id, start_step, end_step))

    def get_row(self, run_id: str, step: int) -> dict | None:
        rows = self.db.query("SELECT * FROM measurements WHERE run_id=? AND step=?", (run_id, step))
        return dict(rows[0]) if rows else None

    def history(self, run_id: str, columns: list[str], limit: int) -> pd.DataFrame:
        cols = ", ".join(f'"{c}"' for c in dict.fromkeys(["step", "timestamp"] + columns))
        with self.db.lock:
            df = pd.read_sql_query(
                f"SELECT {cols} FROM measurements WHERE run_id=? ORDER BY step DESC LIMIT ?",
                self.db.conn, params=(run_id, int(limit)))
        return df.iloc[::-1].reset_index(drop=True)

    def load_run(self, run_id: str, columns: list[str] | None = None) -> pd.DataFrame:
        cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        with self.db.lock:
            return pd.read_sql_query(f"SELECT {cols} FROM measurements WHERE run_id=? ORDER BY step",
                                     self.db.conn, params=(run_id,))

    def attack_events(self, run_id: str | None = None, limit: int = 500) -> list[dict]:
        if run_id:
            rows = self.db.query("SELECT * FROM attack_events WHERE run_id=? ORDER BY start_step DESC LIMIT ?",
                                 (run_id, limit))
        else:
            rows = self.db.query("SELECT * FROM attack_events ORDER BY rowid DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            d = dict(r)
            d["buses"] = json.loads(d["buses"])
            d["channels"] = json.loads(d["channels"])
            d.pop("params", None)
            out.append(d)
        return out

    def attack_statistics(self, run_id: str) -> dict:
        rows = self.db.query(
            "SELECT attack_type, COUNT(*) AS n FROM measurements WHERE run_id=? GROUP BY attack_type", (run_id,))
        per_type_steps = {r["attack_type"]: r["n"] for r in rows}
        ev = self.db.query("SELECT attack_type, COUNT(*) AS n, AVG(duration_steps) AS d, AVG(mean_severity) AS s"
                           " FROM attack_events WHERE run_id=? GROUP BY attack_type", (run_id,))
        per_type_events = {r["attack_type"]: {"events": r["n"], "mean_duration_steps": r["d"],
                                              "mean_severity": r["s"]} for r in ev}
        sums = ", ".join(f'SUM("{c}") AS "{c}"' for c in ATTACK_BUS_COLUMNS)
        bus_row = self.db.query(f"SELECT {sums} FROM measurements WHERE run_id=?", (run_id,))
        per_bus = {c.replace("attack_bus_", ""): (bus_row[0][c] or 0) for c in ATTACK_BUS_COLUMNS} if bus_row else {}
        return {"steps_per_type": per_type_steps, "events_per_type": per_type_events,
                "attacked_steps_per_bus": per_bus,
                "total_events": int(sum(v["events"] for v in per_type_events.values()))}

    # ------------------------------------------------------------ predictions
    def get_prediction(self, run_id: str, step: int) -> dict | None:
        rows = self.db.query("SELECT payload FROM predictions WHERE run_id=? AND step=?", (run_id, step))
        return json.loads(rows[0]["payload"]) if rows else None

    def save_prediction(self, run_id: str, step: int, payload: dict) -> None:
        with self.db.transaction():
            self.db.conn.execute("INSERT OR REPLACE INTO predictions(run_id, step, payload, created_at)"
                                 " VALUES(?,?,?,?)", (run_id, step, json.dumps(payload, default=_json_default),
                                                       time.time()))

    def prediction_history(self, run_id: str, limit: int) -> list[dict]:
        rows = self.db.query("SELECT step, payload FROM predictions WHERE run_id=? ORDER BY step DESC LIMIT ?",
                             (run_id, limit))
        return [json.loads(r["payload"]) for r in rows][::-1]

    def update_monitor_stats(self, run_id: str, processed: int, dropped: int, last_step: int,
                             mean_latency_ms: float) -> None:
        with self.db.transaction():
            self.db.conn.execute(
                "INSERT INTO monitor_stats(run_id, processed, dropped, last_processed_step, mean_latency_ms,"
                " updated_at) VALUES(?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET processed=excluded.processed,"
                " dropped=excluded.dropped, last_processed_step=excluded.last_processed_step,"
                " mean_latency_ms=excluded.mean_latency_ms, updated_at=excluded.updated_at",
                (run_id, processed, dropped, last_step, mean_latency_ms, time.time()))

    def monitor_stats(self, run_id: str) -> dict | None:
        rows = self.db.query("SELECT * FROM monitor_stats WHERE run_id=?", (run_id,))
        return dict(rows[0]) if rows else None

    # ----------------------------------------------------------------- alerts
    def open_alert(self, run_id: str, step: int, ts: str, data: dict) -> int:
        with self.db.transaction():
            cur = self.db.conn.execute(
                "INSERT INTO alerts(run_id, start_step, end_step, start_time, end_time, status, predicted_type,"
                " predicted_buses, max_probability, est_severity, physics_status, recommended_action,"
                " ground_truth_attack_ids) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, step, step, ts, ts, "open", data["predicted_type"], json.dumps(data["predicted_buses"]),
                 data["probability"], data.get("est_severity"), data["physics_status"],
                 data["recommended_action"], json.dumps(data["gt_attack_ids"])))
            return int(cur.lastrowid)

    def update_alert(self, alert_id: int, fields: dict) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)
        vals = [json.dumps(v) if isinstance(v, (list, dict)) else v for v in fields.values()]
        with self.db.transaction():
            self.db.conn.execute(f"UPDATE alerts SET {sets} WHERE alert_id=?", (*vals, alert_id))

    def alerts(self, run_id: str | None, limit: int = 200) -> list[dict]:
        if run_id:
            rows = self.db.query("SELECT * FROM alerts WHERE run_id=? ORDER BY alert_id DESC LIMIT ?",
                                 (run_id, limit))
        else:
            rows = self.db.query("SELECT * FROM alerts ORDER BY alert_id DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            d = dict(r)
            d["predicted_buses"] = json.loads(d["predicted_buses"] or "[]")
            d["ground_truth_attack_ids"] = json.loads(d["ground_truth_attack_ids"] or "[]")
            out.append(d)
        return out

    # ----------------------------------------------------------- maintenance
    def prune_run(self, run_id: str, keep_last_steps: int) -> int:
        last = self.latest_step(run_id)
        if last is None:
            return 0
        cutoff = last - keep_last_steps
        with self.db.transaction():
            cur = self.db.conn.execute("DELETE FROM measurements WHERE run_id=? AND step<?", (run_id, cutoff))
            self.db.conn.execute("DELETE FROM predictions WHERE run_id=? AND step<?", (run_id, cutoff))
            return cur.rowcount

    def export_run_csv(self, run_id: str, out_dir: Path) -> Path:
        """Export one run to its own CSV (never overwrites an existing file)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{run_id}_schema_v{SCHEMA_VERSION}.csv"
        if path.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = out_dir / f"{run_id}_schema_v{SCHEMA_VERSION}_{stamp}.csv"
        df = self.load_run(run_id)
        df.to_csv(path, index=False)
        return path


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")
