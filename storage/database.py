"""SQLite storage layer (WAL mode, indexed by run ID, step and timestamp).

One wide ``measurements`` table holds every snapshot (true + received
measurements + labels).  The simulator is the only writer of measurements;
the backend writes predictions/alerts; readers use short read transactions,
so WAL mode lets the API and dashboard read while the simulator writes.

A ``meta`` table records the measurement schema version.  If an existing
database was created with a different column structure we refuse to append
to it instead of mixing incompatible columns.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from simulator.schema import ALL_COLUMNS, TEXT_COLUMNS, INTEGER_COLUMNS, SCHEMA_VERSION

log = logging.getLogger(__name__)


class SchemaMismatchError(RuntimeError):
    pass


def _col_type(c: str) -> str:
    if c in TEXT_COLUMNS:
        return "TEXT"
    if c in INTEGER_COLUMNS:
        return "INTEGER"
    return "REAL"


def _ddl() -> list[str]:
    cols = ",\n  ".join(f'"{c}" {_col_type(c)}' for c in ALL_COLUMNS)
    return [
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)",
        """CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, mode TEXT, purpose TEXT, seed INTEGER,
            config_json TEXT, config_hash TEXT, schema_version INTEGER,
            started_at TEXT, ended_at TEXT, status TEXT, last_step INTEGER,
            last_heartbeat REAL, interval_s REAL, source_run_id TEXT,
            pf_failures INTEGER DEFAULT 0, message TEXT)""",
        f"CREATE TABLE IF NOT EXISTS measurements (\n  {cols},\n  PRIMARY KEY (run_id, step))",
        "CREATE INDEX IF NOT EXISTS idx_meas_run_step ON measurements(run_id, step)",
        "CREATE INDEX IF NOT EXISTS idx_meas_timestamp ON measurements(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_meas_attack ON measurements(run_id, attack_active)",
        """CREATE TABLE IF NOT EXISTS attack_events (
            attack_id TEXT PRIMARY KEY, run_id TEXT, attack_type TEXT,
            start_step INTEGER, end_step INTEGER, duration_steps INTEGER,
            start_time TEXT, buses TEXT, channels TEXT, params TEXT,
            mean_severity REAL, max_severity REAL)""",
        "CREATE INDEX IF NOT EXISTS idx_events_run ON attack_events(run_id, start_step)",
        """CREATE TABLE IF NOT EXISTS predictions (
            run_id TEXT, step INTEGER, payload TEXT, created_at REAL,
            PRIMARY KEY (run_id, step))""",
        """CREATE TABLE IF NOT EXISTS alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT,
            start_step INTEGER, end_step INTEGER, start_time TEXT, end_time TEXT,
            status TEXT, predicted_type TEXT, predicted_buses TEXT,
            max_probability REAL, est_severity REAL, physics_status TEXT,
            recommended_action TEXT, ground_truth_attack_ids TEXT)""",
        "CREATE INDEX IF NOT EXISTS idx_alerts_run ON alerts(run_id, start_step)",
        """CREATE TABLE IF NOT EXISTS monitor_stats (
            run_id TEXT PRIMARY KEY, processed INTEGER, dropped INTEGER,
            last_processed_step INTEGER, mean_latency_ms REAL, updated_at REAL)""",
        """CREATE TABLE IF NOT EXISTS hardware_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT, received_at REAL, device_id TEXT,
            topic TEXT, payload TEXT)""",
    ]


def connect(db_path: str | Path, readonly: bool = False, timeout: float = 30.0) -> sqlite3.Connection:
    """Open a connection with WAL journaling and sane pragmas."""
    db_path = Path(db_path)
    if readonly:
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}. Start the simulator first.")
        uri = f"file:{db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=timeout, check_same_thread=False,
                               isolation_level=None)
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: we issue BEGIN IMMEDIATE ourselves so that a writer
        # takes the write lock up front (and waits via busy_timeout) instead of
        # failing with SQLITE_BUSY when upgrading a stale WAL read snapshot.
        conn = sqlite3.connect(str(db_path), timeout=timeout, check_same_thread=False,
                               isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
    conn.row_factory = sqlite3.Row
    return conn


def initialize(conn: sqlite3.Connection) -> None:
    """Create tables if needed and verify the schema version."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        for stmt in _ddl():
            conn.execute(stmt)
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))
        elif int(row["value"]) != SCHEMA_VERSION:
            raise SchemaMismatchError(
                f"Database schema version {row['value']} != code schema version {SCHEMA_VERSION}. "
                "Use a new database file (paths.database) - the existing data is left untouched.")
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(measurements)")}
    missing = [c for c in ALL_COLUMNS if c not in existing]
    if missing:
        raise SchemaMismatchError(f"measurements table is missing columns {missing[:5]}...")


class Database:
    """Thread-safe wrapper around one SQLite connection."""

    def __init__(self, db_path: str | Path, readonly: bool = False):
        self.path = Path(db_path)
        self.readonly = readonly
        self.conn = connect(self.path, readonly=readonly)
        self.lock = threading.RLock()
        if not readonly:
            initialize(self.conn)

    @contextmanager
    def transaction(self):
        """Write transaction (BEGIN IMMEDIATE ... COMMIT); joins an open one."""
        with self.lock:
            if self.conn.in_transaction:
                yield self.conn
                return
            self._begin_immediate()
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")

    def begin(self) -> None:
        with self.lock:
            if not self.conn.in_transaction:
                self._begin_immediate()

    def _begin_immediate(self, max_wait_s: float = 120.0) -> None:
        """BEGIN IMMEDIATE with retries.  SQLite can return SQLITE_BUSY without
        invoking the busy handler in some WAL situations (e.g. while another
        connection checkpoints), so we retry with back-off instead of failing."""
        deadline = time.monotonic() + max_wait_s
        delay = 0.01
        while True:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) and "busy" not in str(exc):
                    raise
                if time.monotonic() > deadline:
                    raise
                log.debug("database busy, retrying BEGIN IMMEDIATE")
                time.sleep(delay)
                delay = min(delay * 2, 0.5)

    def commit(self) -> None:
        with self.lock:
            if self.conn.in_transaction:
                self.conn.execute("COMMIT")

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        with self.lock:
            return self.conn.execute(sql, params)

    def query(self, sql: str, params=()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self.lock:
            try:
                if not self.readonly:
                    self.commit()
            finally:
                self.conn.close()
