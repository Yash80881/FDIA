"""Start the complete GridSentinel AI platform from one terminal.

    python run_all.py                 # simulator + API + dashboard
    python run_all.py --train-if-missing
    python run_all.py --no-simulator  # e.g. when replaying from another terminal

Each service runs in its own Python process (a crash in one cannot corrupt
another; all share the SQLite database in WAL mode).  Ctrl+C stops the
simulator first (it commits and marks the run 'stopped'), then the dashboard
and the API.  Logs: logs/simulator.log, logs/api.log, logs/dashboard.log.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import load_settings, resolve_path  # noqa: E402


def wait_http(url: str, timeout: float = 60.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status < 500:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def spawn(name: str, module: str, extra: list[str]) -> subprocess.Popen:
    kw = {}
    if os.name == "nt":   # own process group so Ctrl+C is delivered by us, in order
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    print(f"[run_all] starting {name}: python -m {module} {' '.join(extra)}")
    return subprocess.Popen([sys.executable, "-m", module, *extra], cwd=ROOT, **kw)


def stop(name: str, proc: subprocess.Popen, timeout: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    print(f"[run_all] stopping {name} ...")
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
        proc.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-simulator", action="store_true")
    ap.add_argument("--train-if-missing", action="store_true",
                    help="generate the dataset and train models first if no models exist")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_settings()
    models = resolve_path(cfg, "models") / "detection_rf.joblib"
    if not models.exists():
        if args.train_if_missing:
            print("[run_all] no trained models - generating dataset and training (takes several minutes)")
            if subprocess.call([sys.executable, "cli.py", "generate-dataset"], cwd=ROOT) != 0:
                return 1
            if subprocess.call([sys.executable, "cli.py", "train"], cwd=ROOT) != 0:
                return 1
        else:
            print("[run_all] WARNING: no trained models found - the dashboard will show measurements and physics "
                  "verification but no AI predictions. Run 'python cli.py generate-dataset' and "
                  "'python cli.py train' (or pass --train-if-missing).")
    procs: list[tuple[str, subprocess.Popen]] = []
    try:
        api = spawn("API", "backend.main", [])
        procs.append(("API", api))
        api_url = f"http://{cfg['api']['host']}:{cfg['api']['port']}/health"
        if not wait_http(api_url):
            print("[run_all] API did not become healthy - see logs/api.log")
            return 1
        if not args.no_simulator:
            sim_args = ["--seed", str(args.seed)] if args.seed is not None else []
            procs.insert(0, ("simulator", spawn("simulator", "simulator.continuous_simulator", sim_args)))
        procs.append(("dashboard", spawn("dashboard", "dashboard.app", [])))
        print(f"[run_all] dashboard: http://{cfg['dashboard']['host']}:{cfg['dashboard']['port']}  "
              f"API docs: http://{cfg['api']['host']}:{cfg['api']['port']}/docs   (Ctrl+C to stop)")
        while True:
            for name, p in procs:
                if p.poll() is not None:
                    print(f"[run_all] {name} exited with code {p.returncode}; shutting down the rest")
                    raise KeyboardInterrupt
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        for name, p in procs:      # simulator first, so its final commit happens while the API still runs
            stop(name, p)
        print("[run_all] all services stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
