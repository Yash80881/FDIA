"""GridSentinel AI command-line interface (one entry point for every task).

    python cli.py check                  verify interpreter, PYPOWER and the IEEE 30-bus case
    python cli.py simulate               start the live simulator (Ctrl+C to stop)
    python cli.py stop                   ask a running simulator to stop gracefully
    python cli.py replay --source-run ID replay a recorded run at --rate x
    python cli.py generate-dataset       independent train/validation/test/stress runs
    python cli.py validate [--run ID]    dataset validation report
    python cli.py export --run ID        export a run to CSV (never overwrites)
    python cli.py train                  train + evaluate all Random Forest models
    python cli.py predict --run ID       run the models on a stored snapshot
    python cli.py api                    start the FastAPI backend
    python cli.py dashboard              start the Dash dashboard
    python cli.py experiments            research experiments / ablations / latency
    python cli.py test                   run the automated test-suite
    python cli.py network                write the topology figure (results/ieee30_network.html)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def cmd_check(args) -> int:
    import platform
    from importlib.metadata import version
    print(f"Python      : {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform    : {platform.platform()}")
    if sys.version_info[:2] != (3, 11):
        print("WARNING     : the project was developed and tested with Python 3.11")
    try:
        print(f"PYPOWER     : {version('PYPOWER')}")
        from pypower.case30 import case30          # noqa: F401
        from pypower.api import runpf, ppoption    # noqa: F401
    except Exception as exc:
        print(f"ERROR       : PYPOWER not importable ({exc}). pip install PYPOWER==5.1.21")
        return 1
    for pkg in ("numpy", "scipy", "pandas", "scikit-learn", "fastapi", "dash", "plotly"):
        try:
            print(f"{pkg:<12}: {version(pkg)}")
        except Exception:
            print(f"{pkg:<12}: MISSING")
    from simulator.grid_model import GridModel
    from simulator.power_flow import run_ac_power_flow
    g = GridModel()
    pf = run_ac_power_flow(g.new_case())
    s = g.summary()
    print(f"Case        : {s['n_bus']} buses, {s['n_branch']} branches, {s['n_gen']} generators, ref bus {s['ref_bus']}")
    print(f"Base load   : {s['base_total_load_mw']} MW / {s['base_total_load_mvar']} MVAr; generators at {s['generator_buses']}")
    print(f"Power flow  : converged, Vmin {pf.vm_pu.min():.4f} pu, losses {pf.pg_mw.sum() - pf.pd_mw.sum():.3f} MW")
    return 0


def cmd_stop(args) -> int:
    from config.settings import load_settings, resolve_path
    flag = resolve_path(load_settings(None, args.set), "stop_flag")
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("stop requested\n")
    print(f"Stop requested ({flag}). The simulator stops within ~10 steps and closes the database cleanly.")
    return 0


def _forward(module: str, argv: list[str]) -> int:
    return subprocess.call([sys.executable, "-m", module, *argv], cwd=ROOT)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "check":
        return cmd_check(None)
    if cmd == "stop":
        ap = argparse.ArgumentParser()
        ap.add_argument("--set", action="append", default=[])
        return cmd_stop(ap.parse_args(rest))
    routes = {
        "simulate": ("simulator.continuous_simulator", []),
        "replay": ("simulator.continuous_simulator", ["--mode", "replay"]),
        "generate-dataset": ("ml.dataset", ["generate"]),
        "list-runs": ("ml.dataset", ["list"]),
        "validate": ("storage.dataset_validation", [] if rest else ["--all"]),
        "train": ("ml.train_all", []),
        "predict": ("ml.inference", []),
        "api": ("backend.main", []),
        "dashboard": ("dashboard.app", []),
        "experiments": ("experiments.run_all_experiments", []),
        "network": ("simulator.network_visualization", []),
    }
    if cmd == "export":
        ap = argparse.ArgumentParser(prog="cli.py export")
        ap.add_argument("--run", action="append")
        ap.add_argument("--purpose")
        a = ap.parse_args(rest)
        if a.purpose:
            return _forward("ml.dataset", ["export", "--purpose", a.purpose])
        from config.settings import load_settings, resolve_path
        from storage.database import Database
        from storage.repository import Repository
        cfg = load_settings()
        repo = Repository(Database(resolve_path(cfg, "database")))
        for rid in a.run or []:
            print(repo.export_run_csv(rid, resolve_path(cfg, "exports")))
        return 0
    if cmd == "test":
        return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests", *rest], cwd=ROOT)
    if cmd not in routes:
        print(f"Unknown command '{cmd}'.\n{__doc__}")
        return 2
    module, prefix = routes[cmd]
    return _forward(module, prefix + rest)


if __name__ == "__main__":
    raise SystemExit(main())
