"""GridSentinel AI dashboard (Dash + Plotly).

    python -m dashboard.app        (or: python cli.py dashboard)
    open http://127.0.0.1:8050

The dashboard reads everything from the FastAPI backend (it never touches the
CSV files or recomputes models) and refreshes every ``dashboard.refresh_ms``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dash import Dash

from config.settings import load_settings
from dashboard.api_client import APIClient
from dashboard.callbacks import register_callbacks
from dashboard.layout import build_layout


def create_dashboard(cfg: dict, api_url: str | None = None) -> Dash:
    api_url = api_url or f"http://{cfg['api']['host']}:{cfg['api']['port']}"
    app = Dash(__name__, title="GridSentinel AI", update_title=None, suppress_callback_exceptions=True,
               assets_folder=str(Path(__file__).parent / "assets"))
    app.layout = build_layout(int(cfg["dashboard"]["refresh_ms"]), int(cfg["dashboard"].get("tab_refresh_ms", 5000)))
    register_callbacks(app, APIClient(api_url), cfg)
    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GridSentinel AI dashboard")
    ap.add_argument("--api", default=None, help="API base URL (default from config)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = load_settings(None, args.set)
    from simulator.continuous_simulator import setup_logging
    setup_logging(cfg, "dashboard")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app = create_dashboard(cfg, args.api)
    app.run(host=cfg["dashboard"]["host"], port=int(cfg["dashboard"]["port"]), debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
