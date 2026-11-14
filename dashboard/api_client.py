"""Thin HTTP client used by the dashboard (short timeouts, explicit errors)."""
from __future__ import annotations

import requests


class APIUnavailable(RuntimeError):
    pass


class APIClient:
    def __init__(self, base_url: str, timeout: float = 4.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def get(self, path: str, **params) -> dict:
        params = {k: v for k, v in params.items() if v is not None}
        try:
            r = self.session.get(f"{self.base}{path}", params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise APIUnavailable(f"API not reachable at {self.base} ({type(exc).__name__}). "
                                 "Start it with: python cli.py api") from exc
        if r.status_code >= 400:
            try:
                body = r.json()
                msg = body.get("detail") or body
                hint = body.get("hint")
            except ValueError:
                msg, hint = r.text[:200], None
            raise APIUnavailable(f"{path}: {msg}" + (f" - {hint}" if hint else ""))
        return r.json()

    def safe(self, path: str, **params) -> tuple[dict | None, str | None]:
        try:
            return self.get(path, **params), None
        except APIUnavailable as exc:
            return None, str(exc)
