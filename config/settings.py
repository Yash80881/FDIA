"""Central configuration loader for GridSentinel AI.

All modules obtain their parameters through :func:`load_settings` so that no
values are hard-coded throughout the code base.  The YAML file
``config/simulation_config.yaml`` is the single source of truth; it can be
replaced via the ``GRIDSENTINEL_CONFIG`` environment variable and individual
keys can be overridden with dotted ``section.key=value`` strings.
"""
from __future__ import annotations

import copy
import json
import os
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import yaml
    _yaml_load = yaml.safe_load
except ImportError:
    from ruamel.yaml import YAML as _RuamelYAML
    _ruamel = _RuamelYAML(typ="safe", pure=True)
    def _yaml_load(src):
        return _ruamel.load(src)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "simulation_config.yaml"

# Silence joblib/loky's "could not find the number of physical cores" warning on
# Windows by telling it explicitly how many workers it may use.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


class ConfigError(RuntimeError):
    """Raised when the configuration file is missing or invalid."""


def _parse_scalar(text: str) -> Any:
    """Interpret a CLI override value using YAML rules (numbers, bools, lists)."""
    return _yaml_load(text)


def set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set ``cfg['a']['b']['c'] = value`` for ``dotted_key='a.b.c'``."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise ConfigError(f"Unknown configuration section '{part}' in '{dotted_key}'")
        node = node[part]
    if parts[-1] not in node:
        raise ConfigError(f"Unknown configuration key '{dotted_key}'")
    node[parts[-1]] = value


def get_dotted(cfg: Mapping, dotted_key: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def apply_overrides(cfg: dict, overrides: Mapping[str, Any] | Iterable[str] | None) -> dict:
    """Return a copy of ``cfg`` with overrides applied.

    ``overrides`` is either a mapping ``{"noise.voltage_pu": 0.004}`` or an
    iterable of ``"noise.voltage_pu=0.004"`` strings.
    """
    out = copy.deepcopy(cfg)
    if not overrides:
        return out
    if isinstance(overrides, Mapping):
        items = overrides.items()
    else:
        items = []
        for text in overrides:
            if "=" not in text:
                raise ConfigError(f"Override '{text}' must have the form section.key=value")
            key, value = text.split("=", 1)
            items.append((key.strip(), _parse_scalar(value)))
    for key, value in items:
        set_dotted(out, key, value)
    return out


def load_settings(path: str | os.PathLike | None = None,
                  overrides: Mapping[str, Any] | Iterable[str] | None = None) -> dict:
    """Load the YAML configuration, apply overrides and resolve paths."""
    cfg_path = Path(path or os.environ.get("GRIDSENTINEL_CONFIG") or DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        raise ConfigError(f"Configuration file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = _yaml_load(fh.read())
    if not isinstance(cfg, dict):
        raise ConfigError(f"Configuration file {cfg_path} is empty or invalid")
    cfg = apply_overrides(cfg, overrides)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve_path(cfg: Mapping, key: str) -> Path:
    """Absolute path for ``cfg['paths'][key]`` (relative paths -> project root)."""
    raw = cfg["paths"][key]
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def ensure_directories(cfg: Mapping) -> None:
    for key in ("exports", "processed", "raw", "models", "logs", "results"):
        resolve_path(cfg, key).mkdir(parents=True, exist_ok=True)
    resolve_path(cfg, "database").parent.mkdir(parents=True, exist_ok=True)


def config_hash(cfg: Mapping) -> str:
    """Stable short hash of the configuration (stored with runs and models)."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    payload = json.dumps(clean, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]
