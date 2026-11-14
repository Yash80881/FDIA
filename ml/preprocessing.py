"""Split loading for training/evaluation (run-group based, leakage-safe).

``load_splits`` returns, for every dataset purpose (train / validation / test /
stress_*), the engineered features of *whole independent runs* and their
labels.  Training code only ever fits on ``train``; thresholds and model
selection use ``validation``; reported results use ``test`` (and the stress
runs).  When only a single run exists, :func:`ml.dataset.chronological_split`
provides a gap-separated past/future split instead.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ml.dataset import load_features, runs_for
from simulator import schema as S

log = logging.getLogger("gridsentinel.preprocessing")


class Split:
    def __init__(self, name: str, X: pd.DataFrame, y: pd.DataFrame, run_ids: list[str]):
        self.name, self.X, self.y, self.run_ids = name, X, y, run_ids

    @property
    def attacked(self) -> np.ndarray:
        return self.y["attack_active"].to_numpy() == 1

    @property
    def Y_bus(self) -> np.ndarray:
        return self.y[S.ATTACK_BUS_COLUMNS].to_numpy().astype(int)

    def __len__(self):
        return len(self.X)


def load_splits(cfg: dict, purposes: list[str] | None = None) -> tuple[dict[str, Split], dict, dict]:
    from storage.database import Database
    from storage.repository import Repository
    from config.settings import resolve_path
    repo = Repository(Database(resolve_path(cfg, "database")))
    all_purposes = sorted({r["purpose"] for r in repo.list_runs() if r["purpose"] != "live"})
    purposes = purposes or all_purposes
    splits = {}
    by_purpose = {p: runs_for(cfg, p) for p in purposes}
    all_ids = [r for p in purposes for r in by_purpose[p]]
    if not all_ids:
        raise RuntimeError("No training runs found. Generate the dataset first: python -m ml.dataset generate")
    X_all, y_all, groups, local = load_features(cfg, all_ids)     # computed in parallel, cached per run
    for p in purposes:
        rids = by_purpose[p]
        if not rids:
            continue
        m = y_all["run_id"].isin(rids).to_numpy()
        splits[p] = Split(p, X_all.loc[m].reset_index(drop=True), y_all.loc[m].reset_index(drop=True), rids)
        log.info("split %-26s runs=%d rows=%d attacked=%.1f%%", p, len(rids), len(splits[p]),
                 100 * splits[p].attacked.mean())
    if "train" not in splits:
        raise RuntimeError("No training runs found. Generate the dataset first: python -m ml.dataset generate")
    # The run-group split must be disjoint
    seen = {}
    for p, s in splits.items():
        for r in s.run_ids:
            if r in seen:
                raise RuntimeError(f"Run {r} appears in both {seen[r]} and {p}")
            seen[r] = p
    return splits, groups, local


def check_distribution(split: Split) -> dict:
    """Training-data distribution report (per-type, per-bus positives)."""
    y = split.y
    return {
        "rows": int(len(y)),
        "attack_fraction": float(y["attack_active"].mean()),
        "attack_type_rows": y["attack_type"].value_counts().to_dict(),
        "per_bus_positives": {c.replace("attack_bus_", ""): int(y[c].sum()) for c in S.ATTACK_BUS_COLUMNS},
    }
