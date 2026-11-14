"""Optional model D - attack severity estimation (RandomForestRegressor).

Target: the simulator's noise-normalised severity (RMS perturbation of the
attacked sensors in units of their noise standard deviation; see
``attacks/attack_scheduler.py``).  Severity is a *target only*; it is never
an input.  Trained and evaluated on truly attacked snapshots; reported with
MAE, RMSE, R^2 and Spearman rank correlation.  The estimate is meaningful only
when an attack is actually present (the API reports it only after detection).
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.common import rf_regressor
from ml.feature_engineering import feature_columns, FEATURE_GROUPS

log = logging.getLogger("gridsentinel.train.severity")


def train_severity(cfg: dict, splits: dict, groups: dict, feature_groups=FEATURE_GROUPS) -> tuple[dict, dict]:
    p = cfg["ml"]["severity"]
    cols = feature_columns(groups, tuple(feature_groups))
    tr = splits["train"]
    m = tr.attacked
    model = rf_regressor(p, cfg["ml"]["n_jobs"], cfg["project"]["random_seed"])
    model.fit(tr.X.loc[m, cols].to_numpy(), tr.y.loc[m, "attack_severity"].to_numpy())
    metrics = {"target": "noise-normalised RMS perturbation (sigma units)", "splits": {}}
    for name, s in splits.items():
        if name == "train" or s.attacked.sum() < 2:
            continue
        att = s.attacked
        y = s.y.loc[att, "attack_severity"].to_numpy()
        pred = model.predict(s.X.loc[att, cols].to_numpy())
        metrics["splits"][name] = {
            "n": int(att.sum()),
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred)),
            "spearman": float(spearmanr(y, pred).statistic),
            "target_mean": float(y.mean()), "target_std": float(y.std()),
        }
    log.info("severity: test MAE=%.3f R2=%.3f", metrics["splits"].get("test", {}).get("mae", float("nan")),
             metrics["splits"].get("test", {}).get("r2", float("nan")))
    bundle = {"model": model, "features": cols, "meta": {"model": "severity", "params": p}}
    return bundle, metrics
