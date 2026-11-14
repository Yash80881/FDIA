"""Attack-type classification experiments: per-type report and confusion matrix
on each independent evaluation split, plus a feature-group ablation."""
from __future__ import annotations

import logging

import pandas as pd

from ml.common import rf_classifier
from ml.feature_engineering import feature_columns
from ml.train_attack_type import _report

log = logging.getLogger("gridsentinel.exp.types")


def per_split(type_bundle, splits) -> pd.DataFrame:
    model, cols = type_bundle["model"], type_bundle["features"]
    rows = []
    for name in ("test", "stress_high_noise", "stress_heavy_load", "stress_low_intensity"):
        if name not in splits:
            continue
        s = splits[name]
        att = s.attacked
        pred = model.predict(s.X.loc[att, cols].to_numpy())
        rep = _report(s.y.loc[att, "attack_type"].to_numpy(), pred)
        row = {"split": name, "n": rep["n"], "accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"]}
        for t in ("additive", "scaling", "replay", "coordinated"):
            row[f"f1_{t}"] = rep["classification_report"][t]["f1-score"]
        rows.append(row)
    return pd.DataFrame(rows)


def ablation(cfg, splits, groups, n_estimators: int) -> pd.DataFrame:
    tr = splits["train"]
    m = tr.attacked
    rows = []
    for name, sel in (("bus only", ("bus",)), ("bus + line", ("bus", "line")),
                      ("bus + line + temporal", ("bus", "line", "temporal")),
                      ("full", ("bus", "line", "electrical", "temporal"))):
        cols = feature_columns(groups, sel)
        model = rf_classifier(dict(cfg["ml"]["attack_type"], n_estimators=n_estimators), cfg["ml"]["n_jobs"],
                              cfg["project"]["random_seed"])
        model.fit(tr.X.loc[m, cols].to_numpy(), tr.y.loc[m, "attack_type"].to_numpy())
        s = splits["test"]
        att = s.attacked
        rep = _report(s.y.loc[att, "attack_type"].to_numpy(), model.predict(s.X.loc[att, cols].to_numpy()))
        rows.append({"features": name, "accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"]})
        log.info("type ablation %s macro-F1 %.4f", name, rep["macro_f1"])
    return pd.DataFrame(rows)
