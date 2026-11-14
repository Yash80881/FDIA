"""Model A - Random Forest FDIA detection (NORMAL / ATTACK + probability).

Fit on the training runs, calibrate the decision threshold on the validation
runs (maximum F1, or a fixed value from the config), report on the independent
test runs and the stress-test runs.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from ml.common import rf_classifier, binary_metrics, best_f1_threshold
from ml.feature_engineering import feature_columns, FEATURE_GROUPS

log = logging.getLogger("gridsentinel.train.detection")


def train_detection(cfg: dict, splits: dict, groups: dict, feature_groups=FEATURE_GROUPS,
                    params: dict | None = None) -> tuple[dict, dict]:
    p = params or cfg["ml"]["detection"]
    seed = cfg["project"]["random_seed"]
    cols = feature_columns(groups, tuple(feature_groups))
    tr = splits["train"]
    model = rf_classifier(p, cfg["ml"]["n_jobs"], seed)
    t0 = time.perf_counter()
    model.fit(tr.X[cols].to_numpy(), tr.y["attack_active"].to_numpy())
    fit_s = time.perf_counter() - t0

    thr_cfg = cfg["ml"]["detection_threshold"]
    va = splits.get("validation")
    if thr_cfg == "calibrate" and va is not None:
        prob_va = model.predict_proba(va.X[cols].to_numpy())[:, 1]
        threshold = best_f1_threshold(va.y["attack_active"], prob_va)
    else:
        threshold = float(thr_cfg) if thr_cfg != "calibrate" else 0.5

    metrics = {"threshold": threshold, "fit_seconds": fit_s, "n_features": len(cols),
               "feature_groups": list(feature_groups), "train_rows": len(tr), "splits": {}}
    probs = {}
    for name, s in splits.items():
        prob = model.predict_proba(s.X[cols].to_numpy())[:, 1]
        probs[name] = prob
        if name == "train":
            continue
        metrics["splits"][name] = binary_metrics(s.y["attack_active"], prob >= threshold, prob)
        # recall per attack type (which attacks are hard to detect)
        per_type = {}
        for t in sorted(s.y["attack_type"].unique()):
            if t == "normal":
                continue
            m = (s.y["attack_type"] == t).to_numpy()
            per_type[t] = float(np.mean(prob[m] >= threshold))
        metrics["splits"][name]["recall_by_attack_type"] = per_type
    imp = sorted(zip(cols, model.feature_importances_), key=lambda x: -x[1])[:25]
    metrics["top_features"] = [{"feature": f, "importance": float(v)} for f, v in imp]
    bundle = {"model": model, "features": cols, "threshold": threshold,
              "meta": {"model": "detection", "params": p}}
    log.info("detection: threshold=%.3f test F1=%.4f", threshold,
             metrics["splits"].get("test", {}).get("f1", float("nan")))
    return bundle, {"metrics": metrics, "probs": probs}
