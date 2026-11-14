"""Model C - attack-type classification (additive / scaling / replay / coordinated).

Trained on truly attacked training snapshots only.  The simulator's attack
type is the *target*; it never appears among the inputs.  Evaluated on
(a) truly attacked test snapshots (classifier alone) and (b) snapshots the
two-stage system actually forwards (detector alarm AND truly attacked).
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from ml.common import rf_classifier
from ml.feature_engineering import feature_columns, FEATURE_GROUPS
from simulator.schema import ATTACK_TYPES

log = logging.getLogger("gridsentinel.train.attack_type")


def _report(y_true, y_pred) -> dict:
    labels = [t for t in ATTACK_TYPES]
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True,
                                                        zero_division=0),
        "confusion_matrix": {"labels": labels,
                             "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
    }


def train_attack_type(cfg: dict, splits: dict, groups: dict, det_probs: dict, det_threshold: float,
                      feature_groups=FEATURE_GROUPS) -> tuple[dict, dict]:
    p = cfg["ml"]["attack_type"]
    cols = feature_columns(groups, tuple(feature_groups))
    tr = splits["train"]
    m = tr.attacked
    y_tr = tr.y.loc[m, "attack_type"].to_numpy()
    missing = set(ATTACK_TYPES) - set(y_tr)
    if missing:
        raise RuntimeError(f"Attack types missing from training data: {missing}")
    model = rf_classifier(p, cfg["ml"]["n_jobs"], cfg["project"]["random_seed"])
    model.fit(tr.X.loc[m, cols].to_numpy(), y_tr)
    metrics = {"train_distribution": dict(zip(*np.unique(y_tr, return_counts=True))), "splits": {}}
    metrics["train_distribution"] = {str(k): int(v) for k, v in metrics["train_distribution"].items()}
    for name, s in splits.items():
        if name == "train":
            continue
        att = s.attacked
        if att.sum() == 0:
            continue
        pred = model.predict(s.X.loc[att, cols].to_numpy())
        truth = s.y.loc[att, "attack_type"].to_numpy()
        det = det_probs[name][att] >= det_threshold
        metrics["splits"][name] = {
            "attacked_snapshots": _report(truth, pred),
            "two_stage_detected_snapshots": _report(truth[det], pred[det]),
            "class_distribution": {t: int((truth == t).sum()) for t in ATTACK_TYPES},
        }
    log.info("attack type: test macro-F1=%.4f",
             metrics["splits"].get("test", {}).get("attacked_snapshots", {}).get("macro_f1", float("nan")))
    bundle = {"model": model, "features": cols, "classes": list(model.classes_),
              "meta": {"model": "attack_type", "params": p}}
    return bundle, metrics
