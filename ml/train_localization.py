"""Model B - bus-level FDIA localisation (30 independent Random Forests).

Design choices addressing the previous project's "all buses predicted 0" issue:

* one RandomForestClassifier per bus with ``class_weight='balanced'``;
* each bus model sees *bus-local* features (its own measurements, its
  neighbours' V/angle, its incident-branch consistency features) plus global
  features, so it learns where an inconsistency originates;
* per-bus probability thresholds calibrated on the validation runs (max F1);
* buses with fewer than ``min_positive_examples`` training positives get no
  model and are reported as "insufficient data" instead of silently 0.

Two training regimes are compared - *attack_only* (fit on truly attacked
snapshots) and *all* (fit on every snapshot) - and the regime with the better
validation two-stage micro-F1 is kept.  Results are reported separately for

* oracle localisation: only truly attacked test snapshots (localiser alone);
* two-stage system: every test snapshot, detector first, localiser only when
  the detector raises an alarm (detection errors propagate).
"""
from __future__ import annotations

import logging
import time

import numpy as np

from ml.common import rf_classifier, best_f1_threshold, multilabel_metrics

log = logging.getLogger("gridsentinel.train.localization")


def _fit_regime(cfg, splits, local, regime: str, bus_numbers, params, global_cols=None):
    tr = splits["train"]
    mask = tr.attacked if regime == "attack_only" else np.ones(len(tr), bool)
    min_pos = cfg["ml"]["localization"]["min_positive_examples"]
    models, feats, info = {}, {}, {}
    for j, b in enumerate(bus_numbers):
        cols = global_cols if global_cols is not None else local[b]
        y = tr.Y_bus[mask, j]
        pos = int(y.sum())
        info[b] = {"train_positives": pos, "train_rows": int(mask.sum())}
        if pos < min_pos or pos == len(y):
            models[b], feats[b] = None, cols
            info[b]["status"] = "insufficient data"
            log.warning("bus %d: only %d positive training examples - no model", b, pos)
            continue
        m = rf_classifier(params, cfg["ml"]["n_jobs"], cfg["project"]["random_seed"] + b)
        m.fit(tr.X.loc[mask, cols].to_numpy(), y)
        models[b], feats[b] = m, cols
        info[b]["status"] = "trained"
    return models, feats, info


def predict_bus_probs(models, feats, X, bus_numbers) -> np.ndarray:
    P = np.zeros((len(X), len(bus_numbers)))
    for j, b in enumerate(bus_numbers):
        m = models[b]
        if m is None or len(X) == 0:
            continue
        P[:, j] = m.predict_proba(X[feats[b]].to_numpy())[:, 1]
    return P


def _calibrate(P, Y, bus_numbers, min_pos):
    return {b: best_f1_threshold(Y[:, j], P[:, j], default=0.5, min_positives=min_pos)
            for j, b in enumerate(bus_numbers)}


def _apply(P, thresholds, bus_numbers):
    thr = np.array([thresholds[b] for b in bus_numbers])
    return (P >= thr).astype(int)


def evaluate(models, feats, thresholds, split, det_flag, bus_numbers) -> dict:
    P = predict_bus_probs(models, feats, split.X, bus_numbers)
    Yhat = _apply(P, thresholds, bus_numbers)
    Y = split.Y_bus
    att = split.attacked
    oracle = multilabel_metrics(Y[att], Yhat[att], bus_numbers)
    two_stage_pred = Yhat * det_flag[:, None]
    two_stage = multilabel_metrics(Y, two_stage_pred, bus_numbers)
    # two-stage restricted to truly attacked snapshots (detection misses -> all-zero rows)
    two_stage_att = multilabel_metrics(Y[att], two_stage_pred[att], bus_numbers)
    return {"oracle_attacked_snapshots": oracle, "two_stage_all_snapshots": two_stage,
            "two_stage_attacked_snapshots": two_stage_att}


def train_localization(cfg: dict, splits: dict, local: dict, det_probs: dict, det_threshold: float,
                       bus_numbers: list[int], regimes: list[str] | None = None,
                       feature_scope: str | None = None, groups: dict | None = None) -> tuple[dict, dict]:
    lc = cfg["ml"]["localization"]
    scope = feature_scope or lc["feature_scope"]
    if regimes is None:
        regimes = ["attack_only", "all"] if lc["training_set"] == "compare" else [lc["training_set"]]
    global_cols = None
    if scope == "global":
        global_cols = [c for g in groups.values() for c in g]
    va = splits["validation"]
    det_va = (det_probs["validation"] >= det_threshold).astype(int)
    results, candidates = {}, {}
    for regime in regimes:
        t0 = time.perf_counter()
        models, feats, info = _fit_regime(cfg, splits, local, regime, bus_numbers, lc, global_cols)
        fit_s = time.perf_counter() - t0
        # calibrate thresholds on validation snapshots that reach the localiser
        P_va = predict_bus_probs(models, feats, va.X, bus_numbers)
        gate = det_va == 1
        thresholds = _calibrate(P_va[gate], va.Y_bus[gate], bus_numbers, max(5, lc["min_positive_examples"] // 5))
        val_eval = evaluate(models, feats, thresholds, va, det_va, bus_numbers)
        res = {"fit_seconds": fit_s, "bus_info": info, "thresholds": thresholds, "validation": val_eval,
               "splits": {}}
        for name, s in splits.items():
            if name in ("train", "validation"):
                continue
            det = (det_probs[name] >= det_threshold).astype(int)
            res["splits"][name] = evaluate(models, feats, thresholds, s, det, bus_numbers)
        results[regime] = res
        candidates[regime] = (models, feats, thresholds)
        log.info("localization[%s/%s]: val two-stage micro-F1=%.4f, test oracle micro-F1=%.4f", regime, scope,
                 val_eval["two_stage_all_snapshots"]["micro_f1"],
                 res["splits"].get("test", {}).get("oracle_attacked_snapshots", {}).get("micro_f1", float("nan")))
    best = max(regimes, key=lambda r: results[r]["validation"]["two_stage_all_snapshots"]["micro_f1"])
    models, feats, thresholds = candidates[best]
    bundle = {"models": models, "features": feats, "thresholds": thresholds, "bus_numbers": bus_numbers,
              "regime": best, "feature_scope": scope,
              "meta": {"model": "localization", "params": lc,
                       "selection": "higher validation two-stage micro-F1"}}
    return bundle, {"selected_regime": best, "feature_scope": scope, "regimes": results}
