"""Shared ML utilities: model bundles, metrics, threshold calibration."""
from __future__ import annotations

import json
import platform
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, precision_recall_curve)

from simulator import schema as S


class ModelLoadError(RuntimeError):
    pass


def rf_classifier(params: dict, n_jobs: int, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=params["n_estimators"], max_depth=params.get("max_depth"),
        min_samples_leaf=params.get("min_samples_leaf", 1), max_features=params.get("max_features", "sqrt"),
        class_weight=params.get("class_weight"), n_jobs=n_jobs, random_state=seed)


def rf_regressor(params: dict, n_jobs: int, seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=params["n_estimators"], max_depth=params.get("max_depth"),
        min_samples_leaf=params.get("min_samples_leaf", 1), max_features=params.get("max_features", 1.0),
        n_jobs=n_jobs, random_state=seed)


def binary_metrics(y_true, y_pred, y_prob=None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "n": int(len(y_true)), "positives": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return m


def best_f1_threshold(y_true, y_prob, default: float = 0.5, min_positives: int = 1) -> float:
    """Threshold maximising F1 on a validation set (falls back to ``default``)."""
    y_true = np.asarray(y_true).astype(int)
    if y_true.sum() < min_positives or y_true.sum() == len(y_true):
        return default
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1[:-1])) if len(thr) else 0
    return float(np.clip(thr[i], 0.05, 0.95)) if len(thr) else default


def multilabel_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, bus_numbers: list[int]) -> dict:
    """Micro/macro F1 + per-bus table (support, predicted positives, P, R, F1)."""
    Y_true = np.asarray(Y_true).astype(int)
    Y_pred = np.asarray(Y_pred).astype(int)
    per_bus = []
    for j, b in enumerate(bus_numbers):
        t, p = Y_true[:, j], Y_pred[:, j]
        per_bus.append({
            "bus": int(b), "actual_positive": int(t.sum()), "predicted_positive": int(p.sum()),
            "precision": float(precision_score(t, p, zero_division=0)),
            "recall": float(recall_score(t, p, zero_division=0)),
            "f1": float(f1_score(t, p, zero_division=0)),
        })
    has_support = [r["f1"] for r in per_bus if r["actual_positive"] > 0]
    exact = float(np.mean(np.all(Y_true == Y_pred, axis=1))) if len(Y_true) else 0.0
    return {
        "n_snapshots": int(len(Y_true)),
        "micro_precision": float(precision_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "macro_f1": float(np.mean(has_support)) if has_support else 0.0,
        "macro_f1_note": "mean of per-bus F1 over buses with at least one true positive",
        "exact_match_ratio": exact,
        "all_zero_prediction_rate": float(np.mean(Y_pred.sum(axis=1) == 0)) if len(Y_pred) else 0.0,
        "per_bus": per_bus,
    }


def save_bundle(path: Path, bundle: dict) -> None:
    bundle = dict(bundle)
    bundle.setdefault("meta", {})
    bundle["meta"].update({
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__, "python": platform.python_version(),
        "schema_version": S.SCHEMA_VERSION,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_bundle(path: Path) -> dict:
    if not path.exists():
        raise ModelLoadError(f"Model file not found: {path}. Train the models first (python cli.py train).")
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(f"Could not load {path}: {type(exc).__name__}: {exc}") from exc
    meta = bundle.get("meta", {})
    if meta.get("schema_version") != S.SCHEMA_VERSION:
        raise ModelLoadError(f"{path.name} was trained on schema v{meta.get('schema_version')}, "
                             f"code uses v{S.SCHEMA_VERSION}. Retrain the models.")
    if meta.get("sklearn_version") != sklearn.__version__:
        warnings.warn(f"{path.name} was trained with scikit-learn {meta.get('sklearn_version')}, "
                      f"running {sklearn.__version__}; retrain if predictions look wrong.")
    return bundle


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default))


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def set_inference_threads(model) -> None:
    """Single-sample inference is fastest (and warning-free) on one thread."""
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
