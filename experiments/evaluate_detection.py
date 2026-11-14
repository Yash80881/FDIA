"""Detection experiments (all on independent test / stress runs).

1. Feature-group ablation: bus | bus+line | bus+line+temporal | bus+line+electrical | full
   (the bus+line Random Forest is the "RF baseline"; full = RF with feature engineering)
2. Alternative models vs Random Forest (full features): ExtraTrees, HistGradientBoosting,
   logistic regression
3. Breakdowns of the trained detector: attack type, single vs multi-bus, attack severity
   (magnitude) bins, operating scenario (load level), stress runs (noise, heavy load,
   low intensity)
4. AI-only vs physics-only (WLS chi-square / LNR) vs combined decision
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.common import rf_classifier, binary_metrics, best_f1_threshold, load_bundle
from ml.feature_engineering import feature_columns

log = logging.getLogger("gridsentinel.exp.detection")

ABLATIONS = {
    "bus only": ("bus",),
    "bus + line (RF baseline)": ("bus", "line"),
    "bus + line + temporal": ("bus", "line", "temporal"),
    "bus + line + electrical": ("bus", "line", "electrical"),
    "full feature engineering": ("bus", "line", "electrical", "temporal"),
}
EVAL_SPLITS = ("test", "stress_high_noise", "stress_heavy_load", "stress_low_intensity")


def _fit_eval(model, cols, splits) -> dict:
    tr, va = splits["train"], splits["validation"]
    t0 = time.perf_counter()
    model.fit(tr.X[cols].to_numpy(), tr.y["attack_active"].to_numpy())
    fit_s = time.perf_counter() - t0
    thr = best_f1_threshold(va.y["attack_active"], model.predict_proba(va.X[cols].to_numpy())[:, 1])
    out = {"threshold": thr, "fit_seconds": fit_s, "n_features": len(cols)}
    for name in EVAL_SPLITS:
        if name in splits:
            s = splits[name]
            p = model.predict_proba(s.X[cols].to_numpy())[:, 1]
            out[name] = binary_metrics(s.y["attack_active"], p >= thr, p)
    return out


def ablation(cfg, splits, groups, n_estimators: int) -> pd.DataFrame:
    params = dict(cfg["ml"]["detection"], n_estimators=n_estimators)
    rows = []
    for name, sel in ABLATIONS.items():
        cols = feature_columns(groups, sel)
        r = _fit_eval(rf_classifier(params, cfg["ml"]["n_jobs"], cfg["project"]["random_seed"]), cols, splits)
        for split in EVAL_SPLITS:
            if split in r:
                m = r[split]
                rows.append({"features": name, "n_features": r["n_features"], "split": split,
                             "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
                             "f1": m["f1"], "fpr": m["false_positive_rate"], "fnr": m["false_negative_rate"],
                             "roc_auc": m.get("roc_auc")})
        log.info("ablation %-28s test F1=%.4f", name, r["test"]["f1"])
    return pd.DataFrame(rows)


def alternative_models(cfg, splits, groups, n_estimators: int) -> pd.DataFrame:
    cols = feature_columns(groups, ("bus", "line", "electrical", "temporal"))
    seed = cfg["project"]["random_seed"]
    models = {
        "Random Forest": rf_classifier(dict(cfg["ml"]["detection"], n_estimators=n_estimators), cfg["ml"]["n_jobs"], seed),
        "Extra Trees": ExtraTreesClassifier(n_estimators=n_estimators, min_samples_leaf=2, class_weight="balanced",
                                            n_jobs=cfg["ml"]["n_jobs"], random_state=seed),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=seed),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1,
                                                                                  class_weight="balanced")),
    }
    rows = []
    for name, m in models.items():
        r = _fit_eval(m, cols, splits)
        for split in EVAL_SPLITS:
            if split in r:
                mm = r[split]
                rows.append({"model": name, "split": split, "accuracy": mm["accuracy"], "precision": mm["precision"],
                             "recall": mm["recall"], "f1": mm["f1"], "fpr": mm["false_positive_rate"],
                             "roc_auc": mm.get("roc_auc"), "fit_seconds": r["fit_seconds"]})
        log.info("model %-22s test F1=%.4f", name, r["test"]["f1"])
    return pd.DataFrame(rows)


def breakdowns(cfg, splits, det_bundle) -> dict:
    cols, thr, model = det_bundle["features"], det_bundle["threshold"], det_bundle["model"]
    s = splits["test"]
    p = model.predict_proba(s.X[cols].to_numpy())[:, 1]
    pred = p >= thr
    y = s.y
    att = y["attack_active"] == 1
    out = {}
    out["by_attack_type"] = {t: float(pred[(y["attack_type"] == t).to_numpy()].mean())
                             for t in ["additive", "scaling", "replay", "coordinated"]}
    single = (att & (y["attacked_bus_count"] == 1)).to_numpy()
    multi = (att & (y["attacked_bus_count"] > 1)).to_numpy()
    out["single_vs_multi_bus_recall"] = {"single-bus": float(pred[single].mean()), "multi-bus": float(pred[multi].mean())}
    bins = [0, 2, 4, 8, 16, 1e9]
    labels = ["<2σ", "2-4σ", "4-8σ", "8-16σ", ">16σ"]
    sev_bin = pd.cut(y.loc[att, "attack_severity"], bins=bins, labels=labels)
    out["recall_by_severity"] = {str(k): {"recall": float(pred[att.to_numpy()][(sev_bin == k).to_numpy()].mean()),
                                          "n": int((sev_bin == k).sum())}
                                 for k in labels if (sev_bin == k).sum() > 0}
    out["by_scenario"] = {}
    for sc in sorted(y["scenario"].unique()):
        m = (y["scenario"] == sc).to_numpy()
        bm = binary_metrics(y["attack_active"].to_numpy()[m], pred[m])
        out["by_scenario"][sc] = {"n": int(m.sum()), "f1": bm["f1"], "fpr": bm["false_positive_rate"],
                                  "recall": bm["recall"]}
    out["recall_by_attack_step"] = {}
    for k in (0, 1, 2, 5, 10):
        m = (y["attack_step_in_event"] == k).to_numpy()
        if m.sum():
            out["recall_by_attack_step"][f"step {k} of event"] = float(pred[m].mean())
    return out


def ai_vs_physics(cfg, splits, det_bundle, max_rows: int | None = None) -> pd.DataFrame:
    """Compare AI-only, physics-only (WLS bad-data test) and the combined rule."""
    from physics.state_estimation import WLSStateEstimator
    from physics.verification import verify_snapshot, combine
    from storage.database import Database
    from storage.repository import Repository
    from config.settings import resolve_path
    import json
    repo = Repository(Database(resolve_path(cfg, "database")))
    se = WLSStateEstimator()
    cols, thr, model = det_bundle["features"], det_bundle["threshold"], det_bundle["model"]
    rows = []
    for split in EVAL_SPLITS:
        if split not in splits:
            continue
        s = splits[split]
        probs = model.predict_proba(s.X[cols].to_numpy())[:, 1]
        res = {"ai": [], "phys": [], "comb": [], "y": []}
        pos = {(r, int(st)): i for i, (r, st) in enumerate(zip(s.y["run_id"], s.y["step"]))}
        for rid in s.run_ids:
            run_cfg = json.loads(repo.get_run(rid)["config_json"])   # the run's own sensor-noise model
            pcfg = dict(cfg, noise=run_cfg["noise"])
            df = repo.load_run(rid)
            if max_rows:
                df = df.iloc[: max_rows // len(s.run_ids)]
            for row in df.to_dict("records"):
                i = pos[(rid, int(row["step"]))]
                ph = verify_snapshot(se, row, pcfg, identify=False)
                c = combine(float(probs[i]), thr, ph)
                res["ai"].append(probs[i] >= thr)
                res["phys"].append(ph.get("bad_data_detected", False))
                res["comb"].append(c["decision"] == "ATTACK")
                res["y"].append(int(row["attack_active"]))
        y = np.array(res["y"])
        for name, key in (("AI only (Random Forest)", "ai"), ("Physics only (WLS χ²/LNR)", "phys"),
                          ("Combined decision", "comb")):
            m = binary_metrics(y, np.array(res[key]).astype(int))
            rows.append({"method": name, "split": split, "n": m["n"], "accuracy": m["accuracy"],
                         "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                         "fpr": m["false_positive_rate"]})
        log.info("AI vs physics on %s done (%d rows)", split, len(y))
    return pd.DataFrame(rows)
