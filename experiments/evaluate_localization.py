"""Localisation experiments.

* training regime: attack-only vs all snapshots (from the main training run)
* feature scope: bus-local features vs all (global) features per bus model
* single multi-output Random Forest (one forest predicting all 30 labels) baseline
* physics baseline: buses named by LNR bad-data identification (WLS)
* detection vs localisation performance, single- vs multi-bus attacks
All results on independent test runs; oracle (true attacked snapshots) and
two-stage (detector-gated) settings are reported separately.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ml.common import rf_classifier, multilabel_metrics
from ml.feature_engineering import feature_columns
from ml.train_localization import train_localization

log = logging.getLogger("gridsentinel.exp.localization")


def scope_comparison(cfg, splits, groups, local, det_probs, det_thr, bus_numbers, n_estimators: int) -> pd.DataFrame:
    rows = []
    for scope in ("local", "global"):
        c = dict(cfg)
        c["ml"] = dict(cfg["ml"])
        c["ml"]["localization"] = dict(cfg["ml"]["localization"], n_estimators=n_estimators)
        _, res = train_localization(c, splits, local, det_probs, det_thr, bus_numbers, regimes=["attack_only", "all"],
                                    feature_scope=scope, groups=groups)
        for regime, r in res["regimes"].items():
            for split, m in r["splits"].items():
                rows.append({"feature_scope": scope, "regime": regime, "split": split,
                             "oracle_micro_f1": m["oracle_attacked_snapshots"]["micro_f1"],
                             "oracle_macro_f1": m["oracle_attacked_snapshots"]["macro_f1"],
                             "two_stage_micro_f1": m["two_stage_all_snapshots"]["micro_f1"],
                             "two_stage_macro_f1": m["two_stage_all_snapshots"]["macro_f1"],
                             "fit_seconds": r["fit_seconds"]})
        log.info("scope %s done", scope)
    return pd.DataFrame(rows)


def multioutput_baseline(cfg, splits, groups, det_probs, det_thr, bus_numbers, n_estimators: int) -> pd.DataFrame:
    """One RandomForest fitted on the 30-column label matrix (native multi-output)."""
    cols = feature_columns(groups, ("bus", "line", "electrical", "temporal"))
    tr = splits["train"]
    m = tr.attacked
    params = dict(cfg["ml"]["localization"], n_estimators=n_estimators, class_weight=None)
    model = rf_classifier(params, cfg["ml"]["n_jobs"], cfg["project"]["random_seed"])
    model.fit(tr.X.loc[m, cols].to_numpy(), tr.Y_bus[m])
    rows = []
    for split in ("test", "stress_high_noise", "stress_heavy_load", "stress_low_intensity"):
        if split not in splits:
            continue
        s = splits[split]
        proba = model.predict_proba(s.X[cols].to_numpy())
        P = np.column_stack([p[:, 1] if p.shape[1] == 2 else np.zeros(len(s)) for p in proba])
        Yhat = (P >= 0.5).astype(int)
        det = (det_probs[split] >= det_thr).astype(int)
        o = multilabel_metrics(s.Y_bus[s.attacked], Yhat[s.attacked], bus_numbers)
        t = multilabel_metrics(s.Y_bus, Yhat * det[:, None], bus_numbers)
        rows.append({"method": "multi-output RF (attack-only, threshold 0.5)", "split": split,
                     "oracle_micro_f1": o["micro_f1"], "oracle_macro_f1": o["macro_f1"],
                     "two_stage_micro_f1": t["micro_f1"], "two_stage_macro_f1": t["macro_f1"]})
    return pd.DataFrame(rows)


def breakdown_by_bus_count(loc_bundle, splits, bus_numbers) -> pd.DataFrame:
    from ml.train_localization import predict_bus_probs, _apply
    s = splits["test"]
    P = predict_bus_probs(loc_bundle["models"], loc_bundle["features"], s.X, bus_numbers)
    Yhat = _apply(P, loc_bundle["thresholds"], bus_numbers)
    rows = []
    cnt = s.y["attacked_bus_count"].to_numpy()
    for label, mask in (("single-bus", cnt == 1), ("2-3 buses", (cnt >= 2) & (cnt <= 3)), ("4+ buses", cnt >= 4)):
        if mask.sum() == 0:
            continue
        m = multilabel_metrics(s.Y_bus[mask], Yhat[mask], bus_numbers)
        rows.append({"attacked buses": label, "snapshots": int(mask.sum()), "micro_f1": m["micro_f1"],
                     "micro_precision": m["micro_precision"], "micro_recall": m["micro_recall"],
                     "exact_match": m["exact_match_ratio"]})
    for t in ("additive", "scaling", "replay", "coordinated"):
        mask = (s.y["attack_type"] == t).to_numpy()
        m = multilabel_metrics(s.Y_bus[mask], Yhat[mask], bus_numbers)
        rows.append({"attacked buses": f"type: {t}", "snapshots": int(mask.sum()), "micro_f1": m["micro_f1"],
                     "micro_precision": m["micro_precision"], "micro_recall": m["micro_recall"],
                     "exact_match": m["exact_match_ratio"]})
    return pd.DataFrame(rows)


def physics_localization_baseline(cfg, splits, bus_numbers, max_rows: int = 600) -> dict:
    """Buses named by LNR identification on truly attacked test snapshots."""
    from physics.state_estimation import WLSStateEstimator
    from physics.verification import verify_snapshot
    from storage.database import Database
    from storage.repository import Repository
    from config.settings import resolve_path
    repo = Repository(Database(resolve_path(cfg, "database")))
    se = WLSStateEstimator()
    s = splits["test"]
    Y, Yhat = [], []
    per_run = max_rows // len(s.run_ids)
    for rid in s.run_ids:
        pcfg = dict(cfg, noise=json.loads(repo.get_run(rid)["config_json"])["noise"])
        df = repo.load_run(rid)
        df = df[df["attack_active"] == 1].iloc[:per_run]
        for row in df.to_dict("records"):
            ph = verify_snapshot(se, row, pcfg, identify=True)
            buses = {r["bus"] for r in ph.get("identified_bad_measurements", []) if r["kind"] in ("V", "Ang")}
            Yhat.append([int(b in buses) for b in bus_numbers])
            Y.append([int(row[f"attack_bus_{b}"]) for b in bus_numbers])
    m = multilabel_metrics(np.array(Y), np.array(Yhat), bus_numbers)
    return {"method": "physics LNR identification (V/angle sensors)", "snapshots": len(Y),
            "micro_f1": m["micro_f1"], "micro_precision": m["micro_precision"], "micro_recall": m["micro_recall"],
            "macro_f1": m["macro_f1"]}
