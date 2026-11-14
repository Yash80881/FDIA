"""Train and evaluate all GridSentinel models in one go.

    python -m ml.train_all                 # uses the generated dataset runs

Order: detection -> localisation (needs detector outputs for the two-stage
evaluation) -> attack type -> severity.  Models go to ``models/``, metrics to
``models/metrics/*.json`` and a readable report to ``results/model_report.md``.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import load_settings, resolve_path, config_hash
from ml.common import save_bundle, write_json
from ml.dataset import FEATURE_VERSION
from ml.preprocessing import load_splits, check_distribution
from ml.train_detection import train_detection
from ml.train_localization import train_localization
from ml.train_attack_type import train_attack_type
from ml.train_severity import train_severity
from simulator.grid_model import GridModel

log = logging.getLogger("gridsentinel.train")

MODEL_FILES = {"detection": "detection_rf.joblib", "localization": "localization_rf.joblib",
               "attack_type": "attack_type_rf.joblib", "severity": "severity_rf.joblib"}


def run(cfg: dict) -> dict:
    models_dir = resolve_path(cfg, "models")
    metrics_dir = models_dir / "metrics"
    t0 = time.perf_counter()
    splits, groups, local = load_splits(cfg)
    bus_numbers = GridModel().bus_numbers
    dist = {name: check_distribution(s) for name, s in splits.items()}
    write_json(metrics_dir / "data_distribution.json", dist)
    common_meta = {"config_hash": config_hash(cfg), "train_runs": splits["train"].run_ids,
                   "validation_runs": splits["validation"].run_ids if "validation" in splits else [],
                   "feature_window": cfg["features"]["rolling_window"], "feature_version": FEATURE_VERSION}

    det_bundle, det = train_detection(cfg, splits, groups)
    det_bundle["meta"].update(common_meta)
    save_bundle(models_dir / MODEL_FILES["detection"], det_bundle)
    write_json(metrics_dir / "detection.json", det["metrics"])

    loc_bundle, loc = train_localization(cfg, splits, local, det["probs"], det_bundle["threshold"],
                                         bus_numbers, groups=groups)
    loc_bundle["meta"].update(common_meta)
    save_bundle(models_dir / MODEL_FILES["localization"], loc_bundle)
    write_json(metrics_dir / "localization.json", loc)

    typ_bundle, typ = train_attack_type(cfg, splits, groups, det["probs"], det_bundle["threshold"])
    typ_bundle["meta"].update(common_meta)
    save_bundle(models_dir / MODEL_FILES["attack_type"], typ_bundle)
    write_json(metrics_dir / "attack_type.json", typ)

    sev = None
    if cfg["ml"]["severity"]["enabled"]:
        sev_bundle, sev = train_severity(cfg, splits, groups)
        sev_bundle["meta"].update(common_meta)
        save_bundle(models_dir / MODEL_FILES["severity"], sev_bundle)
        write_json(metrics_dir / "severity.json", sev)

    summary = summarize(det["metrics"], loc, typ, sev)
    summary["training_seconds"] = time.perf_counter() - t0
    summary["runs"] = {name: s.run_ids for name, s in splits.items()}
    write_json(metrics_dir / "summary.json", summary)
    write_report(resolve_path(cfg, "results") / "model_report.md", summary, det["metrics"], loc, typ, sev, dist)
    log.info("All models trained in %.1fs", summary["training_seconds"])
    return summary


def summarize(det, loc, typ, sev) -> dict:
    sel = loc["selected_regime"]
    out = {"detection": {}, "localization": {"selected_regime": sel}, "attack_type": {}, "severity": {}}
    for name, m in det["splits"].items():
        out["detection"][name] = {k: m[k] for k in ("accuracy", "precision", "recall", "f1",
                                                    "false_positive_rate", "false_negative_rate", "roc_auc")
                                  if k in m}
    for regime, r in loc["regimes"].items():
        for name, m in r["splits"].items():
            out["localization"].setdefault(regime, {})[name] = {
                "oracle_micro_f1": m["oracle_attacked_snapshots"]["micro_f1"],
                "oracle_macro_f1": m["oracle_attacked_snapshots"]["macro_f1"],
                "two_stage_micro_f1": m["two_stage_all_snapshots"]["micro_f1"],
                "two_stage_macro_f1": m["two_stage_all_snapshots"]["macro_f1"],
            }
    for name, m in typ["splits"].items():
        out["attack_type"][name] = {"accuracy": m["attacked_snapshots"]["accuracy"],
                                    "macro_f1": m["attacked_snapshots"]["macro_f1"]}
    if sev:
        for name, m in sev["splits"].items():
            out["severity"][name] = {k: m[k] for k in ("mae", "rmse", "r2", "spearman")}
    out["detection_threshold"] = det["threshold"]
    return out


def _f(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)


def write_report(path: Path, summary, det, loc, typ, sev, dist) -> None:
    L = ["# GridSentinel AI - offline model evaluation", "",
         "All numbers below were produced by `python -m ml.train_all` on independent simulation runs "
         "(run-group split: models fit on *train* runs, thresholds / regime chosen on *validation* runs, "
         "results reported on unseen *test* and *stress* runs).", ""]
    L += ["## Data", "", "| split | rows | attack fraction | additive | scaling | replay | coordinated |",
          "|---|---|---|---|---|---|---|"]
    for name, d in dist.items():
        t = d["attack_type_rows"]
        L.append(f"| {name} | {d['rows']} | {d['attack_fraction']:.3f} | {t.get('additive', 0)} | "
                 f"{t.get('scaling', 0)} | {t.get('replay', 0)} | {t.get('coordinated', 0)} |")
    L += ["", f"## Model A - detection (threshold {det['threshold']:.3f}, {det['n_features']} features)", "",
          "| split | accuracy | precision | recall | F1 | FPR | FNR | ROC AUC |", "|---|---|---|---|---|---|---|---|"]
    for name, m in det["splits"].items():
        L.append(f"| {name} | {_f(m['accuracy'])} | {_f(m['precision'])} | {_f(m['recall'])} | {_f(m['f1'])} | "
                 f"{_f(m['false_positive_rate'])} | {_f(m['false_negative_rate'])} | {_f(m.get('roc_auc', 'n/a'))} |")
    L += ["", "Detection recall by attack type:", "", "| split | " + " | ".join(
        ["additive", "scaling", "replay", "coordinated"]) + " |", "|---|---|---|---|---|"]
    for name, m in det["splits"].items():
        r = m["recall_by_attack_type"]
        L.append(f"| {name} | " + " | ".join(_f(r.get(t, float('nan'))) for t in
                                              ["additive", "scaling", "replay", "coordinated"]) + " |")
    L += ["", f"## Model B - localisation (selected regime: **{loc['selected_regime']}**, "
              f"feature scope: {loc['feature_scope']})", "",
          "| regime | split | oracle micro-F1 | oracle macro-F1 | two-stage micro-F1 | two-stage macro-F1 |",
          "|---|---|---|---|---|---|"]
    for regime, r in loc["regimes"].items():
        v = r["validation"]
        L.append(f"| {regime} | validation | {_f(v['oracle_attacked_snapshots']['micro_f1'])} | "
                 f"{_f(v['oracle_attacked_snapshots']['macro_f1'])} | {_f(v['two_stage_all_snapshots']['micro_f1'])} | "
                 f"{_f(v['two_stage_all_snapshots']['macro_f1'])} |")
        for name, m in r["splits"].items():
            L.append(f"| {regime} | {name} | {_f(m['oracle_attacked_snapshots']['micro_f1'])} | "
                     f"{_f(m['oracle_attacked_snapshots']['macro_f1'])} | "
                     f"{_f(m['two_stage_all_snapshots']['micro_f1'])} | {_f(m['two_stage_all_snapshots']['macro_f1'])} |")
    L += ["", "*Oracle* = localiser applied to truly attacked snapshots only. *Two-stage* = every snapshot; the "
              "localiser only runs when the detector raises an alarm, so detection misses and false alarms count.", ""]
    sel = loc["regimes"][loc["selected_regime"]]["splits"].get("test")
    if sel:
        L += ["### Per-bus localisation on the test runs (selected regime, oracle setting)", "",
              "| bus | actual + | predicted + | precision | recall | F1 |", "|---|---|---|---|---|---|"]
        for r in sel["oracle_attacked_snapshots"]["per_bus"]:
            L.append(f"| {r['bus']} | {r['actual_positive']} | {r['predicted_positive']} | {_f(r['precision'])} | "
                     f"{_f(r['recall'])} | {_f(r['f1'])} |")
    L += ["", "## Model C - attack-type classification", "",
          "| split | accuracy (attacked snapshots) | macro-F1 | two-stage accuracy | two-stage macro-F1 |",
          "|---|---|---|---|---|"]
    for name, m in typ["splits"].items():
        a, b = m["attacked_snapshots"], m["two_stage_detected_snapshots"]
        L.append(f"| {name} | {_f(a['accuracy'])} | {_f(a['macro_f1'])} | {_f(b['accuracy'])} | {_f(b['macro_f1'])} |")
    t = typ["splits"].get("test")
    if t:
        cm = t["attacked_snapshots"]["confusion_matrix"]
        L += ["", "Test confusion matrix (rows = simulator truth, columns = prediction):", "",
              "| truth \\ pred | " + " | ".join(cm["labels"]) + " |", "|---" * (len(cm["labels"]) + 1) + "|"]
        for lab, row in zip(cm["labels"], cm["matrix"]):
            L.append(f"| {lab} | " + " | ".join(str(v) for v in row) + " |")
    if sev:
        L += ["", "## Model D - severity regression (attacked snapshots)", "",
              "| split | MAE | RMSE | R² | Spearman |", "|---|---|---|---|---|"]
        for name, m in sev["splits"].items():
            L.append(f"| {name} | {_f(m['mae'])} | {_f(m['rmse'])} | {_f(m['r2'])} | {_f(m['spearman'])} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train all GridSentinel Random Forest models")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = load_settings(None, args.set)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    summary = run(cfg)
    print(f"Done. Report: {resolve_path(cfg, 'results') / 'model_report.md'}")
    print({k: v for k, v in summary["detection"].items() if k == "test"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
