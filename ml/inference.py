"""Live inference with the trained Random Forest models.

Pipeline per snapshot (two-stage):
  features(last N rows of the run)  ->  Model A detection probability
     if probability >= calibrated threshold:
        Model B per-bus probabilities -> buses above their calibrated thresholds
        Model C attack type (+ class probabilities)
        Model D severity estimate
Per-bus probabilities are computed for every snapshot (useful for the
digital twin), but buses are only *declared* compromised when the detector
alarms - exactly as evaluated offline in the two-stage results.

    python -m ml.inference --run <run_id> [--step N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import load_settings, resolve_path
from ml.common import load_bundle, set_inference_threads, ModelLoadError
from ml.feature_engineering import FeatureBuilder
from ml.train_all import MODEL_FILES
from simulator import schema as S


class GridSentinelPredictor:
    def __init__(self, cfg: dict, models_dir: Path | None = None):
        self.cfg = cfg
        d = models_dir or resolve_path(cfg, "models")
        self.detection = load_bundle(d / MODEL_FILES["detection"])
        self.localization = load_bundle(d / MODEL_FILES["localization"])
        self.attack_type = load_bundle(d / MODEL_FILES["attack_type"])
        from ml.dataset import FEATURE_VERSION
        for name, b in (("detection", self.detection), ("localization", self.localization),
                        ("attack_type", self.attack_type)):
            if b["meta"].get("feature_version") != FEATURE_VERSION:
                raise ModelLoadError(f"The {name} model was trained with feature version "
                                     f"{b['meta'].get('feature_version')} but the code uses {FEATURE_VERSION}. "
                                     "Retrain the models (python cli.py train).")
        sev_path = d / MODEL_FILES["severity"]
        self.severity = load_bundle(sev_path) if sev_path.exists() else None
        for b in (self.detection, self.attack_type, self.severity):
            if b:
                set_inference_threads(b["model"])
        for m in self.localization["models"].values():
            if m is not None:
                set_inference_threads(m)
        self.window = self.detection["meta"].get("feature_window", cfg["features"]["rolling_window"])
        self.physics = bool(cfg["features"].get("physics_features", False))
        self._builders: dict[str, FeatureBuilder] = {}
        self.fb = self.builder(cfg["noise"])
        self.history_steps = int(cfg["features"]["history_steps"])
        self.bus_numbers = self.localization["bus_numbers"]
        self.threshold = float(self.detection["threshold"])

    @property
    def info(self) -> dict:
        return {
            "detection_threshold": self.threshold,
            "localization_regime": self.localization["regime"],
            "localization_feature_scope": self.localization["feature_scope"],
            "buses_without_model": [b for b, m in self.localization["models"].items() if m is None],
            "attack_type_classes": self.attack_type["classes"],
            "severity_model": self.severity is not None,
            "trained_at": self.detection["meta"].get("saved_at"),
            "sklearn_version": self.detection["meta"].get("sklearn_version"),
            "train_runs": self.detection["meta"].get("train_runs"),
        }

    def builder(self, noise: dict) -> FeatureBuilder:
        """Feature builder for a sensor-accuracy specification (cached)."""
        key = json.dumps({k: noise[k] for k in sorted(noise) if k != "seed_offset"}, sort_keys=True)
        if key not in self._builders:
            self._builders[key] = FeatureBuilder(rolling_window=self.window, noise=noise, physics=self.physics)
        return self._builders[key]

    def features_for_last(self, window_df: pd.DataFrame, noise: dict | None = None) -> pd.DataFrame:
        """Feature row of the last snapshot in ``window_df`` (sorted by step)."""
        fb = self.builder(noise) if noise else self.fb
        X = fb.build(window_df, physics_last_only=True).X
        return X.iloc[[-1]].astype(np.float32)

    def predict(self, window_df: pd.DataFrame, noise: dict | None = None) -> dict:
        """``noise``: sensor-accuracy spec of the measurement system (default: config)."""
        t0 = time.perf_counter()
        x = self.features_for_last(window_df, noise)
        t_feat = time.perf_counter()
        prob = float(self.detection["model"].predict_proba(x[self.detection["features"]].to_numpy())[0, 1])
        alarm = prob >= self.threshold
        loc = self.localization
        bus_prob, compromised = {}, []
        for b in self.bus_numbers:
            m = loc["models"][b]
            p = 0.0 if m is None else float(m.predict_proba(x[loc["features"][b]].to_numpy())[0, 1])
            bus_prob[b] = p
            if alarm and m is not None and p >= loc["thresholds"][b]:
                compromised.append(b)
        type_pred, type_probs, sev = None, {}, None
        if alarm:
            tm = self.attack_type
            pr = tm["model"].predict_proba(x[tm["features"]].to_numpy())[0]
            type_probs = {c: float(v) for c, v in zip(tm["classes"], pr)}
            type_pred = max(type_probs, key=type_probs.get)
            if self.severity:
                sev = float(self.severity["model"].predict(x[self.severity["features"]].to_numpy())[0])
        t_end = time.perf_counter()
        last = window_df.iloc[-1]
        return {
            "source": "AI prediction (Random Forest)",
            "run_id": str(last.get("run_id", "")), "step": int(last["step"]),
            "attack_probability": prob,
            "detection_threshold": self.threshold,
            "prediction": "ATTACK" if alarm else "NORMAL",
            "bus_probabilities": {str(b): p for b, p in bus_prob.items()},
            "bus_thresholds": {str(b): float(loc["thresholds"][b]) for b in self.bus_numbers},
            "predicted_compromised_buses": compromised,
            "predicted_attack_type": type_pred,
            "attack_type_probabilities": type_probs,
            "estimated_severity": sev,
            "latency_ms": {"features": (t_feat - t0) * 1000, "models": (t_end - t_feat) * 1000,
                           "total": (t_end - t0) * 1000},
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the trained models on a stored snapshot")
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, default=None, help="default: latest step")
    args = ap.parse_args(argv)
    cfg = load_settings()
    from storage.database import Database
    from storage.repository import Repository
    repo = Repository(Database(resolve_path(cfg, "database")))
    try:
        pred = GridSentinelPredictor(cfg)
    except ModelLoadError as exc:
        print(f"ERROR: {exc}")
        return 2
    step = args.step if args.step is not None else repo.latest_step(args.run)
    if step is None:
        print("ERROR: run has no measurements")
        return 2
    df = repo.get_rows(args.run, step - pred.history_steps + 1, step,
                       ["run_id", "step"] + S.ML_INPUT_COLUMNS + ["attack_active", "attack_type"]
                       + S.ATTACK_BUS_COLUMNS)
    out = pred.predict(df)
    truth = df.iloc[-1]
    out["ground_truth (simulator, for comparison only)"] = {
        "attack_active": int(truth["attack_active"]), "attack_type": truth["attack_type"],
        "attacked_buses": [b for b in range(1, 31) if truth[f"attack_bus_{b}"] == 1]}
    out.pop("bus_probabilities")
    out.pop("bus_thresholds")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
