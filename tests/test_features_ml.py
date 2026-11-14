"""Tests 22-25: feature engineering (no leakage, train/live identity), Random Forest
training, model save/reload, bus-level localisation and attack-type prediction."""
import numpy as np
import pandas as pd
import pytest

from config.settings import apply_overrides, resolve_path
from ml.feature_engineering import FeatureBuilder, FEATURE_GROUPS
from simulator import schema as S


@pytest.fixture(scope="module")
def small_project(tmp_path_factory):
    """A tiny but complete dataset + trained models in a temporary folder."""
    from config.settings import load_settings
    from simulator.continuous_simulator import ContinuousSimulator
    from ml import train_all
    tmp = tmp_path_factory.mktemp("proj")
    cfg = load_settings(overrides={
        "paths.database": str(tmp / "t.db"), "paths.models": str(tmp / "models"),
        "paths.processed": str(tmp / "processed"), "paths.results": str(tmp / "results"),
        "paths.logs": str(tmp / "logs"), "paths.exports": str(tmp / "exports"), "paths.raw": str(tmp / "raw"),
        "paths.stop_flag": str(tmp / "STOP"),
        "ml.detection.n_estimators": 25, "ml.localization.n_estimators": 15, "ml.attack_type.n_estimators": 25,
        "ml.severity.n_estimators": 15, "ml.localization.min_positive_examples": 5,
        "attacks.start_probability": 0.08, "attacks.replay.min_history_steps": 120,
        "attacks.replay.lag_steps": [40, 110]})
    for purpose, seeds in (("train", (11, 12)), ("validation", (21,)), ("test", (31,))):
        for sd in seeds:
            ContinuousSimulator(cfg, purpose=purpose, seed=sd, fast=True, max_steps=700, tag=purpose).run()
    summary = train_all.run(cfg)
    return cfg, summary


def _df(cfg, steps=120, seed=3):
    from simulator.continuous_simulator import SimulationEngine
    eng = SimulationEngine(cfg, "f", seed)
    return pd.DataFrame([eng.step(t, "", float(t))[0] for t in range(steps)])


def test_features_have_no_leakage(cfg):
    fs = FeatureBuilder(noise=cfg["noise"]).build(_df(cfg, 30))
    S.assert_no_leakage(list(fs.X.columns))
    for c in fs.X.columns:
        assert "_true" not in c and "attack" not in c and "severity" not in c
    assert set(fs.groups) == set(FEATURE_GROUPS)
    assert not fs.X.isna().any().any()


def test_features_ignore_truth_and_labels(cfg):
    df = _df(cfg, 40)
    fb = FeatureBuilder(noise=cfg["noise"])
    a = fb.build(df).X
    df2 = df.copy()
    for c in S.BUS_TRUE_COLUMNS + S.ATTACK_BUS_COLUMNS + ["attack_active", "attack_severity"]:
        df2[c] = 0
    b = fb.build(df2).X
    pd.testing.assert_frame_equal(a, b)


def test_live_features_equal_training_features(cfg):
    df = _df(cfg, 60)
    fb = FeatureBuilder(noise=cfg["noise"])
    full = fb.build(df).X
    for end in (10, 35, 59):
        live = fb.build(df.iloc[end - 5:end + 1]).X.iloc[-1]
        assert np.allclose(live.to_numpy(), full.iloc[end].to_numpy(), atol=1e-9)


def test_consistency_zscores_are_calibrated(cfg):
    from config.settings import apply_overrides
    c = apply_overrides(cfg, {"attacks.enabled": False})
    X = FeatureBuilder(noise=c["noise"]).build(_df(c, 150)).X
    assert 0.6 < X["z_mean_square"].mean() < 1.5      # ~1 for pure sensor noise


def test_training_produces_all_models(small_project):
    cfg, summary = small_project
    d = resolve_path(cfg, "models")
    for f in ("detection_rf.joblib", "localization_rf.joblib", "attack_type_rf.joblib", "severity_rf.joblib"):
        assert (d / f).exists()
    assert "test" in summary["detection"]
    assert (resolve_path(cfg, "results") / "model_report.md").exists()


def test_model_reload_and_prediction(small_project):
    cfg, _ = small_project
    from ml.inference import GridSentinelPredictor
    from storage.database import Database
    from storage.repository import Repository
    from ml.dataset import runs_for
    pred = GridSentinelPredictor(cfg)
    repo = Repository(Database(resolve_path(cfg, "database")))
    rid = runs_for(cfg, "test")[0]
    df = repo.load_run(rid)
    probs, outs = [], []
    for step in range(200, 700, 25):
        w = df[(df.step > step - 6) & (df.step <= step)]
        out = pred.predict(w, cfg["noise"])
        assert 0.0 <= out["attack_probability"] <= 1.0
        assert len(out["bus_probabilities"]) == 30
        if out["prediction"] == "ATTACK":
            assert out["predicted_attack_type"] in S.ATTACK_TYPES
            assert set(out["predicted_compromised_buses"]) <= set(range(1, 31))
        probs.append(out["attack_probability"])
        outs.append(out)
    assert len(set(np.round(probs, 6))) > 3              # probability is not constant
    assert any(o["prediction"] == "ATTACK" for o in outs)


def test_localization_not_all_zero(small_project):
    cfg, _ = small_project
    import json
    loc = json.loads((resolve_path(cfg, "models") / "metrics" / "localization.json").read_text())
    sel = loc["regimes"][loc["selected_regime"]]["splits"]["test"]["oracle_attacked_snapshots"]
    assert sel["all_zero_prediction_rate"] < 0.9
    assert sum(r["predicted_positive"] for r in sel["per_bus"]) > 0
    assert len(sel["per_bus"]) == 30


def test_missing_model_gives_clear_error(cfg):
    from ml.common import load_bundle, ModelLoadError
    with pytest.raises(ModelLoadError, match="Train the models"):
        load_bundle(resolve_path(cfg, "models") / "does_not_exist.joblib")


def test_chronological_split_gap():
    from ml.dataset import chronological_split
    tr, te = chronological_split(1000, 0.25, gap=100)
    assert tr.max() + 100 < te.min()
