"""Tests 14-19: the four FDIA types, label correctness and truth preservation."""
import json

import numpy as np
import pytest

from simulator.continuous_simulator import SimulationEngine
from attacks import additive, scaling, replay, coordinated
from attacks.base import AttackEvent
from simulator.power_flow import run_ac_power_flow
from simulator.measurement_generator import MeasurementGenerator


def _snap(cfg, grid, step=0, seed=0):
    pf = run_ac_power_flow(grid.new_case())
    return MeasurementGenerator(grid, cfg, np.random.default_rng(seed)).snapshot(step, pf)


def test_additive(cfg, grid):
    rng = np.random.default_rng(0)
    bc, params = additive.create(rng, [3, 7], ["V", "Ang"], cfg["attacks"]["additive"], 1.0)
    ev = AttackEvent("x", "additive", 0, 5, bc, params)
    s = _snap(cfg, grid)
    additive.apply(ev, s, grid.bus_index)
    d = s.meas_bus.vm - s.clean_meas_bus.vm
    changed = set(np.flatnonzero(np.abs(d) > 0) + 1) | set(np.flatnonzero(np.abs(s.meas_bus.va - s.clean_meas_bus.va) > 0) + 1)
    assert changed == {3, 7}
    assert np.array_equal(s.true_bus.vm, run_ac_power_flow(grid.new_case()).vm_pu)


def test_scaling(cfg, grid):
    rng = np.random.default_rng(1)
    bc, params = scaling.create(rng, [1, 9], ["Ang"], cfg["attacks"]["scaling"], 1.0, grid.ref_bus)
    assert bc[1] == ["V"]     # reference-bus angle is never scaled
    ev = AttackEvent("x", "scaling", 0, 5, bc, params)
    s = _snap(cfg, grid)
    scaling.apply(ev, s, grid.bus_index)
    i = grid.bus_index[9]
    assert s.meas_bus.va[i] == pytest.approx(s.clean_meas_bus.va[i] * params["factors"][9]["Ang"])


def test_replay_requires_history(cfg, grid):
    rng = np.random.default_rng(2)
    with pytest.raises(replay.ReplayHistoryError):
        replay.create(rng, [4], ["V"], cfg["attacks"]["replay"], history_len=10)
    bc, params = replay.create(rng, [4], ["V"], cfg["attacks"]["replay"], history_len=400)
    ev = AttackEvent("x", "replay", 400, 5, bc, params)
    old = _snap(cfg, grid, step=400 - params["lag_steps"], seed=5)
    s = _snap(cfg, grid, step=400, seed=6)
    replay.apply(ev, s, grid.bus_index, history={old.step: old.clean_meas_bus})
    assert s.meas_bus.vm[grid.bus_index[4]] == old.clean_meas_bus.vm[grid.bus_index[4]]
    with pytest.raises(replay.ReplayHistoryError):
        replay.apply(ev, _snap(cfg, grid, step=401), grid.bus_index, history={})


def test_coordinated_cluster_is_connected_and_correlated(cfg, grid):
    rng = np.random.default_rng(3)
    bc, params = coordinated.create(rng, grid, 5, ["V"], cfg["attacks"]["coordinated"], 1.0)
    buses = set(bc)
    assert len(buses) == 5
    # cluster is connected in the real network graph
    seed = params["seed_bus"]
    assert seed in buses
    for b in buses:
        assert params["hops"][b] <= 4
    signs = {np.sign(params["offsets"][b]["V"]) for b in buses}
    assert len(signs) == 1                                    # correlated (same sign)
    vals = [params["offsets"][b]["V"] for b in buses]
    assert len(set(np.round(vals, 9))) == 5                   # but not identical values


def _run_engine(cfg, steps, seed=11):
    eng = SimulationEngine(cfg, "test", seed)
    rows = [eng.step(t, "t", float(t))[0] for t in range(steps)]
    return eng, rows


def test_all_attack_types_generated_and_labels_consistent(cfg, grid):
    eng, rows = _run_engine(cfg, 1500)
    types = {r["attack_type"] for r in rows}
    assert {"normal", "additive", "scaling", "replay", "coordinated"} <= types
    for r in rows:
        buses = [r[f"attack_bus_{b}"] for b in range(1, 31)]
        if any(buses):
            assert r["attack_active"] == 1
        if r["attack_active"] == 0:
            assert r["attack_type"] == "normal" and r["attack_severity"] == 0 and r["attack_id"] == ""
        else:
            assert r["attack_severity"] > 0 and r["attack_id"]
            assert r["attacked_bus_count"] == sum(buses)
            # labels match actual manipulation
            for b in range(1, 31):
                dv = abs(r[f"V_meas_bus_{b}"] - r[f"V_true_bus_{b}"])
                if not r[f"attack_bus_{b}"]:
                    assert dv < 6 * cfg["noise"]["voltage_pu"]


def test_attack_sequence_not_fixed(cfg):
    _, a = _run_engine(cfg, 800, seed=1)
    _, b = _run_engine(cfg, 800, seed=2)
    sa = [r["attack_active"] for r in a]
    sb = [r["attack_active"] for r in b]
    assert sa != sb


def test_reproducible_with_seed(cfg):
    _, a = _run_engine(cfg, 300, seed=5)
    _, b = _run_engine(cfg, 300, seed=5)
    assert [r["V_meas_bus_10"] for r in a] == [r["V_meas_bus_10"] for r in b]
    assert [r["attack_id"] for r in a] == [r["attack_id"] for r in b]


def test_noise_change_keeps_attack_sequence(cfg):
    from config.settings import apply_overrides
    _, a = _run_engine(cfg, 600, seed=4)
    _, b = _run_engine(apply_overrides(cfg, {"noise.voltage_pu": 0.004}), 600, seed=4)
    assert [r["attack_type"] for r in a] == [r["attack_type"] for r in b]


def test_true_values_untouched_by_attacks(cfg):
    from config.settings import apply_overrides
    _, a = _run_engine(cfg, 400, seed=9)
    _, b = _run_engine(apply_overrides(cfg, {"attacks.enabled": False}), 400, seed=9)
    assert any(r["attack_active"] for r in a)
    assert not any(r["attack_active"] for r in b)
    for ra, rb in zip(a, b):
        assert ra["V_true_bus_5"] == rb["V_true_bus_5"]
        assert ra["Ang_true_bus_12"] == rb["Ang_true_bus_12"]
        assert ra["P_from_true_line_7"] == rb["P_from_true_line_7"]
