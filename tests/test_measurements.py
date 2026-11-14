"""Tests 9-13, 19: bus/line measurement extraction, currents, loading, noise, truth preservation."""
import numpy as np

from simulator.power_flow import run_ac_power_flow
from simulator.measurement_generator import MeasurementGenerator, line_loading


def _snap(cfg, grid, seed=0):
    pf = run_ac_power_flow(grid.new_case())
    mg = MeasurementGenerator(grid, cfg, np.random.default_rng(seed))
    return pf, mg, mg.snapshot(0, pf)


def test_bus_measurements_match_power_flow(cfg, grid):
    pf, _, s = _snap(cfg, grid)
    assert np.array_equal(s.true_bus.vm, pf.vm_pu)
    assert np.array_equal(s.true_bus.va, pf.va_deg)
    # bus 2 is a PV generator bus: injection = Pg - Pd
    i = grid.bus_index[2]
    pg2 = pf.pg_mw[grid.generator_buses.index(2)]
    assert abs(s.true_bus.p[i] - (pg2 - grid.base_pd[i])) < 1e-3
    # pure load bus 3 has negative injection
    assert s.true_bus.p[grid.bus_index[3]] < 0


def test_line_measurements_and_current(cfg, grid):
    pf, _, s = _snap(cfg, grid)
    assert np.array_equal(s.true_line.p_from, pf.pf_mw)
    k = 0   # line 1: bus 1 -> 2
    sf = np.hypot(pf.pf_mw[k], pf.qf_mvar[k])
    expected = sf / (np.sqrt(3) * pf.vm_pu[0] * 135.0) * 1000
    assert abs(s.true_line.i_from[k] - expected) < 1e-9
    assert s.true_line.i_from.shape == (41,)


def test_line_loading(cfg, grid):
    pf, _, s = _snap(cfg, grid)
    rate = grid.rate_a
    exp = np.maximum(np.hypot(pf.pf_mw, pf.qf_mvar), np.hypot(pf.pt_mw, pf.qt_mvar)) / rate * 100
    assert np.allclose(s.true_line.loading, exp)
    # unrated line -> NaN, never an invented limit
    out = line_loading(np.array([10.0]), np.array([0.0]), np.array([-10.0]), np.array([0.0]), np.array([np.nan]))
    assert np.isnan(out[0])


def test_noise_is_present_and_bounded(cfg, grid):
    pf, mg, _ = _snap(cfg, grid)
    diffs = []
    for seed in range(50):
        mg.rng = np.random.default_rng(seed)
        s = mg.snapshot(0, pf)
        diffs.append(s.meas_bus.vm - s.true_bus.vm)
    d = np.array(diffs)
    sigma = cfg["noise"]["voltage_pu"]
    assert 0.7 * sigma < d.std() < 1.3 * sigma
    assert np.all(s.meas_line.p_from != s.true_line.p_from)


def test_truth_is_independent_copy(cfg, grid):
    pf, _, s = _snap(cfg, grid)
    s.meas_bus.vm[:] = 0.0
    s.meas_line.p_from[:] = 0.0
    assert np.all(s.true_bus.vm > 0.9)
    assert np.array_equal(s.true_line.p_from, pf.pf_mw)
    assert np.all(s.clean_meas_bus.vm > 0.9)
