"""WLS state estimation, bad-data tests and mitigation."""
import numpy as np
import pytest

from physics.state_estimation import WLSStateEstimator, EstimationError
from physics.residual_analysis import analyze, identify_bad_data
from physics.mitigation import MitigationAdvisor
from simulator.power_flow import run_ac_power_flow
from simulator.measurement_generator import MeasurementGenerator


@pytest.fixture(scope="module")
def se():
    return WLSStateEstimator()


def _ms(se, cfg, grid, seed=0, dv=None):
    pf = run_ac_power_flow(grid.new_case())
    s = MeasurementGenerator(grid, cfg, np.random.default_rng(seed)).snapshot(0, pf)
    b, l = s.meas_bus, s.meas_line
    vm = b.vm.copy()
    if dv:
        for bus, d in dv.items():
            vm[grid.bus_index[bus]] += d
    return pf, se.build_measurements(vm, b.va, b.p, b.q, l.p_from, l.q_from, l.p_to, l.q_to, cfg["noise"])


def test_noise_free_estimate_is_exact(se, cfg, grid):
    pf = run_ac_power_flow(grid.new_case())
    mg = MeasurementGenerator(grid, cfg, np.random.default_rng(0))
    b, l = mg.true_measurements(pf)
    ms = se.build_measurements(b.vm, b.va, b.p, b.q, l.p_from, l.q_from, l.p_to, l.q_to, cfg["noise"])
    r = se.estimate(ms)
    assert r.converged and r.observable
    assert np.max(np.abs(r.vm - pf.vm_pu)) < 1e-6
    assert np.max(np.abs(r.va_deg - pf.va_deg)) < 1e-5
    assert r.objective < 1e-6


def test_noisy_estimate_statistics(se, cfg, grid):
    js = []
    for seed in range(15):
        pf, ms = _ms(se, cfg, grid, seed)
        r = se.estimate(ms)
        js.append(r.objective)
        assert np.max(np.abs(r.vm - pf.vm_pu)) < 3 * cfg["noise"]["voltage_pu"]
    # J ~ chi2(m - n): its mean is close to the degrees of freedom
    assert 0.8 * r.dof < np.mean(js) < 1.2 * r.dof


def test_bad_data_detected_and_identified(se, cfg, grid):
    _, ms = _ms(se, cfg, grid, 1, dv={10: 0.03})
    r = se.estimate(ms)
    rep = analyze(r, 0.01, 3.0)
    assert rep.chi2_alarm or rep.lnr_alarm
    assert rep.suspicious[0]["measurement"] == "V_bus_10"
    ident = identify_bad_data(se, ms)
    assert ident["removed"][0]["measurement"] == "V_bus_10"
    assert ident["final_chi2_ok"]


def test_unobservable_detected(se, cfg, grid):
    _, ms = _ms(se, cfg, grid)
    keep = np.array([k in ("V",) for k in ms.kinds])      # only voltage magnitudes -> angles unobservable
    with pytest.raises(EstimationError):
        se.estimate(ms.subset(keep))


def test_mitigation_recovers_voltage(se, cfg, grid):
    from simulator.continuous_simulator import SimulationEngine
    eng = SimulationEngine(cfg, "m", 5)
    row = eng.step(0, "", 0.0)[0]
    row["V_meas_bus_12"] += 0.03
    adv = MitigationAdvisor(se, cfg)
    out = adv.recommend(row, predicted_buses=[12])
    assert out["status"] == "ok" and out["operator_supervised"]
    assert "V_bus_12" in out["excluded_measurements"]
    ev = out["evaluation_against_ground_truth"]
    assert ev["recovered_v_rmse"] < ev["received_v_rmse"]
    assert any(a["action"] == "isolate-sensors" for a in out["recommendations"])
    assert not out["after"]["chi2_alarm"]
