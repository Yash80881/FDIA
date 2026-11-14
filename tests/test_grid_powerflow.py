"""Tests 1-8: PYPOWER installation, IEEE 30-bus case, AC power flow, loads, generators."""
import numpy as np
import pytest

from simulator.power_flow import run_ac_power_flow, PowerFlowError, network_injections
from simulator.load_profiles import LoadProfileModel
from simulator.dispatch import GeneratorDispatcher, check_generator_limits


def test_pypower_imports_and_version():
    import pypower
    from pypower.case30 import case30          # noqa: F401
    from pypower.api import runpf, ppoption    # noqa: F401
    from importlib.metadata import version
    assert version("PYPOWER") == "5.1.21"


def test_case_dimensions(grid):
    assert grid.n_bus == 30
    assert grid.n_branch == 41
    assert grid.n_gen == 6
    assert grid.bus_numbers == list(range(1, 31))


def test_generator_records(grid):
    assert grid.ref_bus == 1
    assert sorted(grid.generator_buses) == [1, 2, 13, 22, 23, 27]
    for g in grid.generators:
        assert g["pmin_mw"] <= g["pg_mw"] <= g["pmax_mw"] or g["bus"] == grid.ref_bus
        assert g["qmin_mvar"] < g["qmax_mvar"]


def test_case_data_matches_pypower(grid):
    from pypower.case30 import case30
    ref = case30()
    assert np.array_equal(ref["branch"], grid.base_case["branch"])
    assert np.array_equal(ref["bus"], grid.base_case["bus"])
    assert abs(grid.base_pd.sum() - 189.2) < 1e-9
    assert all(b.rate_a_mva and b.rate_a_mva > 0 for b in grid.branches)
    assert grid.shunts == {5: (0.0, 0.19), 24: (0.0, 0.04)}


def test_connectivity(grid):
    assert set(grid.hop_distances(1)) == set(grid.bus_numbers)   # connected network
    assert sum(len(v) for v in grid.adjacency.values()) == 2 * 41


def test_base_power_flow_converges(grid):
    pf = run_ac_power_flow(grid.new_case())
    assert pf.vm_pu.shape == (30,)
    assert pf.va_deg[grid.bus_index[grid.ref_bus]] == 0.0
    # injections equal Pg - Pd and sum to the losses
    losses = pf.pg_mw.sum() - pf.pd_mw.sum()
    assert abs(pf.p_inj_mw.sum() - losses) < 1e-3
    assert np.allclose(pf.pf_mw + pf.pt_mw >= -1e-6, True)          # resistive losses >= 0
    assert abs((pf.pf_mw + pf.pt_mw).sum() - losses) < 1e-3


def test_failed_power_flow_raises(grid):
    case = grid.new_case()
    case["bus"][:, 2] *= 40          # absurd load -> divergence
    case["bus"][:, 3] *= 40
    with pytest.raises(PowerFlowError):
        run_ac_power_flow(case)


def test_network_injection_helper(grid):
    pf = run_ac_power_flow(grid.new_case())
    s = network_injections(pf.solved_case, pf.vm_pu, pf.va_deg)
    assert np.allclose(s.real, pf.p_inj_mw)


def test_load_variation_changes_total(cfg, grid):
    lp = LoadProfileModel(grid, cfg, np.random.default_rng(1))
    totals = [lp.step(t).total_p_mw for t in range(400)]
    q = [lp.step(t).total_q_mvar for t in range(400, 420)]
    assert np.std(totals) > 2.0                    # genuinely changing demand
    assert len(set(np.round(totals, 6))) == 400   # never constant
    assert np.std(q) > 0
    m = lp.step(500).multipliers
    assert np.all(m >= cfg["load_profile"]["min_multiplier"]) and np.all(m <= cfg["load_profile"]["max_multiplier"])


def test_dispatch_respects_limits(cfg, grid):
    rng = np.random.default_rng(3)
    d = GeneratorDispatcher(grid, cfg, rng)
    for load in (120, 189.2, 260):
        pg = d.dispatch(load)
        for k, g in enumerate(grid.generators):
            if g["bus"] != grid.ref_bus:
                assert g["pmin_mw"] - 1e-9 <= pg[k] <= g["pmax_mw"] + 1e-9


def test_generator_limit_checker(grid):
    pg = np.array([g["pg_mw"] for g in grid.generators])
    qg = np.zeros(6)
    pg[1] = 1000
    issues = check_generator_limits(pg, qg, grid)
    assert any("Gen 2" in s for s in issues)
