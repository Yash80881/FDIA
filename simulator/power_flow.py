"""AC power-flow wrapper around PYPOWER ``runpf``.

Every call checks convergence and the numerical sanity of the solution.  A
non-converged or non-finite solution raises :class:`PowerFlowError`; callers
must never build measurements from such a result.
"""
from __future__ import annotations

import io
import contextlib
from dataclasses import dataclass

import numpy as np
from pypower.api import runpf, ppoption, makeYbus
from pypower.ext2int import ext2int
from pypower import idx_bus as IB
from pypower import idx_brch as IBR
from pypower import idx_gen as IG


class PowerFlowError(RuntimeError):
    """AC power flow did not converge or produced an invalid solution."""


@dataclass
class PowerFlowResult:
    """Solved AC operating point (external bus numbering, PYPOWER units)."""

    vm_pu: np.ndarray        # (n_bus,)
    va_deg: np.ndarray       # (n_bus,)
    p_inj_mw: np.ndarray     # net bus injection P = Pg - Pd (network side, shunts in network)
    q_inj_mvar: np.ndarray   # net bus injection Q = Qg - Qd
    pf_mw: np.ndarray        # branch flows at "from" end (into the branch)
    qf_mvar: np.ndarray
    pt_mw: np.ndarray        # branch flows at "to" end (into the branch)
    qt_mvar: np.ndarray
    pg_mw: np.ndarray        # generator outputs
    qg_mvar: np.ndarray
    pd_mw: np.ndarray        # bus loads used in this solution
    qd_mvar: np.ndarray
    iterations: int
    max_mismatch_mva: float
    solved_case: dict


def pf_options(enforce_q_limits: bool = False):
    return ppoption(VERBOSE=0, OUT_ALL=0, ENFORCE_Q_LIMS=int(bool(enforce_q_limits)))


def network_injections(case: dict, vm: np.ndarray, va_deg: np.ndarray) -> np.ndarray:
    """Complex injections S = V conj(Ybus V) in MVA for the given voltages.

    Ybus includes the bus shunts, so S equals (Pg - Pd) + j(Qg - Qd) at a
    converged power-flow solution.
    """
    ppc = ext2int(case)
    ybus, _, _ = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
    v = vm * np.exp(1j * np.deg2rad(va_deg))
    return v * np.conj(ybus @ v) * ppc["baseMVA"]


def run_ac_power_flow(case: dict, enforce_q_limits: bool = False) -> PowerFlowResult:
    """Run ``runpf`` and return a validated :class:`PowerFlowResult`."""
    # PYPOWER may print warnings (e.g. singular Jacobian) to stdout; capture them
    # so they can be attached to the error instead of polluting the console.
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            results, success = runpf(case, pf_options(enforce_q_limits))
    except Exception as exc:  # numerical failure inside PYPOWER
        raise PowerFlowError(f"runpf raised {type(exc).__name__}: {exc}") from exc
    if not success:
        raise PowerFlowError("AC power flow did not converge. " + buf.getvalue().strip()[:300])

    bus, branch, gen = results["bus"], results["branch"], results["gen"]
    vm = bus[:, IB.VM].copy()
    va = bus[:, IB.VA].copy()
    s_inj = network_injections(results, vm, va)
    arrays = [vm, va, branch[:, IBR.PF], branch[:, IBR.QF], branch[:, IBR.PT],
              branch[:, IBR.QT], gen[:, IG.PG], gen[:, IG.QG], s_inj.real, s_inj.imag]
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise PowerFlowError("Power-flow solution contains non-finite values")
    if np.any(vm <= 0.5) or np.any(vm >= 1.5):
        raise PowerFlowError(f"Unrealistic voltage magnitudes (min {vm.min():.3f}, max {vm.max():.3f} pu)")

    # Verify the bus power balance: network injection must equal Pg - Pd.
    pg_bus = np.zeros(len(vm))
    qg_bus = np.zeros(len(vm))
    bus_row = {int(b): i for i, b in enumerate(bus[:, IB.BUS_I])}
    for g in gen:
        if g[IG.GEN_STATUS] > 0:
            pg_bus[bus_row[int(g[IG.GEN_BUS])]] += g[IG.PG]
            qg_bus[bus_row[int(g[IG.GEN_BUS])]] += g[IG.QG]
    spec = (pg_bus - bus[:, IB.PD]) + 1j * (qg_bus - bus[:, IB.QD])
    mismatch = float(np.max(np.abs(s_inj - spec)))
    if mismatch > 1e-3 * results["baseMVA"]:
        raise PowerFlowError(f"Bus power mismatch {mismatch:.4f} MVA after power flow")

    return PowerFlowResult(
        vm_pu=vm, va_deg=va,
        p_inj_mw=s_inj.real.copy(), q_inj_mvar=s_inj.imag.copy(),
        pf_mw=branch[:, IBR.PF].copy(), qf_mvar=branch[:, IBR.QF].copy(),
        pt_mw=branch[:, IBR.PT].copy(), qt_mvar=branch[:, IBR.QT].copy(),
        pg_mw=gen[:, IG.PG].copy(), qg_mvar=gen[:, IG.QG].copy(),
        pd_mw=bus[:, IB.PD].copy(), qd_mvar=bus[:, IB.QD].copy(),
        iterations=int(results.get("iterations", 0) or 0),
        max_mismatch_mva=mismatch,
        solved_case=results,
    )
