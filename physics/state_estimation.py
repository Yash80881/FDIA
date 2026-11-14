"""AC Weighted-Least-Squares (WLS) state estimation for the IEEE 30-bus case.

Measurement model   z = h(x) + e,   e ~ N(0, R),  R = diag(sigma^2)
State               x = [theta_2..theta_30 (rad), V_1..V_30 (pu)]   (bus 1 = reference, theta_1 = 0)
Objective           J(x) = (z - h(x))^T R^-1 (z - h(x))
Gauss-Newton        G dx = H^T R^-1 (z - h(x)),   G = H^T R^-1 H

Measurement set (all in per unit on the 100 MVA base, angles in rad):
    V_i (30), PMU angles theta_i (30, optional), P_i / Q_i injections (30 + 30),
    P/Q at both ends of all 41 branches (164)                     -> m = 284

h(x) and the Jacobian H use PYPOWER's own network matrices
(``makeYbus``) and derivative routines (``dSbus_dV``, ``dSbr_dV``), so the
estimator shares exactly the same branch model as the power flow.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import issparse
from pypower.api import makeYbus
from pypower.dSbus_dV import dSbus_dV
from pypower.dSbr_dV import dSbr_dV
from pypower.ext2int import ext2int

from simulator.grid_model import GridModel


class EstimationError(RuntimeError):
    pass


def _dense(a):
    return a.toarray() if issparse(a) else np.asarray(a)


@dataclass
class MeasurementSet:
    """Measurement vector with names, types, owning bus and standard deviations (pu)."""
    z: np.ndarray
    sigma: np.ndarray
    names: list[str]
    kinds: list[str]        # V, Ang, P, Q, Pf, Qf, Pt, Qt
    index: np.ndarray       # bus row index (V/Ang/P/Q) or branch index (flows)
    bus: list[int]          # bus the sensor is installed at (external number)

    def subset(self, keep: np.ndarray) -> "MeasurementSet":
        keep = np.asarray(keep, bool)
        return MeasurementSet(self.z[keep], self.sigma[keep], [n for n, k in zip(self.names, keep) if k],
                              [n for n, k in zip(self.kinds, keep) if k], self.index[keep],
                              [n for n, k in zip(self.bus, keep) if k])

    def __len__(self):
        return len(self.z)


@dataclass
class EstimationResult:
    converged: bool
    iterations: int
    vm: np.ndarray                  # estimated |V| (pu)
    va_deg: np.ndarray              # estimated angle (deg)
    residuals: np.ndarray           # z - h(x_hat) (pu)
    normalized_residuals: np.ndarray  # |r_i| / sqrt(Omega_ii)  (NaN for critical measurements)
    objective: float                # J(x_hat)
    dof: int                        # m - n
    observable: bool
    measurements: MeasurementSet
    h: np.ndarray = field(repr=False, default=None)


class WLSStateEstimator:
    def __init__(self, grid: GridModel | None = None, max_iter: int = 15, tol: float = 1e-6):
        self.grid = grid or GridModel()
        self.max_iter, self.tol = max_iter, tol
        ppc = ext2int(self.grid.new_case())
        self.base = float(ppc["baseMVA"])
        self.branch = ppc["branch"]
        ybus, yf, yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
        self.ybus, self.yf, self.yt = ybus.tocsr(), yf.tocsr(), yt.tocsr()
        self.nb = self.grid.n_bus
        self.ref = self.grid.bus_index[self.grid.ref_bus]
        self.non_ref = np.array([i for i in range(self.nb) if i != self.ref])
        self.fi, self.ti = self.grid.branch_from_idx, self.grid.branch_to_idx

    # ----------------------------------------------------------- measurements
    def build_measurements(self, vm, va_deg, p_mw, q_mvar, p_from, q_from, p_to, q_to, noise: dict,
                           use_angles: bool = True) -> MeasurementSet:
        """Assemble the measurement vector from received measurements (MW/MVAr/pu/deg)."""
        g = self.grid
        z, sig, names, kinds, idx, bus = [], [], [], [], [], []

        def add(kind, values, sigmas, owners, label):
            for j, (v, s) in enumerate(zip(values, sigmas)):
                z.append(v)
                sig.append(s)
                kinds.append(kind)
                idx.append(j)
                bus.append(owners[j])
                names.append(f"{label}_{j + 1}")

        buses = g.bus_numbers
        f_bus = [b.from_bus for b in g.branches]
        t_bus = [b.to_bus for b in g.branches]
        add("V", vm, np.full(self.nb, noise["voltage_pu"]), buses, "V_bus")
        if use_angles:
            add("Ang", np.deg2rad(va_deg), np.full(self.nb, np.deg2rad(noise["angle_deg"])), buses, "Ang_bus")

        def psig(vals):
            return (noise["power_relative"] * np.abs(vals) + noise["power_floor_mw"]) / self.base

        for kind, vals, owners, label in (("P", p_mw, buses, "P_bus"), ("Q", q_mvar, buses, "Q_bus"),
                                          ("Pf", p_from, f_bus, "P_from_line"), ("Qf", q_from, f_bus, "Q_from_line"),
                                          ("Pt", p_to, t_bus, "P_to_line"), ("Qt", q_to, t_bus, "Q_to_line")):
            vals = np.asarray(vals, float)
            add(kind, vals / self.base, psig(vals), owners, label)
        return MeasurementSet(np.array(z), np.array(sig), names, kinds, np.array(idx), bus)

    # --------------------------------------------------------- model h(x), H
    def _h_and_H(self, ms: MeasurementSet, vm: np.ndarray, va: np.ndarray):
        V = vm * np.exp(1j * va)
        sbus = V * np.conj(self.ybus @ V)
        dS_dVm, dS_dVa = dSbus_dV(self.ybus, V)
        dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St = dSbr_dV(self.branch, self.yf, self.yt, V)
        dS_dVm, dS_dVa = _dense(dS_dVm), _dense(dS_dVa)
        dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm = map(_dense, (dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm))
        nb = self.nb
        m = len(ms)
        h = np.zeros(m)
        H = np.zeros((m, 2 * nb))   # columns: [dtheta (nb), dV (nb)]; ref column removed later
        kinds = np.asarray(ms.kinds)
        idx = ms.index
        blocks = {
            "P": (np.real, sbus, dS_dVa, dS_dVm), "Q": (np.imag, sbus, dS_dVa, dS_dVm),
            "Pf": (np.real, Sf, dSf_dVa, dSf_dVm), "Qf": (np.imag, Sf, dSf_dVa, dSf_dVm),
            "Pt": (np.real, St, dSt_dVa, dSt_dVm), "Qt": (np.imag, St, dSt_dVa, dSt_dVm),
        }
        rows = np.flatnonzero(kinds == "V")
        h[rows] = vm[idx[rows]]
        H[rows, nb + idx[rows]] = 1.0
        rows = np.flatnonzero(kinds == "Ang")
        h[rows] = va[idx[rows]]
        H[rows, idx[rows]] = 1.0
        for kind, (part, val, d_va, d_vm) in blocks.items():
            rows = np.flatnonzero(kinds == kind)
            if len(rows) == 0:
                continue
            j = idx[rows]
            h[rows] = part(np.asarray(val).ravel()[j])
            H[rows, :nb] = part(d_va[j])
            H[rows, nb:] = part(d_vm[j])
        keep_cols = np.concatenate([self.non_ref, nb + np.arange(nb)])
        return h, H[:, keep_cols]

    # --------------------------------------------------------------- solve
    def estimate(self, ms: MeasurementSet, vm0=None, va0_deg=None) -> EstimationResult:
        nb = self.nb
        vm = np.ones(nb) if vm0 is None else np.asarray(vm0, float).copy()
        va = np.zeros(nb) if va0_deg is None else np.deg2rad(np.asarray(va0_deg, float))
        va[self.ref] = 0.0
        W = 1.0 / ms.sigma ** 2
        n_state = len(self.non_ref) + nb
        converged = False
        it = 0
        H = None
        for it in range(1, self.max_iter + 1):
            h, H = self._h_and_H(ms, vm, va)
            r = ms.z - h
            G = H.T @ (W[:, None] * H)
            try:
                dx = np.linalg.solve(G, H.T @ (W * r))
            except np.linalg.LinAlgError as exc:
                raise EstimationError("Gain matrix is singular: the measurement set is not observable") from exc
            va[self.non_ref] += dx[:len(self.non_ref)]
            vm += dx[len(self.non_ref):]
            if np.max(np.abs(dx)) < self.tol:
                converged = True
                break
        h, H = self._h_and_H(ms, vm, va)
        r = ms.z - h
        G = H.T @ (W[:, None] * H)
        rank = np.linalg.matrix_rank(H)
        observable = rank == n_state
        # residual covariance Omega = R - H G^-1 H^T  -> normalised residuals
        try:
            Ginv = np.linalg.inv(G)
            omega = ms.sigma ** 2 - np.einsum("ij,jk,ik->i", H, Ginv, H)
            with np.errstate(invalid="ignore", divide="ignore"):
                rn = np.where(omega > 1e-14 * np.maximum(ms.sigma ** 2, 1e-30), np.abs(r) / np.sqrt(omega), np.nan)
        except np.linalg.LinAlgError:
            rn = np.full(len(r), np.nan)
        return EstimationResult(converged, it, vm.copy(), np.rad2deg(va), r, rn, float(np.sum(W * r * r)),
                                len(ms) - n_state, bool(observable), ms, h)

    def flows_from_state(self, vm: np.ndarray, va_deg: np.ndarray) -> dict:
        """Branch flows / injections (MW, MVAr) implied by a state estimate."""
        V = vm * np.exp(1j * np.deg2rad(va_deg))
        sf = V[self.fi] * np.conj(self.yf @ V) * self.base
        st = V[self.ti] * np.conj(self.yt @ V) * self.base
        sb = V * np.conj(self.ybus @ V) * self.base
        return {"p_from": sf.real, "q_from": sf.imag, "p_to": st.real, "q_to": st.imag,
                "p_inj": sb.real, "q_inj": sb.imag}
