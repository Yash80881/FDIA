"""Physics-derived ML features from WLS state estimation (feature group ``physics``).

For every snapshot a full AC WLS state estimate is computed from the received
measurements (same estimator as the verification module) and the normalised
residuals are turned into features:

    rnV_bus_i, rnAng_bus_i, rnP_bus_i, rnQ_bus_i   signed normalised residual
                                                   r_i / sqrt(Omega_ii) of bus i's sensors
    rnFlowMax_bus_i                                max |r_N| of the branch-flow sensors
                                                   installed at bus i's terminals
    phys_J_ratio                                   J(x_hat) / (m - n)   (~1 for pure noise)
    phys_rN_max                                    largest |r_N| in the snapshot

These depend only on the received measurements of the *same* snapshot, the
network model and the sensor-accuracy specification, so training and live
inference use exactly the same computation.  This is the "physics-informed"
input that lets the Random Forest localise an inconsistency to a bus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from physics.state_estimation import WLSStateEstimator, EstimationError
from simulator import schema as S

KIND_PREFIX = {"V": "rnV", "Ang": "rnAng", "P": "rnP", "Q": "rnQ"}


def physics_feature_names(n_bus: int = S.N_BUS) -> list[str]:
    names = []
    for b in range(1, n_bus + 1):
        names += [f"rnV_bus_{b}", f"rnAng_bus_{b}", f"rnP_bus_{b}", f"rnQ_bus_{b}", f"rnFlowMax_bus_{b}"]
    return names + ["phys_J_ratio", "phys_rN_max"]


class PhysicsFeatureBuilder:
    def __init__(self, noise: dict, se: WLSStateEstimator | None = None, use_angles: bool = True):
        self.noise = noise
        self.se = se or WLSStateEstimator(max_iter=10, tol=1e-5)
        self.use_angles = use_angles
        self.names = physics_feature_names(self.se.nb)

    def row_features(self, vm, va, p, q, pf_, qf_, pt_, qt_) -> np.ndarray:
        ms = self.se.build_measurements(vm, va, p, q, pf_, qf_, pt_, qt_, self.noise, use_angles=self.use_angles)
        out = np.zeros(len(self.names))
        try:
            res = self.se.estimate(ms, vm, va)
        except EstimationError:
            out[:] = np.nan
            return out
        rn = np.nan_to_num(res.normalized_residuals, nan=0.0)
        signed = np.sign(res.residuals) * rn
        nb = self.se.nb
        flow_max = np.zeros(nb)
        kinds = ms.kinds
        for i, (kind, j, bus) in enumerate(zip(kinds, ms.index, ms.bus)):
            if kind in KIND_PREFIX:
                out[j * 5 + ("V", "Ang", "P", "Q").index(kind)] = signed[i]
            else:
                bi = self.se.grid.bus_index[bus]
                flow_max[bi] = max(flow_max[bi], rn[i])
        out[4:nb * 5:5] = flow_max
        out[-2] = res.objective / max(res.dof, 1)
        out[-1] = float(rn.max())
        return out

    def build(self, df: pd.DataFrame, rows: np.ndarray | None = None) -> pd.DataFrame:
        """Physics features for ``df`` (all rows, or only the positional ``rows``)."""
        cols = {k: df[S.bus_cols(p)].to_numpy(float) for k, p in
                (("vm", "V_meas"), ("va", "Ang_meas"), ("p", "P_meas"), ("q", "Q_meas"))}
        for k, q in (("pf", "P_from"), ("qf", "Q_from"), ("pt", "P_to"), ("qt", "Q_to")):
            cols[k] = df[S.line_cols(q)].to_numpy(float)
        F = np.full((len(df), len(self.names)), np.nan)
        idx = range(len(df)) if rows is None else rows
        for r in idx:
            F[r] = self.row_features(cols["vm"][r], cols["va"][r], cols["p"][r], cols["q"][r],
                                     cols["pf"][r], cols["qf"][r], cols["pt"][r], cols["qt"][r])
        return pd.DataFrame(F, columns=self.names, index=df.index)
