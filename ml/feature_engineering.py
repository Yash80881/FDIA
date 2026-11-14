"""Electrical feature engineering (identical for training and live inference).

Inputs are ONLY received measurements (``schema.ML_INPUT_COLUMNS``) plus static
network parameters from the PYPOWER case (branch admittances, topology).
True values, labels, attack metadata and severity are never touched.

Feature groups (used for ablation experiments):

* ``bus``       received V, angle, P and Q of all 30 buses
* ``line``      received line P/Q (both ends), currents and loading of all 41 lines
* ``electrical`` physics-informed consistency features:
    - voltage / angle differences across every branch
    - branch-flow mismatch: measured P/Q_from, P/Q_to minus the AC flow
      computed from the *measured* bus voltages/angles with the branch model
      S_from = V_f conj(Y_f V)  (independent sensors disagreeing => manipulation)
    - bus-injection mismatch: measured P/Q injection minus V conj(Ybus V)
    - KCL mismatch: measured injection minus the sum of measured line flows
    - per-bus summaries of incident-branch mismatches (max / min / mean)
    - branch losses (P_from + P_to) and global voltage/angle statistics
* ``physics``   WLS state-estimation normalised residuals per bus sensor
  (see :mod:`ml.physics_features`)
* ``temporal``  step-to-step changes, rolling std and deviation from the
  rolling mean of V and angle, changes of line flows and of bus mismatches

Temporal features use only past/current rows of the same run (no look-ahead),
so the same function works on a full run (training) or on the last few rows
(live inference).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pypower.api import makeYbus
from pypower.ext2int import ext2int

from simulator.grid_model import GridModel
from simulator import schema as S

FEATURE_GROUPS = ("bus", "line", "electrical", "temporal", "physics")

# Physics-informed consistency residuals are expressed as z-scores using the
# *sensor accuracy specification* (the same sigmas a WLS estimator puts in R):
#   z = mismatch / sigma_mismatch,
#   sigma_mismatch^2 = sum_j (dh/dx_j * sigma_x_j)^2 + sigma_meas^2
# where the sensitivity of the computed flow/injection to the V/angle sensor
# noise is linearised at the base-case operating point.  This makes the
# features comparable across sensor classes (noise levels).


@dataclass
class FeatureSet:
    X: pd.DataFrame
    groups: dict[str, list[str]]          # group name -> feature columns
    bus_local: dict[int, list[str]]       # bus -> local feature columns (localisation)


class FeatureBuilder:
    def __init__(self, grid: GridModel | None = None, rolling_window: int = 5, noise: dict | None = None,
                 physics: bool = False):
        self.grid = grid or GridModel()
        self.window = int(rolling_window)
        if noise is None:
            from config.settings import load_settings
            noise = load_settings()["noise"]
        self.noise = {k: float(noise[k]) for k in ("voltage_pu", "angle_deg", "power_relative", "power_floor_mw")}
        g = self.grid
        ppc = ext2int(g.new_case())
        self.base = float(ppc["baseMVA"])
        ybus, yf, yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
        self.ybus = ybus.toarray()
        self.yf = yf.toarray()
        self.yt = yt.toarray()
        self._model_variances(ppc, ybus, yf, yt)
        self.physics = None
        if physics:
            from ml.physics_features import PhysicsFeatureBuilder
            self.physics = PhysicsFeatureBuilder(self.noise)
        self.fi = g.branch_from_idx
        self.ti = g.branch_to_idx
        self.n_bus, self.n_line = g.n_bus, g.n_branch
        self.incident = {b: [k - 1 for k in g.incident_branches[b]] for b in g.bus_numbers}
        self._names = None

    def _model_variances(self, ppc, ybus, yf, yt) -> None:
        """Variance (MW^2 / MVAr^2) of computed flows/injections caused by V and
        angle sensor noise, linearised at the base-case power-flow solution."""
        from pypower.dSbus_dV import dSbus_dV
        from pypower.dSbr_dV import dSbr_dV
        from simulator.power_flow import run_ac_power_flow
        pf = run_ac_power_flow(self.grid.new_case())
        v0 = pf.vm_pu * np.exp(1j * np.deg2rad(pf.va_deg))
        sv2 = self.noise["voltage_pu"] ** 2
        sa2 = np.deg2rad(self.noise["angle_deg"]) ** 2
        dSb_dVm, dSb_dVa = dSbus_dV(ybus, v0)
        dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, _, _ = dSbr_dV(ppc["branch"], yf, yt, v0)

        def var(d_va, d_vm, part):
            a = part(np.asarray(d_va.todense()))
            m = part(np.asarray(d_vm.todense()))
            return ((a ** 2).sum(1) * sa2 + (m ** 2).sum(1) * sv2) * self.base ** 2

        self.var_model = {
            "Pf": var(dSf_dVa, dSf_dVm, np.real), "Qf": var(dSf_dVa, dSf_dVm, np.imag),
            "Pt": var(dSt_dVa, dSt_dVm, np.real), "Qt": var(dSt_dVa, dSt_dVm, np.imag),
            "P": var(dSb_dVa, dSb_dVm, np.real), "Q": var(dSb_dVa, dSb_dVm, np.imag),
        }

    def _sig(self, values: np.ndarray) -> np.ndarray:
        """Standard deviation of a power sensor reading (MW/MVAr)."""
        return self.noise["power_relative"] * np.abs(values) + self.noise["power_floor_mw"]

    # ------------------------------------------------------------------ core
    def build(self, df: pd.DataFrame, physics_last_only: bool = False) -> FeatureSet:
        """Features for the rows of ``df`` (one run, sorted by step).

        ``physics_last_only``: compute the (costly) WLS physics features only for
        the last row - used in live inference where only that row is predicted.
        """
        if "run_id" in df.columns and df["run_id"].nunique() > 1:
            parts = [self.build(g, physics_last_only) for _, g in df.groupby("run_id", sort=False)]
            X = pd.concat([p.X for p in parts])
            return FeatureSet(X, parts[0].groups, parts[0].bus_local)
        if "step" in df.columns and not df["step"].is_monotonic_increasing:
            df = df.sort_values("step")
        missing = [c for c in S.ML_INPUT_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"Missing measurement columns for features: {missing[:5]}")

        nb, nl = self.n_bus, self.n_line
        vm = df[S.bus_cols("V_meas")].to_numpy(float)
        va = df[S.bus_cols("Ang_meas")].to_numpy(float)
        pm = df[S.bus_cols("P_meas")].to_numpy(float)
        qm = df[S.bus_cols("Q_meas")].to_numpy(float)
        line = {q: df[S.line_cols(q)].to_numpy(float) for q in S.LINE_QUANTITIES}
        line["loading"] = np.nan_to_num(line["loading"], nan=-1.0)   # unrated line marker

        feats: dict[str, np.ndarray] = {}
        groups: dict[str, list[str]] = {g: [] for g in FEATURE_GROUPS}

        def add(group: str, name: str, values: np.ndarray):
            feats[name] = values
            groups[group].append(name)

        # ---- bus / line raw received measurements
        for prefix, arr in (("V_meas", vm), ("Ang_meas", va), ("P_meas", pm), ("Q_meas", qm)):
            for i, c in enumerate(S.bus_cols(prefix)):
                add("bus", c, arr[:, i])
        for q in S.LINE_QUANTITIES:
            for k, c in enumerate(S.line_cols(q)):
                add("line", c, line[q][:, k])

        # ---- physics-informed electrical consistency features
        v = vm * np.exp(1j * np.deg2rad(va))                      # (n, 30) complex
        s_from = v[:, self.fi] * np.conj(v @ self.yf.T) * self.base   # (n, 41) MVA
        s_to = v[:, self.ti] * np.conj(v @ self.yt.T) * self.base
        s_inj = v * np.conj(v @ self.ybus.T) * self.base
        meas = {"Pf": line["P_from"], "Qf": line["Q_from"], "Pt": line["P_to"], "Qt": line["Q_to"]}
        calc = {"Pf": s_from.real, "Qf": s_from.imag, "Pt": s_to.real, "Qt": s_to.imag}
        z = {k: (meas[k] - calc[k]) / np.sqrt(self.var_model[k] + self._sig(meas[k]) ** 2) for k in meas}
        norm_f = np.maximum(np.abs(z["Pf"]), np.abs(z["Qf"]))
        norm_t = np.maximum(np.abs(z["Pt"]), np.abs(z["Qt"]))
        line_norm = np.maximum(norm_f, norm_t)                     # (n, 41) max |z| of the branch
        dv_line = vm[:, self.fi] - vm[:, self.ti]
        da_line = va[:, self.fi] - va[:, self.ti]
        for k in range(nl):
            ln = k + 1
            add("electrical", f"dV_line_{ln}", dv_line[:, k])
            add("electrical", f"dAng_line_{ln}", da_line[:, k])
            for key, arr in z.items():
                add("electrical", f"misZ{key}_line_{ln}", arr[:, k])
            add("electrical", f"misZmax_line_{ln}", line_norm[:, k])
            add("electrical", f"loss_line_{ln}", line["P_from"][:, k] + line["P_to"][:, k])

        inj_p = (pm - s_inj.real) / np.sqrt(self.var_model["P"] + self._sig(pm) ** 2)
        inj_q = (qm - s_inj.imag) / np.sqrt(self.var_model["Q"] + self._sig(qm) ** 2)
        # KCL with measured flows: injection = sum of flows leaving the bus
        flow_p = np.zeros_like(pm)
        flow_q = np.zeros_like(qm)
        var_p = self._sig(pm) ** 2
        var_q = self._sig(qm) ** 2
        for k in range(nl):
            for idx, pk, qk in ((self.fi[k], "P_from", "Q_from"), (self.ti[k], "P_to", "Q_to")):
                flow_p[:, idx] += line[pk][:, k]
                flow_q[:, idx] += line[qk][:, k]
                var_p[:, idx] += self._sig(line[pk][:, k]) ** 2
                var_q[:, idx] += self._sig(line[qk][:, k]) ** 2
        kcl_p = (pm - flow_p) / np.sqrt(var_p)
        kcl_q = (qm - flow_q) / np.sqrt(var_q)
        bus_max = np.zeros((len(df), nb))
        bus_min = np.zeros((len(df), nb))
        bus_mean = np.zeros((len(df), nb))
        for i, b in enumerate(self.grid.bus_numbers):
            # mismatch at this bus's own terminal of each incident branch
            ends = []
            for k in self.incident[b]:
                ends.append(norm_f[:, k] if self.fi[k] == i else norm_t[:, k])
            e = np.column_stack(ends)
            bus_max[:, i], bus_min[:, i], bus_mean[:, i] = e.max(1), e.min(1), e.mean(1)
            add("electrical", f"injZP_bus_{b}", inj_p[:, i])
            add("electrical", f"injZQ_bus_{b}", inj_q[:, i])
            add("electrical", f"kclZP_bus_{b}", kcl_p[:, i])
            add("electrical", f"kclZQ_bus_{b}", kcl_q[:, i])
            add("electrical", f"busMisMax_bus_{b}", bus_max[:, i])
            add("electrical", f"busMisMin_bus_{b}", bus_min[:, i])
            add("electrical", f"busMisMean_bus_{b}", bus_mean[:, i])
            add("electrical", f"Vdev_bus_{b}", vm[:, i] - 1.0)

        add("electrical", "V_mean", vm.mean(1))
        add("electrical", "V_std", vm.std(1))
        add("electrical", "V_max", vm.max(1))
        add("electrical", "V_min", vm.min(1))
        add("electrical", "V_range", vm.max(1) - vm.min(1))
        add("electrical", "dAng_abs_mean", np.abs(da_line).mean(1))
        add("electrical", "dAng_abs_max", np.abs(da_line).max(1))
        add("electrical", "Ang_abs_max", np.abs(va).max(1))
        add("electrical", "P_inj_sum", pm.sum(1))          # = losses if consistent
        add("electrical", "Q_inj_sum", qm.sum(1))
        add("electrical", "loss_sum", (line["P_from"] + line["P_to"]).sum(1))
        zz = np.concatenate([z[k] for k in z] + [inj_p, inj_q], axis=1)
        add("electrical", "z_mean_square", (zz ** 2).mean(1))
        add("electrical", "misZ_mean", line_norm.mean(1))
        add("electrical", "misZ_max", line_norm.max(1))
        add("electrical", "misZ_count_gt3", (line_norm > 3).sum(1).astype(float))
        add("electrical", "injZ_abs_max", np.maximum(np.abs(inj_p), np.abs(inj_q)).max(1))
        add("electrical", "busMisMin_max", bus_min.max(1))

        # ---- temporal features (past and current rows only)
        w = self.window

        def diff(a):
            d = np.diff(a, axis=0, prepend=a[:1])
            return d

        def roll(a, fn):
            return getattr(pd.DataFrame(a).rolling(w, min_periods=1), fn)().to_numpy()

        sv, sa = self.noise["voltage_pu"], self.noise["angle_deg"]
        # temporal features in units of the sensor noise standard deviation
        dvm, dva = diff(vm) / sv, diff(va) / sa
        sd_v, sd_a = roll(vm, "std") / sv, roll(va, "std") / sa
        mu_v, mu_a = roll(vm, "mean"), roll(va, "mean")
        d_busmis = diff(bus_min)
        for i, b in enumerate(self.grid.bus_numbers):
            add("temporal", f"dV_t_bus_{b}", dvm[:, i])
            add("temporal", f"dAng_t_bus_{b}", dva[:, i])
            add("temporal", f"rollStdV_bus_{b}", np.nan_to_num(sd_v[:, i]))
            add("temporal", f"rollStdAng_bus_{b}", np.nan_to_num(sd_a[:, i]))
            add("temporal", f"devRollV_bus_{b}", (vm[:, i] - mu_v[:, i]) / sv)
            add("temporal", f"devRollAng_bus_{b}", (va[:, i] - mu_a[:, i]) / sa)
            add("temporal", f"dBusMisMin_t_bus_{b}", d_busmis[:, i])
        dpf = diff(line["P_from"]) / self._sig(line["P_from"])
        dld = diff(line["loading"])
        for k in range(nl):
            add("temporal", f"dPfrom_t_line_{k+1}", dpf[:, k])
            add("temporal", f"dLoading_t_line_{k+1}", dld[:, k])
        add("temporal", "dV_t_abs_max", np.abs(dvm).max(1))
        add("temporal", "dAng_t_abs_max", np.abs(dva).max(1))
        add("temporal", "dMisZ_mean_t", diff(line_norm.mean(1, keepdims=True))[:, 0])

        X = pd.DataFrame(feats, index=df.index)
        if self.physics is not None:
            rows = np.array([len(df) - 1]) if physics_last_only else None
            P = self.physics.build(df, rows)
            X = pd.concat([X, P], axis=1)
            groups["physics"] = list(P.columns)
        S.assert_no_leakage(list(X.columns))
        return FeatureSet(X, groups, self.bus_local_features(groups))

    # ------------------------------------------------------------- locality
    def bus_local_features(self, groups: dict[str, list[str]]) -> dict[int, list[str]]:
        """Features describing bus b, its neighbours, incident branches and the system."""
        names = set(c for g in groups.values() for c in g)
        global_feats = [c for c in names if "_bus_" not in c and "_line_" not in c]
        out = {}
        for b in self.grid.bus_numbers:
            cols = []
            buses = [b] + self.grid.adjacency[b]
            for bb in buses:
                suffix = f"_bus_{bb}"
                cols += [c for c in names if c.endswith(suffix) and
                         (bb == b or c.startswith(("V_meas", "Ang_meas", "busMisMin", "busMisMax",
                                                   "rnV", "rnAng", "rnFlowMax")))]
            for k in self.grid.incident_branches[b]:
                suffix = f"_line_{k}"
                cols += [c for c in names if c.endswith(suffix)]
            out[b] = sorted(set(cols)) + sorted(global_feats)
        return out


def feature_columns(groups: dict[str, list[str]], selection: tuple[str, ...]) -> list[str]:
    cols = []
    for g in selection:
        cols += groups[g]
    return cols
