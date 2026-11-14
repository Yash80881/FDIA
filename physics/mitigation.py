"""Operator-supervised mitigation and recovery recommendations.

Nothing here acts on equipment.  The module produces *recommendations* for an
operator, based on a simulated recovery:

1. collect suspicious measurements: V/angle sensors of buses localised by the
   Random Forest plus measurements identified by the LNR bad-data test;
2. exclude them from the trusted measurement set (only if the remaining set is
   still observable);
3. re-run WLS state estimation on the trusted set -> recovered state;
4. recompute branch flows / loading from the recovered state (power-flow
   equations with the PYPOWER network model) and reconstruct the excluded
   measurements as h(x_recovered);
5. compare received (possibly corrupted) vs recovered values, and - for
   offline evaluation only - vs simulator ground truth (clearly labelled);
6. recommend actions: isolate sensors, use reconstructed values, and DC-PTDF
   based corrective re-dispatch if a recovered line loading exceeds its limit.
"""
from __future__ import annotations

import numpy as np
from pypower.makePTDF import makePTDF
from pypower.ext2int import ext2int

from physics.state_estimation import WLSStateEstimator, EstimationError
from physics.residual_analysis import analyze, identify_bad_data
from physics.verification import measurement_arrays
from simulator import schema as S


class MitigationAdvisor:
    def __init__(self, se: WLSStateEstimator, cfg: dict):
        self.se = se
        self.cfg = cfg
        self.grid = se.grid
        ppc = ext2int(self.grid.new_case())
        self.ptdf = makePTDF(ppc["baseMVA"], ppc["bus"], ppc["branch"])   # (n_branch, n_bus), slack = ref

    def recommend(self, row: dict, predicted_buses: list[int], include_truth: bool = True) -> dict:
        pc = self.cfg["physics"]
        m = measurement_arrays(row)
        ms = self.se.build_measurements(m["vm"], m["va"], m["p"], m["q"], m["p_from"], m["q_from"], m["p_to"],
                                        m["q_to"], self.cfg["noise"], use_angles=pc["use_pmu_angles"])
        try:
            before = self.se.estimate(ms, m["vm"], m["va"])
        except EstimationError as exc:
            return {"status": "error", "message": f"State estimation failed: {exc}"}

        # 1) ML-suspected sensors: V and angle sensors of predicted buses
        excl = np.zeros(len(ms), bool)
        pred = set(int(b) for b in predicted_buses)
        for i, (kind, bus) in enumerate(zip(ms.kinds, ms.bus)):
            if kind in ("V", "Ang") and bus in pred:
                excl[i] = True
        trusted = ms.subset(~excl)
        source = "ML localisation"
        try:
            after_ml = self.se.estimate(trusted, before.vm, before.va_deg)
            if not after_ml.observable:
                raise EstimationError("trusted set unobservable")
        except EstimationError:
            excl[:] = False
            trusted, after_ml = ms, before
            source = "none (exclusion would make the system unobservable)"
        # 2) physics LNR identification on the remaining set
        ident = identify_bad_data(self.se, trusted, pc["normalized_residual_threshold"], pc["max_lnr_removals"],
                                  pc["chi2_alpha"])
        recovered = ident["result"]
        excluded_names = [n for n, e in zip(ms.names, excl) if e] + [r["measurement"] for r in ident["removed"]]
        if ident["removed"]:
            source += " + physics LNR identification" if source == "ML localisation" else ""

        flows_rec = self.se.flows_from_state(recovered.vm, recovered.va_deg)
        rate = self.grid.rate_a
        with np.errstate(invalid="ignore", divide="ignore"):
            load_rec = np.maximum(np.hypot(flows_rec["p_from"], flows_rec["q_from"]),
                                  np.hypot(flows_rec["p_to"], flows_rec["q_to"])) / rate * 100
            load_meas = np.array([row.get(c) if row.get(c) is not None else np.nan
                                  for c in S.line_cols("loading")], float)

        # reconstruction of excluded V / angle sensors
        reconstructed = []
        for name in excluded_names:
            kind, num = name.rsplit("_", 1)
            j = int(num) - 1
            if kind == "V_bus":
                reconstructed.append({"measurement": f"V_meas_bus_{j+1}", "received": float(m["vm"][j]),
                                      "reconstructed": float(recovered.vm[j]), "unit": "pu"})
            elif kind == "Ang_bus":
                reconstructed.append({"measurement": f"Ang_meas_bus_{j+1}", "received": float(m["va"][j]),
                                      "reconstructed": float(recovered.va_deg[j]), "unit": "deg"})
            elif kind in ("P_bus", "Q_bus"):
                key = "p_inj" if kind == "P_bus" else "q_inj"
                reconstructed.append({"measurement": f"{kind[0]}_meas_bus_{j+1}",
                                      "received": float(m["p" if kind == "P_bus" else "q"][j]),
                                      "reconstructed": float(flows_rec[key][j]), "unit": "MW" if kind == "P_bus" else "MVAr"})

        rep_before = analyze(before, pc["chi2_alpha"], pc["normalized_residual_threshold"])
        rep_after = analyze(recovered, pc["chi2_alpha"], pc["normalized_residual_threshold"])
        out = {
            "status": "ok",
            "operator_supervised": True,
            "note": "Recommendations only - no equipment is switched automatically.",
            "exclusion_source": source,
            "excluded_measurements": excluded_names,
            "reconstructed_measurements": reconstructed,
            "before": {"chi2_statistic": rep_before.chi2_statistic, "chi2_alarm": rep_before.chi2_alarm,
                       "max_normalized_residual": rep_before.max_normalized_residual,
                       "vm": before.vm.tolist(), "va_deg": before.va_deg.tolist()},
            "after": {"chi2_statistic": rep_after.chi2_statistic, "chi2_alarm": rep_after.chi2_alarm,
                      "max_normalized_residual": rep_after.max_normalized_residual,
                      "chi2_threshold": rep_after.chi2_threshold, "observable": recovered.observable,
                      "vm": recovered.vm.tolist(), "va_deg": recovered.va_deg.tolist()},
            "received_vm": m["vm"].tolist(), "received_va_deg": m["va"].tolist(),
            "line_loading_received": [None if not np.isfinite(x) else float(x) for x in load_meas],
            "line_loading_recovered": [None if not np.isfinite(x) else float(x) for x in load_rec],
        }
        if include_truth and row.get("V_true_bus_1") is not None:
            vt = np.array([row[c] for c in S.bus_cols("V_true")])
            at = np.array([row[c] for c in S.bus_cols("Ang_true")])
            out["evaluation_against_ground_truth"] = {
                "label": "SIMULATOR GROUND TRUTH (offline evaluation only - not available to a real operator)",
                "received_v_rmse": float(np.sqrt(np.mean((m["vm"] - vt) ** 2))),
                "recovered_v_rmse": float(np.sqrt(np.mean((recovered.vm - vt) ** 2))),
                "received_angle_rmse_deg": float(np.sqrt(np.mean((m["va"] - at) ** 2))),
                "recovered_angle_rmse_deg": float(np.sqrt(np.mean((recovered.va_deg - at) ** 2))),
                "true_vm": vt.tolist(), "true_va_deg": at.tolist(),
            }
        out["_p_from_rec"] = flows_rec["p_from"]
        out["recommendations"] = self._actions(out, recovered, load_rec, pred)
        out.pop("_p_from_rec")
        out["recovered_p_from_mw"] = flows_rec["p_from"].tolist()
        return out

    def _actions(self, out, recovered, load_rec, pred) -> list[dict]:
        acts = []
        bad_buses = sorted(pred | {int(n.rsplit("_", 1)[1]) for n in out["excluded_measurements"]
                                   if n.startswith(("V_bus", "Ang_bus", "P_bus", "Q_bus"))})
        if bad_buses:
            acts.append({"priority": "high", "action": "isolate-sensors",
                         "text": f"Quarantine the V/angle sensor channels at bus(es) {bad_buses}; mark their "
                                 "telemetry as untrusted and dispatch a field/cyber check of the RTU/PMU links."})
        if out["reconstructed_measurements"]:
            acts.append({"priority": "high", "action": "use-reconstructed",
                         "text": "Use the state-estimator reconstructed values for the excluded sensors in EMS "
                                 "applications until the sensors are verified."})
        if out["after"]["chi2_alarm"]:
            acts.append({"priority": "medium", "action": "residual-alarm-persists",
                         "text": "Residuals remain inconsistent after exclusion: widen the investigation to "
                                 "neighbouring substations and line-flow sensors."})
        vmin, vmax = 0.95, 1.05
        lims = self.grid.voltage_limits
        for i, b in enumerate(self.grid.bus_numbers):
            lo, hi = lims[b]
            if recovered.vm[i] < lo or recovered.vm[i] > hi:
                acts.append({"priority": "medium", "action": "voltage-support",
                             "text": f"Recovered voltage at bus {b} is {recovered.vm[i]:.3f} pu (limits {lo}-{hi}); "
                                     "adjust reactive support of the nearest PV generator / shunt."})
        for k, ld in enumerate(load_rec):
            if np.isfinite(ld) and ld > 100.0:
                acts.append(self._redispatch(k, ld, float(np.sign(out['_p_from_rec'][k]) or 1.0)))
            elif np.isfinite(ld) and ld > 90.0:
                acts.append({"priority": "low", "action": "monitor-line",
                             "text": f"Line {k+1} recovered loading {ld:.1f}% (>90%); monitor closely."})
        if not acts:
            acts.append({"priority": "info", "action": "none",
                         "text": "No inconsistency requiring action; continue normal monitoring."})
        return acts

    def _redispatch(self, k: int, loading: float, flow_sign: float = 1.0) -> dict:
        """DC-PTDF corrective re-dispatch suggestion for an overloaded branch k."""
        br = self.grid.branches[k]
        excess = (loading - 95.0) / 100.0 * br.rate_a_mva
        best = None
        for g in self.grid.generators:
            if g["bus"] == self.grid.ref_bus:
                continue
            # MW change of the loaded flow per MW injected at the generator (withdrawn at the slack)
            sens = flow_sign * self.ptdf[k, self.grid.bus_index[g["bus"]]]
            if abs(sens) < 0.05:
                continue
            if best is None or abs(sens) > abs(best[1]):
                best = (g, sens)
        if best is None:
            return {"priority": "high", "action": "redispatch",
                    "text": f"Line {k+1} ({br.from_bus}-{br.to_bus}) at {loading:.1f}%: no effective generator "
                            "found by PTDF; consider load transfer."}
        g, sens = best
        direction = "decrease" if sens > 0 else "increase"
        mw = min(excess / abs(sens), g["pmax_mw"] - g["pmin_mw"])
        return {"priority": "high", "action": "redispatch",
                "text": f"Line {k+1} ({br.from_bus}-{br.to_bus}) at {loading:.1f}%: {direction} generator at bus "
                        f"{g['bus']} by about {mw:.1f} MW (PTDF {sens:+.2f}), balanced by the reference generator, "
                        "subject to operator approval and generator limits."}
