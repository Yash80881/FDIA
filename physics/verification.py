"""Physics-based verification of a measurement snapshot and combination with
the Random Forest decision.

``verify_snapshot`` runs WLS state estimation on the received measurements,
applies the chi-square and normalised-residual tests and runs LNR
identification to name suspicious sensors and buses.

``combine`` fuses the ML detector and the physics test:
    ML only / physics only / both / neither  ->  agreement + combined decision.
Combined decision rule (documented, configurable by code):
    ATTACK if ML probability >= threshold OR (physics alarm AND ML probability >= 0.5*threshold)
    i.e. physics alone needs weak ML support, which suppresses alarms caused
    purely by ordinary modelling noise while still exposing ML misses with a
    clear physical inconsistency (reported separately as 'physics-only').
"""
from __future__ import annotations

import time

import numpy as np

from physics.state_estimation import WLSStateEstimator, EstimationError
from physics.residual_analysis import analyze, identify_bad_data
from simulator import schema as S


def measurement_arrays(row: dict) -> dict:
    """Received measurement arrays from a stored snapshot row."""
    def a(cols):
        return np.array([row[c] if row[c] is not None else np.nan for c in cols], float)
    return {
        "vm": a(S.bus_cols("V_meas")), "va": a(S.bus_cols("Ang_meas")),
        "p": a(S.bus_cols("P_meas")), "q": a(S.bus_cols("Q_meas")),
        "p_from": a(S.line_cols("P_from")), "q_from": a(S.line_cols("Q_from")),
        "p_to": a(S.line_cols("P_to")), "q_to": a(S.line_cols("Q_to")),
    }


def verify_snapshot(se: WLSStateEstimator, row: dict, cfg: dict, identify: bool = True) -> dict:
    pc = cfg["physics"]
    m = measurement_arrays(row)
    t0 = time.perf_counter()
    ms = se.build_measurements(m["vm"], m["va"], m["p"], m["q"], m["p_from"], m["q_from"], m["p_to"],
                               m["q_to"], cfg["noise"], use_angles=pc["use_pmu_angles"])
    try:
        res = se.estimate(ms, m["vm"], m["va"])
    except EstimationError as exc:
        return {"status": "error", "message": str(exc)}
    rep = analyze(res, pc["chi2_alpha"], pc["normalized_residual_threshold"])
    out = {
        "status": "ok",
        "converged": res.converged, "iterations": res.iterations, "observable": res.observable,
        "n_measurements": len(ms), "dof": res.dof,
        "chi2_statistic": rep.chi2_statistic, "chi2_threshold": rep.chi2_threshold,
        "chi2_alarm": rep.chi2_alarm,
        "max_normalized_residual": rep.max_normalized_residual, "lnr_alarm": rep.lnr_alarm,
        "bad_data_detected": bool(rep.chi2_alarm or rep.lnr_alarm),
        "estimated_vm": res.vm.tolist(), "estimated_va_deg": res.va_deg.tolist(),
        "top_residuals": rep.suspicious[:15],
        "bus_residual_scores": {str(b): s for b, s in rep.bus_scores.items()},
    }
    if identify and out["bad_data_detected"]:
        ident = identify_bad_data(se, ms, pc["normalized_residual_threshold"], pc["max_lnr_removals"],
                                  pc["chi2_alpha"])
        out["identified_bad_measurements"] = ident["removed"]
        out["suspicious_buses"] = sorted({r["bus"] for r in ident["removed"] if r["kind"] in ("V", "Ang", "P", "Q")}
                                         | {r["bus"] for r in ident["removed"]})
        out["cleaned_chi2_ok"] = ident["final_chi2_ok"]
        clean = ident["result"]
        out["cleaned_estimated_vm"] = clean.vm.tolist()
        out["cleaned_estimated_va_deg"] = clean.va_deg.tolist()
    else:
        out["identified_bad_measurements"] = []
        out["suspicious_buses"] = []
    out["latency_ms"] = (time.perf_counter() - t0) * 1000
    return out


def combine(ml_probability: float | None, ml_threshold: float, physics: dict | None) -> dict:
    ml_alarm = ml_probability is not None and ml_probability >= ml_threshold
    phys_ok = physics is not None and physics.get("status") == "ok"
    phys_alarm = bool(phys_ok and physics["bad_data_detected"])
    if ml_probability is None and not phys_ok:
        return {"decision": "UNKNOWN", "agreement": "unavailable", "ml_alarm": None, "physics_alarm": None}
    weak_ml = ml_probability is not None and ml_probability >= 0.5 * ml_threshold
    decision = ml_alarm or (phys_alarm and (weak_ml or ml_probability is None))
    if ml_alarm and phys_alarm:
        agreement = "both detect"
    elif ml_alarm:
        agreement = "ML only"
    elif phys_alarm:
        agreement = "physics only"
    else:
        agreement = "both normal"
    return {"decision": "ATTACK" if decision else "NORMAL", "agreement": agreement,
            "ml_alarm": bool(ml_alarm), "physics_alarm": phys_alarm}
