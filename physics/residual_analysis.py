"""Bad-data analysis on WLS residuals.

* Chi-square test: J(x_hat) ~ chi2(m - n) for Gaussian noise; J above the
  (1 - alpha) quantile indicates bad data.
* Largest normalised residual (LNR) test: |r_i| / sqrt(Omega_ii) > threshold
  (default 3) flags measurement i; the LNR identification loop removes the
  worst measurement and re-estimates until no residual exceeds the threshold.

Limitation (documented): residual-based detection only sees perturbations
that are inconsistent with the measurement model.  An attacker who knows
H and injects a = H c (a "stealthy" FDIA, Liu et al. 2009) leaves the
residuals unchanged and is NOT detected by these tests.  Critical
measurements (Omega_ii ~ 0) have no normalised residual at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from physics.state_estimation import WLSStateEstimator, MeasurementSet, EstimationResult, EstimationError


@dataclass
class BadDataReport:
    chi2_statistic: float
    chi2_threshold: float
    chi2_alarm: bool
    max_normalized_residual: float
    lnr_alarm: bool
    suspicious: list[dict]          # measurements with r_N above the threshold (current estimate)
    bus_scores: dict[int, float]    # max r_N of the measurements installed at each bus


def analyze(result: EstimationResult, alpha: float = 0.01, rn_threshold: float = 3.0) -> BadDataReport:
    thr = float(chi2.ppf(1 - alpha, max(result.dof, 1)))
    rn = result.normalized_residuals
    ms = result.measurements
    finite = np.where(np.isfinite(rn), rn, 0.0)
    order = np.argsort(-finite)
    suspicious = [{"measurement": ms.names[i], "kind": ms.kinds[i], "bus": int(ms.bus[i]),
                   "normalized_residual": float(rn[i]), "residual_pu": float(result.residuals[i])}
                  for i in order if finite[i] > rn_threshold]
    scores: dict[int, float] = {}
    for i, b in enumerate(ms.bus):
        scores[int(b)] = max(scores.get(int(b), 0.0), float(finite[i]))
    max_rn = float(finite.max()) if len(finite) else 0.0
    return BadDataReport(result.objective, thr, bool(result.objective > thr), max_rn,
                         bool(max_rn > rn_threshold), suspicious, scores)


def identify_bad_data(se: WLSStateEstimator, ms: MeasurementSet, rn_threshold: float = 3.0,
                      max_removals: int = 12, alpha: float = 0.01) -> dict:
    """LNR identification: iteratively remove the largest normalised residual.

    Returns the removed measurements (in removal order), the final estimate on
    the cleaned set and whether the cleaned set still passes the chi-square test.
    """
    keep = np.ones(len(ms), bool)
    removed = []
    result = se.estimate(ms)
    for _ in range(max_removals):
        rn = np.where(np.isfinite(result.normalized_residuals), result.normalized_residuals, 0.0)
        if rn.max() <= rn_threshold:
            break
        worst_local = int(np.argmax(rn))
        worst_global = np.flatnonzero(keep)[worst_local]
        trial = keep.copy()
        trial[worst_global] = False
        try:
            new_result = se.estimate(ms.subset(trial), result.vm, result.va_deg)
        except EstimationError:
            break       # removing it would make the system unobservable
        if not new_result.observable:
            break
        removed.append({"measurement": ms.names[worst_global], "kind": ms.kinds[worst_global],
                        "bus": int(ms.bus[worst_global]), "normalized_residual": float(rn[worst_local])})
        keep, result = trial, new_result
    final = analyze(result, alpha, rn_threshold)
    return {"removed": removed, "keep_mask": keep, "result": result, "final_chi2_ok": not final.chi2_alarm}
