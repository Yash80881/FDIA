"""Generator dispatch variation within PYPOWER generator limits.

When demand changes by dP, the non-slack (PV) generators pick up their share
according to participation factors (proportional to available headroom, or
equal), plus a small random re-dispatch.  Outputs are clipped to
[Pmin, Pmax].  The reference (slack) generator absorbs the remaining imbalance
and network losses; after the power flow we verify that the slack output is
also inside its limits (:func:`check_generator_limits`).
"""
from __future__ import annotations

import numpy as np

from simulator.grid_model import GridModel
from pypower import idx_gen as IG


class GeneratorDispatcher:
    def __init__(self, grid: GridModel, cfg: dict, rng: np.random.Generator):
        self.grid = grid
        self.cfg = cfg["generation"]
        self.rng = rng
        gen = grid.base_case["gen"]
        self.base_pg = gen[:, IG.PG].copy()
        self.pmin = gen[:, IG.PMIN].copy()
        self.pmax = gen[:, IG.PMAX].copy()
        self.is_slack = np.array([int(b) == grid.ref_bus for b in gen[:, IG.GEN_BUS]])
        self.base_load = float(grid.base_pd.sum())
        # Slack share of the base load is kept by the slack; others share the rest.
        self.slack_share = float(self.base_pg[self.is_slack].sum() / max(self.base_load, 1e-9))

    def participation(self) -> np.ndarray:
        pv = ~self.is_slack
        if self.cfg["participation"] == "equal":
            f = pv.astype(float)
        else:  # headroom: proportional to the full operating range
            f = np.where(pv, self.pmax - self.pmin, 0.0)
        return f / f.sum()

    def dispatch(self, total_load_mw: float) -> np.ndarray:
        """Return generator active-power set points (MW) for this load level."""
        if not self.cfg["dispatch_variation"]:
            return self.base_pg.copy()
        delta = total_load_mw - self.base_load
        pg = self.base_pg + (1.0 - self.slack_share) * delta * self.participation()
        sigma = self.cfg["random_redispatch_sigma"]
        if sigma > 0:
            noise = self.rng.normal(0, sigma, len(pg)) * self.pmax
            noise[self.is_slack] = 0.0
            noise -= np.where(~self.is_slack, noise.sum() / max((~self.is_slack).sum(), 1), 0.0)
            pg = pg + noise
        pg = np.clip(pg, self.pmin, self.pmax)
        pg[self.is_slack] = self.base_pg[self.is_slack]  # slack is solved by the power flow
        return pg


def check_generator_limits(pg_mw: np.ndarray, qg_mvar: np.ndarray, grid: GridModel,
                           slack_margin_mw: float = 0.0) -> list[str]:
    """Return human-readable limit violations of a solved operating point."""
    issues = []
    for k, g in enumerate(grid.generators):
        if pg_mw[k] < g["pmin_mw"] - slack_margin_mw - 1e-6 or pg_mw[k] > g["pmax_mw"] + slack_margin_mw + 1e-6:
            issues.append(f"Gen {k+1} (bus {g['bus']}) P={pg_mw[k]:.2f} MW outside "
                          f"[{g['pmin_mw']}, {g['pmax_mw']}]")
        if qg_mvar[k] < g["qmin_mvar"] - 1e-6 or qg_mvar[k] > g["qmax_mvar"] + 1e-6:
            issues.append(f"Gen {k+1} (bus {g['bus']}) Q={qg_mvar[k]:.2f} MVAr outside "
                          f"[{g['qmin_mvar']}, {g['qmax_mvar']}]")
    return issues
