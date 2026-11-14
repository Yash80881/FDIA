"""Time-varying demand models for the continuous simulator.

The demand of load bus *i* at step *t* is

    P_i(t) = P_i,base * m_i(t),      Q_i(t) = Q_i,base * m_i(t) * (1 + e_q,i(t))

with the multiplier

    m_i(t) = S(t) * D_i(t) * C(t) * E_i(t)

* S(t)  - operating scenario level (light / normal / heavy), Markov switching
* D_i(t) - periodic (compressed daily) profile; each bus is assigned a
           residential, commercial or industrial profile with its own peak phase
* C(t)  - system-wide correlated AR(1) factor (weather-like common variation)
* E_i(t) - independent per-bus AR(1) factor
* e_q,i - small independent reactive-power jitter (power-factor variation)

All randomness comes from a seeded ``numpy.random.Generator``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator.grid_model import GridModel


@dataclass
class LoadState:
    step: int
    scenario: str
    scenario_level: float
    periodic_mean: float
    correlated_factor: float
    multipliers: np.ndarray
    pd_mw: np.ndarray
    qd_mvar: np.ndarray

    @property
    def total_p_mw(self) -> float:
        return float(self.pd_mw.sum())

    @property
    def total_q_mvar(self) -> float:
        return float(self.qd_mvar.sum())


class LoadProfileModel:
    def __init__(self, grid: GridModel, cfg: dict, rng: np.random.Generator):
        self.grid = grid
        self.cfg = cfg["load_profile"]
        self.rng = rng
        n = grid.n_bus
        self.base_pd = grid.base_pd
        self.base_qd = grid.base_qd

        # Assign a demand profile to each bus (seeded, weighted).
        profiles = self.cfg["profiles"]
        names = list(profiles)
        weights = np.array([profiles[p]["weight"] for p in names], dtype=float)
        weights /= weights.sum()
        self.bus_profile = [names[i] for i in rng.choice(len(names), size=n, p=weights)]
        self.peak_phase = np.array([profiles[p]["peak_phase"] for p in self.bus_profile])
        # Each bus also gets a small individual phase shift.
        self.peak_phase = self.peak_phase + rng.normal(0, 0.02, n)

        self.correlated = 1.0
        self.independent = np.ones(n)
        sc = self.cfg["scenarios"]
        self.scenario_names = list(sc["levels"])
        self.scenario = "normal" if "normal" in sc["levels"] else self.scenario_names[0]
        # Random start phase so independent runs do not all start at "midnight".
        self.phase_offset = rng.uniform(0, 1)

    def _update_scenario(self) -> None:
        sc = self.cfg["scenarios"]
        if not sc["enabled"]:
            return
        if self.rng.random() < 1.0 / max(sc["mean_dwell_steps"], 1):
            choices = [s for s in self.scenario_names if s != self.scenario]
            self.scenario = str(self.rng.choice(choices))

    def step(self, t: int) -> LoadState:
        c = self.cfg
        n = self.grid.n_bus
        self._update_scenario()
        level = float(c["scenarios"]["levels"][self.scenario]) if c["scenarios"]["enabled"] else 1.0

        if c["periodic"]["enabled"]:
            phase = (t / c["periodic"]["period_steps"] + self.phase_offset) % 1.0
            # cosine peaking at each bus's peak phase
            periodic = 1.0 + c["periodic"]["amplitude"] * np.cos(2 * np.pi * (phase - self.peak_phase))
        else:
            periodic = np.ones(n)

        if c["correlated"]["enabled"]:
            a, s = c["correlated"]["ar_coefficient"], c["correlated"]["sigma"]
            self.correlated = 1.0 + a * (self.correlated - 1.0) + self.rng.normal(0, s)
        if c["independent"]["enabled"]:
            a, s = c["independent"]["ar_coefficient"], c["independent"]["sigma"]
            self.independent = 1.0 + a * (self.independent - 1.0) + self.rng.normal(0, s, n)

        m = level * periodic * self.correlated * self.independent
        m = np.clip(m, c["min_multiplier"], c["max_multiplier"])
        q_jitter = 1.0 + self.rng.normal(0, c["power_factor_jitter"], n)
        pd = self.base_pd * m
        qd = self.base_qd * m * q_jitter
        return LoadState(step=t, scenario=self.scenario, scenario_level=level,
                         periodic_mean=float(np.mean(periodic)),
                         correlated_factor=float(self.correlated),
                         multipliers=m, pd_mw=pd, qd_mvar=qd)
