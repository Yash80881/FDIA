"""Extraction of true measurements from a solved power flow and generation of
noisy received sensor measurements.

    z_true      : quantities computed from the converged PYPOWER solution
    z_received  = z_true + eps,  eps ~ N(0, sigma^2)   (independent per sensor)

The attack engine later manipulates a *copy* of the received measurements;
``true`` arrays are never modified after extraction.

Branch currents are computed from the solved terminal flows:
    I_from [A] = |S_from| [MVA] / (sqrt(3) * |V_from| [pu] * V_base,kV) * 1000
Line loading uses the authentic RATE_A (MVA) of the case:
    loading [%] = max(|S_from|, |S_to|) / RATE_A * 100   (NaN when no rating)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from simulator.grid_model import GridModel
from simulator.power_flow import PowerFlowResult

SQRT3 = np.sqrt(3.0)


@dataclass
class BusMeasurements:
    vm: np.ndarray    # pu
    va: np.ndarray    # deg
    p: np.ndarray     # MW, net injection
    q: np.ndarray     # MVAr, net injection

    def copy(self) -> "BusMeasurements":
        return BusMeasurements(self.vm.copy(), self.va.copy(), self.p.copy(), self.q.copy())

    def channel(self, name: str) -> np.ndarray:
        return {"V": self.vm, "Ang": self.va, "P": self.p, "Q": self.q}[name]


@dataclass
class LineMeasurements:
    p_from: np.ndarray
    q_from: np.ndarray
    p_to: np.ndarray
    q_to: np.ndarray
    i_from: np.ndarray   # A
    i_to: np.ndarray     # A
    loading: np.ndarray  # %, NaN when no rating

    def copy(self) -> "LineMeasurements":
        return LineMeasurements(*(getattr(self, f).copy() for f in
                                  ("p_from", "q_from", "p_to", "q_to", "i_from", "i_to", "loading")))

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"P_from": self.p_from, "Q_from": self.q_from, "P_to": self.p_to, "Q_to": self.q_to,
                "I_from": self.i_from, "I_to": self.i_to, "loading": self.loading}


@dataclass
class Snapshot:
    """One simulation step: true state, received measurements and context."""
    step: int
    true_bus: BusMeasurements
    true_line: LineMeasurements
    meas_bus: BusMeasurements          # received (noisy, possibly attacked)
    meas_line: LineMeasurements
    clean_meas_bus: BusMeasurements    # noisy but never attacked (reference)
    clean_meas_line: LineMeasurements
    pg_mw: np.ndarray
    qg_mvar: np.ndarray
    pd_mw: np.ndarray
    qd_mvar: np.ndarray
    pf_iterations: int
    extra: dict = field(default_factory=dict)


def line_loading(p_from, q_from, p_to, q_to, rate_a) -> np.ndarray:
    s = np.maximum(np.hypot(p_from, q_from), np.hypot(p_to, q_to))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(rate_a) & (rate_a > 0), s / rate_a * 100.0, np.nan)


class MeasurementGenerator:
    def __init__(self, grid: GridModel, cfg: dict, rng: np.random.Generator):
        self.grid = grid
        self.noise = cfg["noise"]
        self.rng = rng
        self.kv_from = grid.base_kv[grid.branch_from_idx]
        self.kv_to = grid.base_kv[grid.branch_to_idx]

    # ------------------------------------------------------------ true values
    def true_measurements(self, pf: PowerFlowResult) -> tuple[BusMeasurements, LineMeasurements]:
        g = self.grid
        bus = BusMeasurements(pf.vm_pu.copy(), pf.va_deg.copy(), pf.p_inj_mw.copy(), pf.q_inj_mvar.copy())
        vf = pf.vm_pu[g.branch_from_idx]
        vt = pf.vm_pu[g.branch_to_idx]
        sf = np.hypot(pf.pf_mw, pf.qf_mvar)
        st = np.hypot(pf.pt_mw, pf.qt_mvar)
        i_from = sf / (SQRT3 * vf * self.kv_from) * 1000.0
        i_to = st / (SQRT3 * vt * self.kv_to) * 1000.0
        line = LineMeasurements(pf.pf_mw.copy(), pf.qf_mvar.copy(), pf.pt_mw.copy(), pf.qt_mvar.copy(),
                                i_from, i_to,
                                line_loading(pf.pf_mw, pf.qf_mvar, pf.pt_mw, pf.qt_mvar, g.rate_a))
        return bus, line

    # ------------------------------------------------------------ noise model
    def power_sigma(self, values: np.ndarray) -> np.ndarray:
        n = self.noise
        return n["power_relative"] * np.abs(values) + n["power_floor_mw"]

    def current_sigma(self, values: np.ndarray) -> np.ndarray:
        n = self.noise
        return n["current_relative"] * np.abs(values) + n["current_floor_a"]

    def add_noise(self, bus: BusMeasurements, line: LineMeasurements
                  ) -> tuple[BusMeasurements, LineMeasurements]:
        n, r = self.noise, self.rng
        nb, nl = len(bus.vm), len(line.p_from)
        mb = BusMeasurements(
            vm=bus.vm + r.normal(0, n["voltage_pu"], nb),
            va=bus.va + r.normal(0, n["angle_deg"], nb),
            p=bus.p + r.normal(0, 1, nb) * self.power_sigma(bus.p),
            q=bus.q + r.normal(0, 1, nb) * self.power_sigma(bus.q),
        )
        pf_ = line.p_from + r.normal(0, 1, nl) * self.power_sigma(line.p_from)
        qf_ = line.q_from + r.normal(0, 1, nl) * self.power_sigma(line.q_from)
        pt_ = line.p_to + r.normal(0, 1, nl) * self.power_sigma(line.p_to)
        qt_ = line.q_to + r.normal(0, 1, nl) * self.power_sigma(line.q_to)
        if_ = np.abs(line.i_from + r.normal(0, 1, nl) * self.current_sigma(line.i_from))
        it_ = np.abs(line.i_to + r.normal(0, 1, nl) * self.current_sigma(line.i_to))
        ml = LineMeasurements(pf_, qf_, pt_, qt_, if_, it_,
                              line_loading(pf_, qf_, pt_, qt_, self.grid.rate_a))
        return mb, ml

    def snapshot(self, step: int, pf: PowerFlowResult) -> Snapshot:
        true_bus, true_line = self.true_measurements(pf)
        meas_bus, meas_line = self.add_noise(true_bus, true_line)
        return Snapshot(step=step, true_bus=true_bus, true_line=true_line,
                        meas_bus=meas_bus, meas_line=meas_line,
                        clean_meas_bus=meas_bus.copy(), clean_meas_line=meas_line.copy(),
                        pg_mw=pf.pg_mw.copy(), qg_mvar=pf.qg_mvar.copy(),
                        pd_mw=pf.pd_mw.copy(), qd_mvar=pf.qd_mvar.copy(),
                        pf_iterations=pf.iterations)

    def channel_sigma(self, channel: str, clean_values: np.ndarray) -> np.ndarray:
        """Noise standard deviation of a bus channel (used for severity)."""
        if channel == "V":
            return np.full(len(clean_values), self.noise["voltage_pu"])
        if channel == "Ang":
            return np.full(len(clean_values), self.noise["angle_deg"])
        return self.power_sigma(clean_values)
