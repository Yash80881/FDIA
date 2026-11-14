"""Common data structures for the FDIA attack engine.

Attacks act exclusively on the *received measurement layer*
(``Snapshot.meas_bus`` / ``Snapshot.meas_line``).  The power-flow case, the
true measurements and the clean received reference are never modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BUS_CHANNELS = ("V", "Ang", "P", "Q")


@dataclass
class AttackEvent:
    attack_id: str
    attack_type: str
    start_step: int
    duration_steps: int
    bus_channels: dict[int, list[str]]          # external bus number -> attacked channels
    params: dict = field(default_factory=dict)  # per-type parameters (offsets, factors, lag)
    line_offsets: dict[int, float] = field(default_factory=dict)  # line-sensor mode only

    @property
    def end_step(self) -> int:
        """Last step (inclusive) at which the attack is applied."""
        return self.start_step + self.duration_steps - 1

    @property
    def buses(self) -> list[int]:
        return sorted(self.bus_channels)

    @property
    def channels(self) -> list[str]:
        return sorted({c for chs in self.bus_channels.values() for c in chs})

    def is_active(self, step: int) -> bool:
        return self.start_step <= step <= self.end_step

    def to_record(self) -> dict:
        def _clean(o):
            if isinstance(o, dict):
                return {str(k): _clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_clean(v) for v in o]
            if isinstance(o, np.generic):
                return o.item()
            return o
        return {
            "attack_id": self.attack_id, "attack_type": self.attack_type,
            "start_step": self.start_step, "end_step": self.end_step,
            "duration_steps": self.duration_steps, "buses": self.buses,
            "channels": self.channels, "bus_channels": _clean(self.bus_channels),
            "params": _clean(self.params), "line_offsets": _clean(self.line_offsets),
        }


def rand_signed(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Magnitude uniform in [lo, hi] with a random sign."""
    return float(rng.uniform(lo, hi) * rng.choice([-1.0, 1.0]))


def choose_channels(rng: np.random.Generator, allowed: list[str]) -> list[str]:
    """Random non-empty subset of the allowed channels (V/Ang weighted)."""
    allowed = list(allowed)
    if len(allowed) == 1:
        return allowed
    k = int(rng.integers(1, len(allowed) + 1))
    return sorted(rng.choice(allowed, size=k, replace=False).tolist())
