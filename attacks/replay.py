"""Attack 3 - replay attack: the received measurements of the compromised
sensors are replaced by legitimate measurements the attacker recorded
``lag`` steps earlier (the recording is played back step by step).

The scheduler only starts a replay when the history buffer holds at least
``min_history_steps`` legitimate snapshots, so indexing never fails; if the
required record is missing at apply time a :class:`ReplayHistoryError` is
raised instead of silently producing a wrong label.
"""
from __future__ import annotations

import numpy as np

from attacks.base import AttackEvent, choose_channels


class ReplayHistoryError(RuntimeError):
    pass


def create(rng: np.random.Generator, buses: list[int], channels: list[str], cfg: dict,
           history_len: int) -> tuple[dict, dict]:
    lo, hi = cfg["lag_steps"]
    hi = min(hi, history_len)
    if hi < lo:
        raise ReplayHistoryError(f"Replay needs >= {lo} steps of history, have {history_len}")
    lag = int(rng.integers(lo, hi + 1))
    chs = choose_channels(rng, channels)   # a replay attacker records the same channel set
    return {b: list(chs) for b in buses}, {"lag_steps": lag}


def apply(event: AttackEvent, snap, bus_index: dict[int, int], history: dict, **_) -> None:
    """``history`` maps step -> clean received BusMeasurements."""
    src_step = snap.step - event.params["lag_steps"]
    record = history.get(src_step)
    if record is None:
        raise ReplayHistoryError(f"No recorded measurements for step {src_step}")
    for b, chs in event.bus_channels.items():
        i = bus_index[b]
        for ch in chs:
            snap.meas_bus.channel(ch)[i] = record.channel(ch)[i]
