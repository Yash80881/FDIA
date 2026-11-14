"""Attack 1 - additive FDIA:  z_a = z + a, with an independent random offset
per compromised bus and channel (random sign and magnitude)."""
from __future__ import annotations

import numpy as np

from attacks.base import AttackEvent, choose_channels, rand_signed


def create(rng: np.random.Generator, buses: list[int], channels: list[str], cfg: dict,
           intensity: float) -> tuple[dict, dict]:
    bus_channels, offsets = {}, {}
    for b in buses:
        chs = choose_channels(rng, channels)
        bus_channels[b] = chs
        offsets[b] = {}
        for ch in chs:
            offsets[b][ch] = _offset(rng, ch, cfg) * intensity
    return bus_channels, {"offsets": offsets}


def _offset(rng, ch, cfg) -> float:
    if ch == "V":
        return rand_signed(rng, *cfg["voltage_offset_pu"])
    if ch == "Ang":
        return rand_signed(rng, *cfg["angle_offset_deg"])
    # P/Q (only when power channels are enabled): 3-15 MW/MVAr
    return rand_signed(rng, 3.0, 15.0)


def apply(event: AttackEvent, snap, bus_index: dict[int, int], **_) -> None:
    for b, chs in event.bus_channels.items():
        i = bus_index[b]
        for ch in chs:
            snap.meas_bus.channel(ch)[i] += event.params["offsets"][b][ch]
