"""Attack 2 - scaling FDIA:  z_a = lambda * z with a random factor per bus and
channel (lambda = 1 +/- dev).  Scaling the reference-bus angle (0 deg) would
leave it unchanged, so the angle channel is never scaled at the reference bus."""
from __future__ import annotations

import numpy as np

from attacks.base import AttackEvent, choose_channels, rand_signed


def create(rng: np.random.Generator, buses: list[int], channels: list[str], cfg: dict,
           intensity: float, ref_bus: int) -> tuple[dict, dict]:
    bus_channels, factors = {}, {}
    for b in buses:
        allowed = [c for c in channels if not (c == "Ang" and b == ref_bus)] or ["V"]
        chs = choose_channels(rng, allowed)
        bus_channels[b] = chs
        factors[b] = {}
        for ch in chs:
            if ch == "V":
                dev = rand_signed(rng, *cfg["voltage_factor_dev"])
            elif ch == "Ang":
                dev = rand_signed(rng, *cfg["angle_factor_dev"])
            else:
                dev = rand_signed(rng, 0.05, 0.25)
            factors[b][ch] = 1.0 + dev * intensity
    return bus_channels, {"factors": factors}


def apply(event: AttackEvent, snap, bus_index: dict[int, int], **_) -> None:
    for b, chs in event.bus_channels.items():
        i = bus_index[b]
        for ch in chs:
            snap.meas_bus.channel(ch)[i] *= event.params["factors"][b][ch]
