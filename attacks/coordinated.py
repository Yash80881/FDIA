"""Attack 4 - coordinated FDIA: several electrically neighbouring buses are
manipulated simultaneously with *correlated* offsets.

A seed bus is drawn at random and the compromised set is its nearest
neighbourhood in the real network graph (breadth-first, random tie-break).
All buses share one sign and one base magnitude per channel; each bus's offset
is the base magnitude times ``hop_decay ** hop_distance`` times a small
individual jitter, giving a spatially smooth, correlated perturbation rather
than identical hard-coded values.
"""
from __future__ import annotations

import numpy as np

from attacks.base import AttackEvent, choose_channels, rand_signed


def select_cluster(rng: np.random.Generator, grid, k: int) -> tuple[int, dict[int, int]]:
    seed = int(rng.choice(grid.bus_numbers))
    dist = grid.hop_distances(seed)
    order = sorted(dist, key=lambda b: (dist[b], rng.random()))
    cluster = order[:k]
    return seed, {b: dist[b] for b in cluster}


def create(rng: np.random.Generator, grid, k: int, channels: list[str], cfg: dict,
           intensity: float) -> tuple[dict, dict]:
    seed, hops = select_cluster(rng, grid, k)
    chs = choose_channels(rng, channels)
    base = {}
    for ch in chs:
        if ch == "V":
            base[ch] = rand_signed(rng, *cfg["voltage_offset_pu"])
        elif ch == "Ang":
            base[ch] = rand_signed(rng, *cfg["angle_offset_deg"])
        else:
            base[ch] = rand_signed(rng, 3.0, 12.0)
    offsets = {}
    for b, h in hops.items():
        w = cfg["hop_decay"] ** h
        offsets[b] = {ch: base[ch] * w * float(rng.uniform(0.85, 1.15)) * intensity for ch in chs}
    return {b: list(chs) for b in hops}, {"seed_bus": seed, "hops": hops,
                                          "base_offsets": base, "offsets": offsets}


def apply(event: AttackEvent, snap, bus_index: dict[int, int], **_) -> None:
    for b, chs in event.bus_channels.items():
        i = bus_index[b]
        for ch in chs:
            snap.meas_bus.channel(ch)[i] += event.params["offsets"][b][ch]
