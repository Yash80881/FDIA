"""Automatic randomised FDIA scheduler and ground-truth label generator.

At every step the scheduler decides (with a seeded RNG) whether an attack
starts, its type, duration, compromised buses, channels and magnitudes, then
applies the active attack to the received-measurement layer and produces the
ground-truth labels.

Start process: after ``initial_delay_steps`` and at least ``min_gap_steps``
normal steps since the previous attack, an attack starts with probability
``start_probability`` per step (geometric inter-arrival times), so the attack
sequence is random rather than a fixed repeated pattern.

Labels are derived from the *actual* perturbation: bus i is labelled attacked
iff at least one of its received measurements differs from the clean received
value.  Severity (dimensionless, 0 for normal operation) is the noise-normalised
RMS perturbation over the attacked bus measurements:

    severity = sqrt( mean_{(i,c) attacked} ((z_a[i,c] - z[i,c]) / sigma_c)^2 )

i.e. "how many noise standard deviations the attacker moved the sensors".
"""
from __future__ import annotations

import json
import logging
from collections import OrderedDict

import numpy as np

from attacks import additive, scaling, replay, coordinated
from attacks.base import AttackEvent
from simulator.measurement_generator import Snapshot, MeasurementGenerator, line_loading
from simulator.schema import ATTACK_TYPES

log = logging.getLogger(__name__)


class AttackScheduler:
    def __init__(self, grid, cfg: dict, rng: np.random.Generator, run_id: str,
                 meas_gen: MeasurementGenerator):
        self.grid = grid
        self.cfg = cfg["attacks"]
        self.rng = rng
        self.run_id = run_id
        self.meas_gen = meas_gen
        self.current: AttackEvent | None = None
        self.counter = 0
        self.last_end_step = -10**9
        self.events: list[AttackEvent] = []
        self.history: "OrderedDict[int, object]" = OrderedDict()
        self.max_history = max(self.cfg["replay"]["lag_steps"][1], self.cfg["replay"]["min_history_steps"]) \
            + self.cfg["duration_steps"][1] + 5
        channels = list(self.cfg["channels"])
        if self.cfg.get("allow_power_channels"):
            channels += [c for c in ("P", "Q") if c not in channels]
        self.channels = channels

    # ------------------------------------------------------------ scheduling
    def _choose_type(self) -> str:
        weights = dict(self.cfg["type_weights"])
        if len(self.history) < self.cfg["replay"]["min_history_steps"]:
            weights["replay"] = 0.0   # not enough recorded history for a replay yet
        names = [t for t in ATTACK_TYPES if weights.get(t, 0) > 0]
        w = np.array([weights[t] for t in names], dtype=float)
        return str(self.rng.choice(names, p=w / w.sum()))

    def _bus_count(self, lo_hi) -> int:
        lo, hi = lo_hi
        return int(self.rng.integers(lo, hi + 1))

    def _new_event(self, step: int) -> AttackEvent:
        atype = self._choose_type()
        tcfg = self.cfg[atype]
        intensity = float(self.cfg["intensity"])
        lo, hi = self.cfg["duration_steps"]
        duration = int(self.rng.integers(lo, hi + 1))
        if atype == "coordinated":
            bus_channels, params = coordinated.create(self.rng, self.grid, self._bus_count(tcfg["bus_count"]),
                                                      self.channels, tcfg, intensity)
        else:
            k = self._bus_count(tcfg["bus_count"])
            buses = sorted(int(b) for b in self.rng.choice(self.grid.bus_numbers, size=k, replace=False))
            if atype == "additive":
                bus_channels, params = additive.create(self.rng, buses, self.channels, tcfg, intensity)
            elif atype == "scaling":
                bus_channels, params = scaling.create(self.rng, buses, self.channels, tcfg, intensity,
                                                      self.grid.ref_bus)
            else:
                bus_channels, params = replay.create(self.rng, buses, self.channels, tcfg, len(self.history))
                # the played-back recording must exist for the whole duration
                duration = min(duration, params["lag_steps"])
        self.counter += 1
        event = AttackEvent(attack_id=f"{self.run_id}-A{self.counter:04d}", attack_type=atype,
                            start_step=step, duration_steps=duration,
                            bus_channels=bus_channels, params=params)
        if self.cfg.get("line_sensor_attack") and atype in ("additive", "coordinated"):
            for b in event.buses:
                for k_line in self.grid.incident_branches[b]:
                    event.line_offsets[k_line] = float(self.rng.uniform(2, 8) * self.rng.choice([-1, 1]))
        log.info("Attack %s started at step %d: %s on buses %s channels %s for %d steps",
                 event.attack_id, step, atype, event.buses, event.channels, duration)
        self.events.append(event)
        return event

    def _maybe_start(self, step: int) -> None:
        if not self.cfg["enabled"] or self.current is not None:
            return
        if step < self.cfg["initial_delay_steps"]:
            return
        if step - self.last_end_step <= self.cfg["min_gap_steps"]:
            return
        if self.rng.random() < self.cfg["start_probability"]:
            self.current = self._new_event(step)

    # --------------------------------------------------------------- applying
    def process(self, snap: Snapshot) -> dict:
        """Apply the active attack (if any) to ``snap`` and return its labels."""
        step = snap.step
        self._maybe_start(step)
        event = self.current
        if event is not None:
            module = {"additive": additive, "scaling": scaling,
                      "replay": replay, "coordinated": coordinated}[event.attack_type]
            module.apply(event, snap, self.grid.bus_index, history=self.history)
            if event.line_offsets:
                for k_line, off in event.line_offsets.items():
                    snap.meas_line.p_from[k_line - 1] += off
                ml = snap.meas_line
                ml.loading = line_loading(ml.p_from, ml.q_from, ml.p_to, ml.q_to, self.grid.rate_a)
        labels = self._labels(snap, event)
        # record the legitimate received measurements (attacker's recording / replay source)
        self.history[step] = snap.clean_meas_bus
        while len(self.history) > self.max_history:
            self.history.popitem(last=False)
        if event is not None and step >= event.end_step:
            self.last_end_step = step
            self.current = None
        return labels

    def _labels(self, snap: Snapshot, event: AttackEvent | None) -> dict:
        n = self.grid.n_bus
        bus_attacked = np.zeros(n, dtype=int)
        z = []
        for ch in ("V", "Ang", "P", "Q"):
            clean = snap.clean_meas_bus.channel(ch)
            delta = snap.meas_bus.channel(ch) - clean
            changed = np.abs(delta) > 1e-12
            bus_attacked |= changed.astype(int)
            if changed.any():
                sigma = self.meas_gen.channel_sigma(ch, clean)
                z.extend((delta[changed] / sigma[changed]).tolist())
        line_changed = bool(event is not None and event.line_offsets)
        active = int(bus_attacked.any() or line_changed)
        if event is not None and not active:
            # Should never happen: an active event that changed nothing.
            log.warning("Attack %s produced no perturbation at step %d", event.attack_id, snap.step)
        labels = {
            "attack_active": active,
            "attack_id": event.attack_id if active else "",
            "attack_type": event.attack_type if active else "normal",
            "attack_duration_steps": event.duration_steps if active else 0,
            "attack_step_in_event": (snap.step - event.start_step) if active else -1,
            "attacked_bus_count": int(bus_attacked.sum()),
            "attack_severity": float(np.sqrt(np.mean(np.square(z)))) if z else 0.0,
            "attack_channels": json.dumps(event.channels) if active else "[]",
            "active_attack_count": active,
        }
        for i, b in enumerate(self.grid.bus_numbers):
            labels[f"attack_bus_{b}"] = int(bus_attacked[i])
        return labels
