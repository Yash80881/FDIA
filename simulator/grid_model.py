"""Authentic IEEE 30-bus network model loaded from PYPOWER.

Every quantity (buses, branches, generators, limits, ratings, shunts) is read
from :func:`pypower.case30.case30`; nothing about the topology is invented.
The PYPOWER source is never modified - we only deep-copy the case dictionary.

Note on the case: PYPOWER's ``case30`` is the MATPOWER 30-bus case based on
Alsac & Stott (1974) with generator data from Ferrero et al. (1997):
6 generators at buses 1, 2, 22, 27, 23, 13 and a base load of 189.2 MW.
(This differs from the "IEEE 30-bus" variant with generators at buses
1, 2, 5, 8, 11, 13; we use exactly what PYPOWER ships.)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from pypower.case30 import case30
from pypower import idx_bus as IB
from pypower import idx_brch as IBR
from pypower import idx_gen as IG

BUS_TYPE_NAMES = {1: "PQ", 2: "PV", 3: "REF", 4: "ISOLATED"}


@dataclass(frozen=True)
class BranchInfo:
    index: int          # 1-based line number used in column names (line_1 ... line_41)
    from_bus: int       # external bus number
    to_bus: int
    r_pu: float
    x_pu: float
    b_pu: float
    rate_a_mva: float | None   # None when the case gives no valid rating
    tap: float
    is_transformer: bool


@dataclass
class GridModel:
    """Read-only description of the PYPOWER IEEE 30-bus case."""

    base_case: dict = field(default_factory=case30)

    def __post_init__(self) -> None:
        self.base_case = copy.deepcopy(self.base_case)
        bus = self.base_case["bus"]
        self.bus_numbers: list[int] = [int(b) for b in bus[:, IB.BUS_I]]
        # external bus number -> row index in ppc['bus']
        self.bus_index: dict[int, int] = {b: i for i, b in enumerate(self.bus_numbers)}
        self.base_mva: float = float(self.base_case["baseMVA"])

    # ------------------------------------------------------------------ sizes
    @property
    def n_bus(self) -> int:
        return self.base_case["bus"].shape[0]

    @property
    def n_branch(self) -> int:
        return self.base_case["branch"].shape[0]

    @property
    def n_gen(self) -> int:
        return self.base_case["gen"].shape[0]

    # ------------------------------------------------------------- bus data
    @cached_property
    def bus_types(self) -> dict[int, str]:
        bus = self.base_case["bus"]
        return {int(b[IB.BUS_I]): BUS_TYPE_NAMES.get(int(b[IB.BUS_TYPE]), "?") for b in bus}

    @cached_property
    def ref_bus(self) -> int:
        refs = [b for b, t in self.bus_types.items() if t == "REF"]
        if len(refs) != 1:
            raise ValueError(f"Expected exactly one reference bus, found {refs}")
        return refs[0]

    @cached_property
    def generator_buses(self) -> list[int]:
        return [int(g) for g in self.base_case["gen"][:, IG.GEN_BUS]]

    @cached_property
    def load_buses(self) -> list[int]:
        bus = self.base_case["bus"]
        return [int(b[IB.BUS_I]) for b in bus if b[IB.PD] != 0 or b[IB.QD] != 0]

    @cached_property
    def base_pd(self) -> np.ndarray:
        return self.base_case["bus"][:, IB.PD].copy()

    @cached_property
    def base_qd(self) -> np.ndarray:
        return self.base_case["bus"][:, IB.QD].copy()

    @cached_property
    def base_kv(self) -> np.ndarray:
        return self.base_case["bus"][:, IB.BASE_KV].copy()

    @cached_property
    def voltage_limits(self) -> dict[int, tuple[float, float]]:
        bus = self.base_case["bus"]
        return {int(b[IB.BUS_I]): (float(b[IB.VMIN]), float(b[IB.VMAX])) for b in bus}

    @cached_property
    def shunts(self) -> dict[int, tuple[float, float]]:
        """Bus shunts (Gs MW, Bs MVAr at V = 1 pu) for buses that have one."""
        bus = self.base_case["bus"]
        return {int(b[IB.BUS_I]): (float(b[IB.GS]), float(b[IB.BS]))
                for b in bus if b[IB.GS] != 0 or b[IB.BS] != 0}

    # ---------------------------------------------------------- branch data
    @cached_property
    def branches(self) -> list[BranchInfo]:
        out = []
        for k, br in enumerate(self.base_case["branch"]):
            rate = float(br[IBR.RATE_A])
            tap = float(br[IBR.TAP])
            out.append(BranchInfo(
                index=k + 1,
                from_bus=int(br[IBR.F_BUS]), to_bus=int(br[IBR.T_BUS]),
                r_pu=float(br[IBR.BR_R]), x_pu=float(br[IBR.BR_X]), b_pu=float(br[IBR.BR_B]),
                rate_a_mva=rate if np.isfinite(rate) and rate > 0 else None,
                tap=tap if tap != 0 else 1.0,
                is_transformer=tap != 0,
            ))
        return out

    @cached_property
    def branch_from_idx(self) -> np.ndarray:
        return np.array([self.bus_index[b.from_bus] for b in self.branches])

    @cached_property
    def branch_to_idx(self) -> np.ndarray:
        return np.array([self.bus_index[b.to_bus] for b in self.branches])

    @cached_property
    def rate_a(self) -> np.ndarray:
        """Branch rating in MVA, NaN where unavailable."""
        return np.array([b.rate_a_mva if b.rate_a_mva else np.nan for b in self.branches])

    @cached_property
    def adjacency(self) -> dict[int, list[int]]:
        adj: dict[int, set[int]] = {b: set() for b in self.bus_numbers}
        for br in self.branches:
            adj[br.from_bus].add(br.to_bus)
            adj[br.to_bus].add(br.from_bus)
        return {b: sorted(n) for b, n in adj.items()}

    @cached_property
    def incident_branches(self) -> dict[int, list[int]]:
        """bus -> list of 1-based branch numbers connected to it."""
        inc: dict[int, list[int]] = {b: [] for b in self.bus_numbers}
        for br in self.branches:
            inc[br.from_bus].append(br.index)
            inc[br.to_bus].append(br.index)
        return inc

    def hop_distances(self, source: int) -> dict[int, int]:
        """Breadth-first graph distance from ``source`` to every bus."""
        dist = {source: 0}
        frontier = [source]
        while frontier:
            nxt = []
            for b in frontier:
                for n in self.adjacency[b]:
                    if n not in dist:
                        dist[n] = dist[b] + 1
                        nxt.append(n)
            frontier = nxt
        return dist

    # ------------------------------------------------------------ generators
    @cached_property
    def generators(self) -> list[dict]:
        gens = []
        for k, g in enumerate(self.base_case["gen"]):
            gens.append({
                "index": k + 1,
                "bus": int(g[IG.GEN_BUS]),
                "pg_mw": float(g[IG.PG]), "qg_mvar": float(g[IG.QG]),
                "pmin_mw": float(g[IG.PMIN]), "pmax_mw": float(g[IG.PMAX]),
                "qmin_mvar": float(g[IG.QMIN]), "qmax_mvar": float(g[IG.QMAX]),
                "vg_pu": float(g[IG.VG]), "in_service": bool(g[IG.GEN_STATUS] > 0),
            })
        return gens

    def new_case(self) -> dict:
        """Fresh deep copy of the base case for one power-flow solution."""
        return copy.deepcopy(self.base_case)

    def summary(self) -> dict:
        """Network facts verified directly from the loaded case."""
        return {
            "case": "PYPOWER case30",
            "base_mva": self.base_mva,
            "n_bus": self.n_bus,
            "n_branch": self.n_branch,
            "n_gen": self.n_gen,
            "ref_bus": self.ref_bus,
            "generator_buses": self.generator_buses,
            "pv_buses": [b for b, t in self.bus_types.items() if t == "PV"],
            "n_load_buses": len(self.load_buses),
            "base_total_load_mw": float(self.base_pd.sum()),
            "base_total_load_mvar": float(self.base_qd.sum()),
            "n_transformers": sum(b.is_transformer for b in self.branches),
            "n_rated_branches": sum(b.rate_a_mva is not None for b in self.branches),
            "shunts": self.shunts,
            "base_kv": sorted(set(float(v) for v in self.base_kv)),
        }
