"""Measurement / label schema shared by the simulator, storage, ML and API.

Naming (bus i = 1..30, line k = 1..41, generator g = 1..6):

    true electrical state       V_true_bus_i  Ang_true_bus_i  P_true_bus_i  Q_true_bus_i
    received (noisy/attacked)   V_meas_bus_i  Ang_meas_bus_i  P_meas_bus_i  Q_meas_bus_i
    line received measurements  P_from_line_k Q_from_line_k P_to_line_k Q_to_line_k
                                I_from_line_k I_to_line_k loading_line_k
    line true values            P_from_true_line_k ... loading_true_line_k
    generators (true)           Pg_true_gen_g  Qg_true_gen_g
    labels                      attack_active attack_id attack_type ... attack_bus_i

Units: V in pu, angles in degrees, P in MW, Q in MVAr, currents in A, loading in %.

Sign convention for bus injections: P_bus_i / Q_bus_i are the *net injections
into the network* at bus i, P = Pg - Pd and Q = Qg - Qd (generation positive,
load consumption negative).  Bus shunts are part of the network model (Ybus).
Branch flows P_from/Q_from are measured at the from-bus terminal flowing *into*
the branch; P_to/Q_to at the to-bus terminal flowing into the branch, so
P_from + P_to = active losses of the branch.

Only ``ml_input_columns`` may ever be used as machine-learning inputs.
"""
from __future__ import annotations

SCHEMA_VERSION = 2   # bump when the column structure changes (v1 = previous project)

N_BUS = 30
N_LINE = 41
N_GEN = 6

ATTACK_TYPES = ["additive", "scaling", "replay", "coordinated"]
ALL_ATTACK_LABELS = ["normal"] + ATTACK_TYPES

BUS_TRUE_PREFIXES = ["V_true", "Ang_true", "P_true", "Q_true"]
BUS_MEAS_PREFIXES = ["V_meas", "Ang_meas", "P_meas", "Q_meas"]
LINE_QUANTITIES = ["P_from", "Q_from", "P_to", "Q_to", "I_from", "I_to", "loading"]

META_COLUMNS = [
    "run_id", "step", "timestamp", "sim_time_s",
    "pf_converged", "pf_iterations", "scenario",
    "total_load_p_mw", "total_load_q_mvar", "total_gen_p_mw", "total_gen_q_mvar",
    "losses_p_mw", "gen_limit_violations",
]

LABEL_COLUMNS = [
    "attack_active", "attack_id", "attack_type", "attack_duration_steps",
    "attack_step_in_event", "attacked_bus_count", "attack_severity",
    "attack_channels", "active_attack_count",
]


def bus_cols(prefix: str, n: int = N_BUS) -> list[str]:
    return [f"{prefix}_bus_{i}" for i in range(1, n + 1)]


def line_cols(quantity: str, true: bool = False, n: int = N_LINE) -> list[str]:
    tag = f"{quantity}_true" if true else quantity
    return [f"{tag}_line_{k}" for k in range(1, n + 1)]


def gen_cols(prefix: str, n: int = N_GEN) -> list[str]:
    return [f"{prefix}_gen_{g}" for g in range(1, n + 1)]


ATTACK_BUS_COLUMNS = bus_cols("attack")

BUS_TRUE_COLUMNS = [c for p in BUS_TRUE_PREFIXES for c in bus_cols(p)]
BUS_MEAS_COLUMNS = [c for p in BUS_MEAS_PREFIXES for c in bus_cols(p)]
LINE_MEAS_COLUMNS = [c for q in LINE_QUANTITIES for c in line_cols(q)]
LINE_TRUE_COLUMNS = [c for q in LINE_QUANTITIES for c in line_cols(q, true=True)]
GEN_COLUMNS = gen_cols("Pg_true") + gen_cols("Qg_true")
LOAD_COLUMNS = bus_cols("Pd_true") + bus_cols("Qd_true")

ALL_COLUMNS = (META_COLUMNS + BUS_TRUE_COLUMNS + BUS_MEAS_COLUMNS + LINE_MEAS_COLUMNS
               + LINE_TRUE_COLUMNS + GEN_COLUMNS + LOAD_COLUMNS
               + LABEL_COLUMNS + ATTACK_BUS_COLUMNS)

TEXT_COLUMNS = {"run_id", "timestamp", "scenario", "attack_id", "attack_type",
                "attack_channels", "gen_limit_violations"}
INTEGER_COLUMNS = {"step", "pf_converged", "pf_iterations", "attack_active",
                   "attack_duration_steps", "attack_step_in_event",
                   "attacked_bus_count", "active_attack_count"} | set(ATTACK_BUS_COLUMNS)

# Columns the ML models may consume: *received* measurements only.
ML_INPUT_COLUMNS = BUS_MEAS_COLUMNS + LINE_MEAS_COLUMNS

# Anything here must never reach a model (checked by tests and at training).
FORBIDDEN_FEATURE_TOKENS = ("_true", "attack", "severity", "scenario", "Pd_", "Qd_",
                            "run_id", "timestamp")


def assert_no_leakage(feature_names: list[str]) -> None:
    bad = [f for f in feature_names if any(tok in f for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise ValueError(f"Target leakage: forbidden columns in feature set: {bad[:10]}")
