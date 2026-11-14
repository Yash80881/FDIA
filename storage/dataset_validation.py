"""Dataset validation and integrity report.

Checks (each reported as PASS/FAIL):
 * column structure equals the current schema (no incompatible appends)
 * no missing values outside columns where NaN is meaningful (unrated line loading)
 * every stored snapshot comes from a converged power flow
 * label consistency: any attack_bus_i = 1  <=>  attack_active = 1
 * attacked_bus_count == sum(attack_bus_i)
 * normal rows: attack_type == 'normal', severity == 0, empty attack_id
 * attacked rows: valid attack type, non-empty attack id, severity > 0
 * received measurements of non-attacked buses are within a plausible noise band
   of the true values (ground-truth preservation)
 * steps are unique and increasing per run
Also prints attack-type distributions, per-bus attack counts and missing values.

    python -m storage.dataset_validation --run <run_id> | --all | --csv file.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import load_settings, resolve_path
from simulator import schema as S


def validate_dataframe(df: pd.DataFrame, noise_cfg: dict | None = None) -> dict:
    checks: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    missing_cols = [c for c in S.ALL_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in S.ALL_COLUMNS]
    check("schema columns", not missing_cols and not extra_cols,
          f"missing={missing_cols[:5]} extra={extra_cols[:5]}")
    if missing_cols:
        return {"passed": False, "checks": checks, "n_rows": len(df)}

    nan_ok = set(S.line_cols("loading")) | set(S.line_cols("loading", true=True))
    text_cols = list(S.TEXT_COLUMNS)
    numeric = df.drop(columns=[c for c in text_cols if c in df.columns])
    na = numeric.isna().sum()
    na_bad = na[(na > 0) & (~na.index.isin(nan_ok))]
    check("missing values", na_bad.empty, f"{len(na_bad)} columns with NaN: {list(na_bad.index[:5])}")
    loading_nan = int(na[na.index.isin(nan_ok)].sum())

    check("power flow converged", (df["pf_converged"] == 1).all(),
          f"{int((df['pf_converged'] != 1).sum())} non-converged rows")

    bus_lab = df[S.ATTACK_BUS_COLUMNS].to_numpy()
    any_bus = bus_lab.any(axis=1)
    check("bus labels imply global label", np.all(df["attack_active"].to_numpy()[any_bus] == 1))
    check("global label implies a bus label (bus-channel attacks)",
          np.all(any_bus[df["attack_active"].to_numpy() == 1]),
          "rows with attack_active=1 but no bus label (line-only attacks)")
    check("attacked_bus_count consistent", np.all(bus_lab.sum(axis=1) == df["attacked_bus_count"].to_numpy()))

    normal = df["attack_active"] == 0
    check("normal rows labelled 'normal'", (df.loc[normal, "attack_type"] == "normal").all())
    check("normal rows have zero severity", (df.loc[normal, "attack_severity"] == 0).all())
    check("normal rows have no attack id", (df.loc[normal, "attack_id"].fillna("") == "").all())
    att = ~normal
    check("attack rows have valid type", df.loc[att, "attack_type"].isin(S.ATTACK_TYPES).all())
    check("attack rows have attack id", (df.loc[att, "attack_id"].fillna("") != "").all())
    check("attack rows have severity > 0", (df.loc[att, "attack_severity"] > 0).all())

    # ground-truth preservation: on non-attacked buses |meas - true| stays within 6 sigma
    if noise_cfg:
        sv, sa = noise_cfg["voltage_pu"], noise_cfg["angle_deg"]
        dv = (df[S.bus_cols("V_meas")].to_numpy() - df[S.bus_cols("V_true")].to_numpy())
        da = (df[S.bus_cols("Ang_meas")].to_numpy() - df[S.bus_cols("Ang_true")].to_numpy())
        clean = bus_lab == 0
        check("non-attacked V within 6 sigma of truth", np.all(np.abs(dv[clean]) < 6 * sv),
              f"max |dV|={np.abs(dv[clean]).max():.5f}")
        check("non-attacked angle within 6 sigma of truth", np.all(np.abs(da[clean]) < 6 * sa),
              f"max |dAng|={np.abs(da[clean]).max():.4f}")
        check("noise present on normal rows", np.std(dv[clean]) > 0.25 * sv,
              f"std dV={np.std(dv[clean]):.5f}")

    for rid, g in df.groupby("run_id"):
        steps = g["step"].to_numpy()
        check(f"steps unique/increasing [{rid}]", np.all(np.diff(steps) > 0))

    # event consistency: one type per attack id, contiguous steps
    ev_ok = True
    for aid, g in df[att].groupby(["run_id", "attack_id"]):
        if g["attack_type"].nunique() != 1 or np.any(np.diff(g["step"].to_numpy()) != 1):
            ev_ok = False
            break
    check("each attack id has one type and contiguous steps", ev_ok)

    report = {
        "passed": all(ok for _, ok, _ in checks),
        "checks": checks,
        "n_rows": int(len(df)),
        "n_runs": int(df["run_id"].nunique()),
        "attack_fraction": float(att.mean()),
        "attack_type_steps": df["attack_type"].value_counts().to_dict(),
        "attack_type_events": df[att].groupby("attack_type")["attack_id"].nunique().to_dict(),
        "per_bus_attacked_steps": {c.replace("attack_bus_", ""): int(df[c].sum()) for c in S.ATTACK_BUS_COLUMNS},
        "severity_by_type": df[att].groupby("attack_type")["attack_severity"].describe().round(3).to_dict("index"),
        "unrated_line_loading_nan_cells": loading_nan,
        "total_load_range_mw": [float(df["total_load_p_mw"].min()), float(df["total_load_p_mw"].max())],
    }
    return report


def validate_runs(repo, run_ids: list[str]) -> dict:
    """Validate each run with the noise settings it was generated with (stored
    in the runs table), then report the distributions over all runs."""
    frames, checks, passed = [], [], True
    for rid in run_ids:
        run = repo.get_run(rid)
        noise = json.loads(run["config_json"])["noise"] if run and run.get("config_json") else None
        df = repo.load_run(rid)
        if df.empty:
            checks.append((f"[{rid}] has measurements", False, "empty run"))
            passed = False
            continue
        r = validate_dataframe(df, noise)
        passed &= r["passed"]
        checks += [(f"[{rid}] {name}", ok, detail) for name, ok, detail in r["checks"]]
        frames.append(df)
    rep = validate_dataframe(pd.concat(frames, ignore_index=True), None) if frames else {"checks": []}
    rep["checks"] = checks
    rep["passed"] = bool(passed and frames)
    return rep


def print_report(rep: dict) -> None:
    print(f"Rows: {rep['n_rows']}  runs: {rep.get('n_runs')}  attack fraction: {rep.get('attack_fraction', 0):.3f}")
    for name, ok, detail in rep["checks"]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    if "attack_type_steps" in rep:
        print("Attack-type steps :", rep["attack_type_steps"])
        print("Attack-type events:", rep["attack_type_events"])
        print("Per-bus attacked steps:", rep["per_bus_attacked_steps"])
        print("Total load range (MW):", [round(x, 2) for x in rep["total_load_range_mw"]])
    print("OVERALL:", "PASS" if rep["passed"] else "FAIL")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate GridSentinel datasets")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="append", help="run ID (repeatable)")
    g.add_argument("--all", action="store_true", help="all runs in the database")
    g.add_argument("--csv", help="exported CSV file")
    ap.add_argument("--json", help="write the report to this JSON file")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = load_settings(args.config, args.set)
    if args.csv:
        df = pd.read_csv(args.csv, keep_default_na=True)
        for c in S.TEXT_COLUMNS:
            if c in df:
                df[c] = df[c].fillna("")
    else:
        from storage.database import Database
        from storage.repository import Repository
        repo = Repository(Database(resolve_path(cfg, "database")))
        runs = args.run or [r["run_id"] for r in repo.list_runs()]
        rep = validate_runs(repo, runs)
        print_report(rep)
        if args.json:
            Path(args.json).write_text(json.dumps(rep, indent=2, default=str))
        return 0 if rep["passed"] else 1
    rep = validate_dataframe(df, cfg["noise"])
    print_report(rep)
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=2, default=str))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
