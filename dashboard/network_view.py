"""Interactive IEEE 30-bus digital-twin view for the dashboard.

Uses the shared topology renderer (:mod:`simulator.network_visualization`) and
feeds it live API data: flows/loading of the latest snapshot, ML-predicted
compromised buses (red fill) and simulator ground truth (black ring) as
separate visual channels.
"""
from __future__ import annotations

from dash import html

from simulator.grid_model import GridModel
from simulator.network_visualization import network_figure

_GRID = GridModel()


def twin_figure(network: dict, buses: list[dict] | None, lines: list[dict] | None):
    layout = {b["bus"]: (b["x"], b["y"]) for b in network["buses"]}
    p_from = loading = vm = None
    predicted, truth, probs = set(), set(), None
    if lines:
        p_from = [ln["received"]["P_from"] for ln in lines]
        loading = [ln["received"]["loading"] if ln["received"]["loading"] is not None else float("nan")
                   for ln in lines]
    if buses:
        vm = [b["received"]["V_pu"] for b in buses]
        truth = {b["bus"] for b in buses if b["ground_truth"]["attacked"]}
        if buses[0].get("ai"):
            predicted = {b["bus"] for b in buses if b["ai"]["predicted_compromised"]}
            probs = {b["bus"]: b["ai"]["attack_probability"] for b in buses}
    import numpy as np
    fig = network_figure(_GRID, layout, p_from=p_from, loading=np.array(loading) if loading else None,
                         vm=vm, predicted_buses=predicted, truth_buses=truth, bus_prob=probs,
                         title="IEEE 30-bus digital twin (live)", height=640, dark=True)
    return fig


def bus_details(bus: dict | None) -> html.Div:
    if bus is None:
        return html.Div("Click a bus in the network (or pick one) to see its details.", className="muted")
    r, g, ai = bus["received"], bus["ground_truth"], bus.get("ai")
    rows = [
        ("Bus", f"{bus['bus']} ({bus['type']}{', generator' if bus['is_generator'] else ''})"),
        ("Voltage magnitude (received)", f"{r['V_pu']:.4f} pu"),
        ("Voltage angle (received)", f"{r['Ang_deg']:.3f}°"),
        ("Active power injection (received)", f"{r['P_mw']:.2f} MW"),
        ("Reactive power injection (received)", f"{r['Q_mvar']:.2f} MVAr"),
        ("Voltage limits", f"{bus['voltage_limits'][0]}–{bus['voltage_limits'][1]} pu"
                           + ("  ⚠ violation" if bus["voltage_violation"] else "")),
    ]
    ai_rows = [("AI attack probability", f"{ai['attack_probability']:.3f} (threshold {ai['threshold']:.2f})"),
               ("AI prediction", "COMPROMISED" if ai["predicted_compromised"] else "normal")] if ai else \
        [("AI prediction", "models not loaded")]
    gt_rows = [("True V / angle", f"{g['V_pu']:.4f} pu / {g['Ang_deg']:.3f}°"),
               ("True load", f"{g['Pd_mw']:.2f} MW / {g['Qd_mvar']:.2f} MVAr"),
               ("Simulator label", "ATTACKED" if g["attacked"] else "normal")]

    def table(title, items, cls):
        return html.Div([html.Div(title, className=f"section-tag {cls}"),
                         html.Table([html.Tr([html.Td(k, className="k"), html.Td(v, className="v")])
                                     for k, v in items], className="kv")])
    return html.Div([table("Received measurements", rows, "tag-meas"),
                     table("AI prediction (Random Forest)", ai_rows, "tag-ai"),
                     table("Simulator ground truth (evaluation only)", gt_rows, "tag-gt")])
