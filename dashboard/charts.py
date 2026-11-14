"""Plotly figure builders for the dashboard (pure functions of API payloads)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TYPE_COLORS = {"additive": "#ef4444", "scaling": "#f59e0b", "replay": "#8b5cf6", "coordinated": "#06b6d4",
               "normal": "#22c55e", "unknown": "#94a3b8"}
BG = "rgba(0,0,0,0)"
GRID = "rgba(148,163,184,0.18)"
FG = "#cbd5e1"


def themed(fig: go.Figure, title: str | None = None, height: int = 320) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=height,
                      margin=dict(l=50, r=15, t=40 if title else 15, b=40), font=dict(color=FG, size=11),
                      title=dict(text=title, font=dict(size=13)) if title else None,
                      legend=dict(orientation="h", y=-0.22, font=dict(size=10)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def empty(message: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=13, color="#f87171"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return themed(fig, height=height)


def _attack_shapes(steps, active, fig, row=None, col=None):
    """Shade ground-truth attack intervals (labelled as ground truth in the legend)."""
    if not steps:
        return
    start = None
    for s, a in zip(steps + [None], list(active) + [0]):
        if a and start is None:
            start = s
        elif not a and start is not None:
            end = s if s is not None else steps[-1]
            kw = dict(row=row, col=col) if row else {}
            fig.add_vrect(x0=start, x1=end, fillcolor="rgba(239,68,68,0.13)", line_width=0, layer="below", **kw)
            start = None


def load_history(h: dict) -> go.Figure:
    d = h["data"]
    if not d.get("step"):
        return empty("No history yet")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=d["step"], y=d["total_load_p_mw"], name="Total load P (MW)",
                             line=dict(color="#38bdf8")), secondary_y=False)
    fig.add_trace(go.Scatter(x=d["step"], y=d["total_gen_p_mw"], name="Total generation P (MW)",
                             line=dict(color="#a3e635", dash="dot")), secondary_y=False)
    fig.add_trace(go.Scatter(x=d["step"], y=d["total_load_q_mvar"], name="Total load Q (MVAr)",
                             line=dict(color="#f472b6")), secondary_y=True)
    fig.update_yaxes(title_text="MW", secondary_y=False)
    fig.update_yaxes(title_text="MVAr", secondary_y=True)
    return themed(fig, "System demand and generation (solved power flow)")


def probability_history(h: dict) -> go.Figure:
    d, ai = h["data"], h.get("ai")
    if not d.get("step"):
        return empty("No history yet")
    fig = go.Figure()
    _attack_shapes(d["step"], d.get("attack_active", []), fig)
    if ai:
        # fig.add_trace(go.Scatter(x=ai["step"], y=ai["attack_probability"], name="AI attack probability",
                fig.add_trace(go.Scatter(x=ai["step"], y=ai["attack_probability"], name="AI attack probability",
                                 connectgaps=True,
                                 line=dict(color="#fbbf24", width=2)))
    fig.add_trace(go.Scatter(x=d["step"], y=d.get("attack_active", []), name="Ground truth attack (simulator)",
                             line=dict(color="#ef4444", width=1, shape="hv", dash="dot")))
    fig.update_yaxes(range=[-0.05, 1.05], title="probability")
    return themed(fig, "AI detection probability vs simulator ground truth (red shading = true attack)")


def voltage_heatmap(h: dict) -> go.Figure:
    d = h["data"]
    if not d.get("step"):
        return empty("No history yet")
    z = np.array([d[f"V_meas_bus_{b}"] for b in range(1, 31)])
    fig = go.Figure(go.Heatmap(z=z, x=d["step"], y=[f"Bus {b}" for b in range(1, 31)], colorscale="RdBu_r",
                               zmid=1.0, colorbar=dict(title="pu")))
    return themed(fig, "Received voltage magnitude, all 30 buses", height=480)


def bus_bar(values, title, ytitle, limits=None, highlight=None, truth=None, est=None) -> go.Figure:
    buses = list(range(1, len(values) + 1))
    colors = ["#ef4444" if highlight and b in highlight else "#38bdf8" for b in buses]
    fig = go.Figure(go.Bar(x=buses, y=values, name="Received", marker_color=colors))
    if truth is not None:
        fig.add_trace(go.Scatter(x=buses, y=truth, mode="markers", name="Ground truth (simulator)",
                                 marker=dict(color="#f8fafc", symbol="line-ew-open", size=14, line=dict(width=2))))
    if est is not None:
        fig.add_trace(go.Scatter(x=buses, y=est, mode="markers", name="WLS estimate",
                                 marker=dict(color="#a3e635", symbol="diamond", size=7)))
    if limits:
        for v in limits:
            fig.add_hline(y=v, line=dict(color="#f87171", dash="dash", width=1))
    if limits:
        lo = min(min(values), limits[0]) - 0.01
        hi = max(max(values), limits[1]) + 0.01
        fig.update_yaxes(range=[lo, hi])
    fig.update_xaxes(title="bus", dtick=1)
    fig.update_yaxes(title=ytitle)
    return themed(fig, title)


def multi_line(h: dict, cols: list[str], title: str, ytitle: str, names: list[str] | None = None) -> go.Figure:
    d = h["data"]
    if not d.get("step"):
        return empty("No history yet")
    fig = go.Figure()
    _attack_shapes(d["step"], d.get("attack_active", []), fig)
    for i, c in enumerate(cols):
        if c in d:
            fig.add_trace(go.Scatter(x=d["step"], y=d[c], name=(names[i] if names else c), mode="lines"))
    fig.update_yaxes(title=ytitle)
    fig.update_xaxes(title="step")
    return themed(fig, title)


def line_bar(values, title, ytitle, ref=None) -> go.Figure:
    lines = list(range(1, len(values) + 1))
    vals = [np.nan if v is None else v for v in values]
    colors = ["#ef4444" if ref and v is not None and v >= ref else
              ("#f59e0b" if ref and v is not None and v >= 0.9 * ref else "#38bdf8") for v in values]
    fig = go.Figure(go.Bar(x=lines, y=vals, marker_color=colors))
    if ref:
        fig.add_hline(y=ref, line=dict(color="#f87171", dash="dash"))
    fig.update_xaxes(title="line", dtick=2)
    fig.update_yaxes(title=ytitle)
    return themed(fig, title)


def two_end_bar(a, b, name_a, name_b, title, ytitle) -> go.Figure:
    lines = list(range(1, len(a) + 1))
    fig = go.Figure([go.Bar(x=lines, y=a, name=name_a, marker_color="#38bdf8"),
                     go.Bar(x=lines, y=b, name=name_b, marker_color="#f472b6")])
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="line", dtick=2)
    fig.update_yaxes(title=ytitle)
    return themed(fig, title)


def angle_diff_bar(network: dict, angles: list[float]) -> go.Figure:
    br = network["branches"]
    diffs = [angles[x["from_bus"] - 1] - angles[x["to_bus"] - 1] for x in br]
    fig = go.Figure(go.Bar(x=[f"{x['from_bus']}-{x['to_bus']}" for x in br], y=diffs, marker_color="#a78bfa"))
    fig.update_yaxes(title="θ_from − θ_to (deg)")
    return themed(fig, "Angle difference across every branch (received)")


def generator_bar(gens: list[dict]) -> go.Figure:
    labels = [f"G{g['gen']} (bus {g['bus']})" for g in gens]
    fig = go.Figure([go.Bar(x=labels, y=[g["p_mw"] for g in gens], name="P (MW)", marker_color="#a3e635"),
                     go.Bar(x=labels, y=[g["q_mvar"] for g in gens], name="Q (MVAr)", marker_color="#f472b6")])
    fig.update_layout(barmode="group")
    return themed(fig, "Generator outputs (solved power flow)")


def event_type_counts(stats: dict) -> go.Figure:
    ev = stats["ground_truth"]["events_per_type"]
    pred = stats["ai"]["predicted_attack_steps_by_type"]
    types = ["additive", "scaling", "replay", "coordinated"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Ground-truth attack events", "AI-predicted attack steps"))
    fig.add_trace(go.Bar(x=types, y=[ev.get(t, {}).get("events", 0) for t in types],
                         marker_color=[TYPE_COLORS[t] for t in types], showlegend=False), 1, 1)
    fig.add_trace(go.Bar(x=types, y=[pred.get(t, 0) for t in types],
                         marker_color=[TYPE_COLORS[t] for t in types], showlegend=False), 1, 2)
    return themed(fig, None, height=300)


def attack_timeline(events: list[dict], attack_type: str) -> go.Figure:
    ev = [e for e in events if e["attack_type"] == attack_type]
    if not ev:
        return empty(f"No {attack_type} attacks yet", height=230)
    fig = go.Figure()
    for e in ev:
        fig.add_trace(go.Scatter(x=[e["start_step"], e["end_step"]], y=[len(e["buses"])] * 2, mode="lines+markers",
                                 line=dict(color=TYPE_COLORS[attack_type], width=6), showlegend=False,
                                 hovertext=f"{e['attack_id']}<br>buses {e['buses']}<br>channels {e['channels']}"
                                           f"<br>mean severity {e['mean_severity']:.2f}", hoverinfo="text"))
    fig.update_yaxes(title="# buses")
    fig.update_xaxes(title="step")
    return themed(fig, f"{attack_type.capitalize()} attacks ({len(ev)} events, ground truth)", height=230)


def per_bus_counts(stats: dict) -> go.Figure:
    pb = stats["ground_truth"]["attacked_steps_per_bus"]
    buses = list(range(1, 31))
    fig = go.Figure(go.Bar(x=buses, y=[pb.get(str(b), 0) for b in buses], marker_color="#f87171"))
    fig.update_xaxes(dtick=1, title="bus")
    fig.update_yaxes(title="attacked steps")
    return themed(fig, "Attacked steps per bus (ground truth)", height=260)


def per_bus_f1(per_bus: list[dict], title: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=[r["bus"] for r in per_bus], y=[r["f1"] for r in per_bus], marker_color="#38bdf8",
                           hovertext=[f"actual+ {r['actual_positive']}, predicted+ {r['predicted_positive']}<br>"
                                      f"P {r['precision']:.3f} R {r['recall']:.3f}" for r in per_bus],
                           hoverinfo="text+y"))
    fig.update_xaxes(dtick=1, title="bus")
    fig.update_yaxes(range=[0, 1.02], title="F1")
    return themed(fig, title, height=280)


def confusion(matrix, labels, title) -> go.Figure:
    z = np.array(matrix)
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, colorscale="Blues", showscale=False,
                               text=z, texttemplate="%{text}"))
    fig.update_xaxes(title="predicted")
    fig.update_yaxes(title="ground truth", autorange="reversed")
    return themed(fig, title, height=320)


def bus_probabilities(ai: dict, truth_buses: list[int]) -> go.Figure:
    buses = list(range(1, 31))
    p = [ai["bus_probabilities"][str(b)] for b in buses]
    thr = [ai["bus_thresholds"][str(b)] for b in buses]
    colors = ["#ef4444" if b in ai["predicted_compromised_buses"] else "#38bdf8" for b in buses]
    fig = go.Figure(go.Bar(x=buses, y=p, marker_color=colors, name="AI bus probability"))
    fig.add_trace(go.Scatter(x=buses, y=thr, mode="markers", name="calibrated threshold",
                             marker=dict(symbol="line-ew-open", size=14, color="#fbbf24", line=dict(width=2))))
    fig.add_trace(go.Scatter(x=truth_buses, y=[1.02] * len(truth_buses), mode="markers",
                             name="ground-truth attacked (simulator)", marker=dict(symbol="star", size=11,
                                                                                  color="#f8fafc")))
    fig.update_xaxes(dtick=1, title="bus")
    fig.update_yaxes(range=[0, 1.08])
    return themed(fig, "Per-bus localisation probability (AI) vs ground truth", height=300)


def chi2_history(h: dict, threshold: float | None) -> go.Figure:
    ai, d = h.get("ai"), h["data"]
    if not ai:
        return empty("No physics history")
    fig = go.Figure()
    _attack_shapes(d["step"], d.get("attack_active", []), fig)
    fig.add_trace(go.Scatter(x=ai["step"], y=ai["physics_chi2"], name="J(x̂) chi-square statistic",
                             line=dict(color="#a3e635")))
    if threshold:
        fig.add_hline(y=threshold, line=dict(color="#f87171", dash="dash"), annotation_text="χ² threshold")
    fig.update_yaxes(type="log", title="J(x̂)")
    return themed(fig, "WLS objective (bad-data test) over time")


def recovery_chart(m: dict) -> go.Figure:
    buses = list(range(1, 31))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=buses, y=m["received_vm"], mode="markers+lines", name="Received (possibly corrupted)",
                             line=dict(color="#ef4444")))
    fig.add_trace(go.Scatter(x=buses, y=m["after"]["vm"], mode="markers+lines", name="Recovered (trusted WLS)",
                             line=dict(color="#22c55e")))
    gt = m.get("evaluation_against_ground_truth")
    if gt:
        fig.add_trace(go.Scatter(x=buses, y=gt["true_vm"], mode="markers", name="Ground truth (offline eval)",
                                 marker=dict(color="#f8fafc", symbol="x")))
    fig.update_xaxes(dtick=1, title="bus")
    fig.update_yaxes(title="V (pu)")
    return themed(fig, "Voltage before / after measurement isolation")


def loading_recovery(m: dict) -> go.Figure:
    lines = list(range(1, len(m["line_loading_received"]) + 1))
    fig = go.Figure([go.Bar(x=lines, y=m["line_loading_received"], name="Received", marker_color="#f472b6"),
                     go.Bar(x=lines, y=m["line_loading_recovered"], name="Recovered (from estimated state)",
                            marker_color="#22c55e")])
    fig.add_hline(y=100, line=dict(color="#f87171", dash="dash"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(dtick=2, title="line")
    fig.update_yaxes(title="loading %")
    return themed(fig, "Line loading before / after recovery")
