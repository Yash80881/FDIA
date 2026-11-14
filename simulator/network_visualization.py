"""Topology visualisation of the PYPOWER IEEE 30-bus case.

The bus positions are an *automatically generated graph layout*
(Kamada-Kawai on the real bus-branch graph, fixed seed) - they are NOT
geographical coordinates.  Edges are the 41 actual branches.  Arrows show the
direction of active-power flow (from the end where P enters the branch) and
line width / labels show the flow magnitude and loading.

Run ``python -m simulator.network_visualization`` to write a standalone HTML
figure of the base-case power flow to ``results/ieee30_network.html``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import plotly.graph_objects as go

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.grid_model import GridModel

LAYOUT_NOTE = "Schematic layout generated from bus-branch connectivity (not geographic)"

COLORS = {
    "normal": "#2e9e5b", "predicted": "#d62f2f", "ref": "#2563eb", "gen": "#f59e0b",
    "line": "#94a3b8", "overload": "#dc2626", "truth": "#111827",
}


def compute_layout(grid: GridModel, seed: int = 7) -> dict[int, tuple[float, float]]:
    g = nx.Graph()
    g.add_nodes_from(grid.bus_numbers)
    g.add_edges_from((b.from_bus, b.to_bus) for b in grid.branches)
    init = nx.spring_layout(g, seed=seed)
    pos = nx.kamada_kawai_layout(g, pos=init)
    return {int(b): (float(p[0]), float(p[1])) for b, p in pos.items()}


def load_or_create_layout(grid: GridModel, path: Path | None = None) -> dict[int, tuple[float, float]]:
    if path and path.exists():
        data = json.loads(path.read_text())
        layout = {int(k): tuple(v) for k, v in data.items()}
        if set(layout) == set(grid.bus_numbers):
            return layout
    layout = compute_layout(grid)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({str(k): v for k, v in layout.items()}, indent=1))
    return layout


def network_figure(grid: GridModel, layout: dict, *, p_from=None, loading=None, vm=None,
                   predicted_buses: set[int] | None = None, truth_buses: set[int] | None = None,
                   bus_prob: dict[int, float] | None = None, title: str = "IEEE 30-bus network",
                   height: int = 640, dark: bool = False) -> go.Figure:
    """Plotly figure of the network.

    ``p_from``  : active power at the from end of each branch (MW) - sets arrow direction
    ``loading`` : branch loading in % (NaN when unrated)
    ``predicted_buses`` : buses flagged by the ML localisation model (red fill)
    ``truth_buses``     : simulator ground-truth attacked buses (black ring marker)
    """
    predicted_buses = predicted_buses or set()
    truth_buses = truth_buses or set()
    fig = go.Figure()
    annotations = []
    for k, br in enumerate(grid.branches):
        x0, y0 = layout[br.from_bus]
        x1, y1 = layout[br.to_bus]
        load_k = None if loading is None else loading[k]
        overloaded = load_k is not None and np.isfinite(load_k) and load_k >= 100
        width = 1.5
        if p_from is not None:
            width = 1.0 + min(abs(p_from[k]) / 10.0, 6.0)
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(color=COLORS["overload"] if overloaded else COLORS["line"], width=width)))
        if p_from is not None:
            # arrow from the sending end (P enters the branch) toward the receiving end
            sx, sy, ex, ey = (x0, y0, x1, y1) if p_from[k] >= 0 else (x1, y1, x0, y0)
            mx, my = sx + 0.62 * (ex - sx), sy + 0.62 * (ey - sy)
            tx, ty = sx + 0.45 * (ex - sx), sy + 0.45 * (ey - sy)
            annotations.append(dict(x=mx, y=my, ax=tx, ay=ty, xref="x", yref="y", axref="x", ayref="y",
                                    showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.4,
                                    arrowcolor=COLORS["overload"] if overloaded else "#64748b", text=""))
    # invisible midpoint markers carry the branch hover text and labels
    mids_x, mids_y, texts, labels = [], [], [], []
    for k, br in enumerate(grid.branches):
        x0, y0 = layout[br.from_bus]
        x1, y1 = layout[br.to_bus]
        mids_x.append((x0 + x1) / 2)
        mids_y.append((y0 + y1) / 2)
        t = f"Line {br.index}: {br.from_bus}→{br.to_bus}"
        if br.is_transformer:
            t += " (transformer)"
        if p_from is not None:
            t += f"<br>P_from = {p_from[k]:.2f} MW"
        if loading is not None:
            t += "<br>loading = " + (f"{loading[k]:.1f} %" if np.isfinite(loading[k]) else "n/a (no rating)")
        t += f"<br>rating = {br.rate_a_mva or 'n/a'} MVA"
        texts.append(t)
        labels.append(f"L{br.index}")
    fig.add_trace(go.Scatter(x=mids_x, y=mids_y, mode="text", text=labels, hovertext=texts,
                             hoverinfo="text", textfont=dict(size=8, color="#64748b"),
                             showlegend=False, customdata=[["line", i + 1] for i in range(len(labels))]))

    xs, ys, colors, symbols, sizes, hover, line_colors, line_widths = [], [], [], [], [], [], [], []
    for b in grid.bus_numbers:
        x, y = layout[b]
        xs.append(x)
        ys.append(y)
        btype = grid.bus_types[b]
        is_gen = b in grid.generator_buses
        if b in predicted_buses:
            colors.append(COLORS["predicted"])
        else:
            colors.append(COLORS["normal"])
        symbols.append("square" if btype == "REF" else ("diamond" if is_gen else "circle"))
        sizes.append(24 if btype == "REF" else (21 if is_gen else 17))
        line_colors.append(COLORS["truth"] if b in truth_buses else
                           (COLORS["ref"] if btype == "REF" else (COLORS["gen"] if is_gen else "#ffffff")))
        line_widths.append(4 if b in truth_buses else 2.5)
        h = f"<b>Bus {b}</b> ({btype}{', generator' if is_gen else ''})"
        if vm is not None:
            h += f"<br>V = {vm[grid.bus_index[b]]:.4f} pu"
        if bus_prob is not None and b in bus_prob:
            h += f"<br>ML attack probability = {bus_prob[b]:.2f}"
        h += "<br>ML prediction: " + ("ATTACKED" if b in predicted_buses else "normal")
        h += "<br>Ground truth (simulator): " + ("attacked" if b in truth_buses else "normal")
        hover.append(h)
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=[str(b) for b in grid.bus_numbers],
        textfont=dict(color="white", size=10), hovertext=hover, hoverinfo="text", showlegend=False,
        customdata=[["bus", b] for b in grid.bus_numbers],
        marker=dict(color=colors, symbol=symbols, size=sizes, line=dict(color=line_colors, width=line_widths))))

    # legend entries
    for name, color, symbol in (("Normal bus (ML)", COLORS["normal"], "circle"),
                                ("ML-predicted compromised", COLORS["predicted"], "circle"),
                                ("Generator bus", COLORS["normal"], "diamond"),
                                ("Reference / external-grid bus", COLORS["normal"], "square")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(color=color, symbol=symbol, size=12)))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Ground-truth attacked (black ring)",
                             marker=dict(color="rgba(0,0,0,0)", size=12, line=dict(color=COLORS["truth"], width=3))))
    fg = "#e5e7eb" if dark else "#111827"
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)), height=height, annotations=annotations,
        margin=dict(l=10, r=10, t=40, b=30), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=fg), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        legend=dict(orientation="h", y=-0.02, font=dict(size=10)), clickmode="event",
    )
    fig.add_annotation(text=LAYOUT_NOTE, xref="paper", yref="paper", x=0, y=1.0, showarrow=False,
                       font=dict(size=9, color="#64748b"), xanchor="left")
    return fig


def main() -> int:
    from config.settings import load_settings, resolve_path
    from simulator.power_flow import run_ac_power_flow
    from simulator.measurement_generator import MeasurementGenerator
    cfg = load_settings()
    grid = GridModel()
    layout = load_or_create_layout(grid, resolve_path(cfg, "processed") / "network_layout.json")
    pf = run_ac_power_flow(grid.new_case())
    mg = MeasurementGenerator(grid, cfg, np.random.default_rng(0))
    _, line = mg.true_measurements(pf)
    fig = network_figure(grid, layout, p_from=pf.pf_mw, loading=line.loading, vm=pf.vm_pu,
                         title="PYPOWER IEEE 30-bus case - base-case AC power flow")
    out = resolve_path(cfg, "results") / "ieee30_network.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
