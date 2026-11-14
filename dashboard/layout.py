"""Dashboard layout (control-centre style)."""
from __future__ import annotations

from dash import dcc, html

TABS = [("overview", "Overview"), ("twin", "Digital twin"), ("measurements", "Measurements"),
        ("attacks", "Attack monitoring"), ("ai", "AI analytics"), ("physics", "Physics verification"),
        ("alerts", "Alerts & mitigation")]


def kpi(id_: str, label: str) -> html.Div:
    return html.Div([html.Div(label, className="kpi-label"), html.Div("–", id=id_, className="kpi-value")],
                    className="kpi")


def graph(id_: str) -> dcc.Graph:
    return dcc.Graph(id=id_, config={"displaylogo": False}, className="graph")


def build_layout(refresh_ms: int, tab_refresh_ms: int = 5000) -> html.Div:
    return html.Div([
        # one interval per callback group; each is paused while its callback runs (see callbacks.py),
        # so a slow refresh is never discarded and replaced by the next one
        dcc.Interval(id="tick-head", interval=refresh_ms, n_intervals=0),
        dcc.Interval(id="tick", interval=refresh_ms, n_intervals=0),
        *[dcc.Interval(id=f"tick-{t}", interval=tab_refresh_ms, n_intervals=0)
          for t in ("twin", "meas", "attacks", "ai", "physics", "alerts")],
        dcc.Store(id="selected-bus", data=None),
        html.Div([
            html.Div([html.Span("GridSentinel AI", className="brand"),
                      html.Span("FDIA detection · localisation · mitigation · PYPOWER IEEE 30-bus",
                                className="subtitle")]),
            html.Div(id="status-badges", className="badges"),
        ], className="header"),
        html.Div(id="banner"),
        html.Div([
            kpi("k-sim", "Simulation"), kpi("k-step", "Latest step"), kpi("k-buses", "Buses / lines monitored"),
            kpi("k-load-p", "Total load P"), kpi("k-load-q", "Total load Q"), kpi("k-events", "Attack events (truth)"),
            kpi("k-active", "Active attacks (truth)"), kpi("k-prob", "AI attack probability"),
            kpi("k-det", "AI detection"), kpi("k-phys", "Physics verification"),
        ], className="kpi-row"),
        dcc.Tabs(id="tabs", value="overview", className="tabs",
                 children=[dcc.Tab(label=l, value=v, className="tab", selected_className="tab-sel")
                           for v, l in TABS]),
        html.Div(id="tab-overview", children=[
            html.Div([graph("g-prob"), graph("g-load")], className="grid2"),
            graph("g-vheat"),
        ]),
        html.Div(id="tab-twin", style={"display": "none"}, children=[
            html.Div([
                html.Div(graph("g-twin"), className="twin-graph"),
                html.Div([
                    html.Div("Bus inspector", className="panel-title"),
                    dcc.Dropdown(id="bus-pick", options=[{"label": f"Bus {b}", "value": b} for b in range(1, 31)],
                                 placeholder="select a bus", className="dropdown"),
                    html.Div(id="bus-details"),
                ], className="panel side"),
            ], className="twin"),
        ]),
        html.Div(id="tab-measurements", style={"display": "none"}, children=[
            html.Div([
                html.Div([html.Span("History buses", className="lbl"),
                          dcc.Dropdown(id="hist-buses", multi=True, value=[1, 5, 10, 30],
                                       options=[{"label": f"Bus {b}", "value": b} for b in range(1, 31)])],
                         className="picker"),
                html.Div([html.Span("History lines", className="lbl"),
                          dcc.Dropdown(id="hist-lines", multi=True, value=[1, 10, 36],
                                       options=[{"label": f"Line {k}", "value": k} for k in range(1, 42)])],
                         className="picker"),
            ], className="pickers"),
            html.H4("Voltage monitoring", className="section"),
            html.Div(id="v-warnings"),
            html.Div([graph("g-vbar"), graph("g-vhist")], className="grid2"),
            html.H4("Phase-angle monitoring", className="section"),
            html.Div([graph("g-abar"), graph("g-adiff")], className="grid2"),
            graph("g-ahist"),
            html.H4("Active and reactive power", className="section"),
            html.Div([graph("g-pbar"), graph("g-qbar")], className="grid2"),
            html.Div([graph("g-gen"), graph("g-load2")], className="grid2"),
            html.H4("Transmission-line monitoring", className="section"),
            html.Div(id="line-warnings"),
            html.Div([graph("g-loading"), graph("g-pq-ends")], className="grid2"),
            html.Div([graph("g-current"), graph("g-line-hist")], className="grid2"),
        ]),
        html.Div(id="tab-attacks", style={"display": "none"}, children=[
            html.Div(id="attack-current", className="grid2 cards"),
            graph("g-type-counts"),
            html.Div([graph("g-tl-additive"), graph("g-tl-scaling")], className="grid2"),
            html.Div([graph("g-tl-replay"), graph("g-tl-coordinated")], className="grid2"),
            graph("g-perbus-gt"),
            html.Div("Attack event history (simulator ground truth)", className="panel-title"),
            html.Div(id="event-table"),
        ]),
        html.Div(id="tab-ai", style={"display": "none"}, children=[
            html.Div("OFFLINE evaluation - independent test runs (from python cli.py train)", className="section-tag tag-off"),
            html.Div(id="offline-kpis", className="kpi-row small"),
            html.Div([graph("g-perbus-f1"), graph("g-cm-type")], className="grid2"),
            html.Div([graph("g-cm-det"), html.Div(id="regime-table", className="panel")], className="grid2"),
            html.Div("LIVE - rolling comparison of current predictions with simulator ground truth",
                     className="section-tag tag-live"),
            html.Div(id="live-kpis", className="kpi-row small"),
            graph("g-busprob"),
        ]),
        html.Div(id="tab-physics", style={"display": "none"}, children=[
            html.Div(id="physics-cards", className="grid3 cards"),
            html.Div([graph("g-est-v"), graph("g-est-a")], className="grid2"),
            html.Div([graph("g-chi2"), html.Div(id="residual-table", className="panel")], className="grid2"),
        ]),
        html.Div(id="tab-alerts", style={"display": "none"}, children=[
            html.Div("Live alert & event history", className="panel-title"),
            html.Div(id="alert-table"),
            html.Div("Operator-supervised mitigation for the latest snapshot (recommendations only)",
                     className="panel-title"),
            html.Div(id="mitigation-panel", className="panel"),
            html.Div([graph("g-recovery"), graph("g-load-recovery")], className="grid2"),
        ]),
        html.Div(id="footer", className="footer"),
    ], className="app")
