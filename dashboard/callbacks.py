"""Dashboard callbacks.

Each ``render_*`` function is a pure function of an :class:`APIClient` (and
selections) returning the component values, so it can be unit tested with a
fake client.  ``register_callbacks`` wires them to Dash; tab renderers run only
for the visible tab, so every refresh costs a handful of indexed API calls.

Nothing here invents data: every number comes from the API.  When the API or
the simulator is not delivering, the dashboard shows an explicit message.
"""
from __future__ import annotations

import logging
from datetime import datetime

from dash import Input, Output, State, html, dash_table, no_update, ctx
from dash.exceptions import PreventUpdate

from dashboard import charts
from dashboard.api_client import APIClient
from dashboard.layout import TABS
from dashboard.network_view import twin_figure, bus_details

N_KPI = 10
log = logging.getLogger("gridsentinel.dashboard")


def _fmt(v, fmt="{:.2f}", none="–"):
    return none if v is None else fmt.format(v)


def _badge(text, kind):
    return html.Span(text, className=f"badge badge-{kind}")


# ------------------------------------------------------------------ header
def render_header(client: APIClient, history_points: int = 300):
    health, err = client.safe("/health")
    if health is None:
        banner = html.Div(f"⚠ {err}", className="banner banner-error")
        return [_badge("API offline", "red")], banner, ["–"] * N_KPI
    stream = health["stream"]
    state = stream["state"]
    kind = {"running": "green", "stale": "amber"}.get(state, "red")
    badges = [_badge(f"Simulator: {state.upper()}", kind),
              _badge("Models loaded" if health["models_loaded"] else "Models NOT loaded",
                     "green" if health["models_loaded"] else "red"),
              _badge(f"DB {health['database']}", "green" if health["database"] == "ok" else "red")]
    mon = health["monitor"]
    if mon.get("dropped"):
        badges.append(_badge(f"dropped {mon['dropped']}", "amber"))
    banner = None
    if not stream["live"]:
        banner = html.Div(f"NOT LIVE - {stream['message']} Values below are the last stored snapshot.",
                          className="banner banner-warn")
    if not health["models_loaded"]:
        msg = f"AI models unavailable: {health.get('model_error')}"
        banner = html.Div([banner, html.Div(msg, className="banner banner-error")]) if banner else \
            html.Div(msg, className="banner banner-error")
    latest, err = client.safe("/latest")
    if latest is None:
        return badges, html.Div([banner, html.Div(err, className="banner banner-warn")]), \
            [state.upper()] + ["–"] * (N_KPI - 1)
    sysd, gt, ai, ph, comb = latest["system"], latest["ground_truth"], latest.get("ai"), latest.get("physics"), \
        latest.get("combined")
    kpis = [
        state.upper(),
        f"{latest['step']}",
        f"{latest['network']['buses_monitored']} / {latest['network']['lines_monitored']}",
        f"{sysd['total_load_p_mw']:.1f} MW",
        f"{sysd['total_load_q_mvar']:.1f} MVAr",
        f"{gt.get('total_attack_events', 0)}",
        f"{1 if gt.get('attack_active') else 0}" + (f" ({gt['attack_type']})" if gt.get("attack_active") else ""),
        _fmt(ai["attack_probability"], "{:.3f}") if ai else "n/a",
        (ai["prediction"] if ai else "n/a") + (f" · combined {comb['decision']}" if comb else ""),
        ("BAD DATA" if ph and ph.get("bad_data_detected") else ("consistent" if ph else "n/a"))
        + (f" · {comb['agreement']}" if comb else ""),
    ]
    return badges, banner, kpis


# --------------------------------------------------------------- overview
def render_overview(client: APIClient, points: int):
    h, err = client.safe("/history", groups="system,voltage", limit=points)
    if h is None:
        e = charts.empty(err)
        return e, e, e
    return charts.probability_history(h), charts.load_history(h), charts.voltage_heatmap(h)


# ----------------------------------------------------------------- twin
def render_twin(client: APIClient):
    net, err = client.safe("/network")
    if net is None:
        return charts.empty(err, 600)
    buses, _ = client.safe("/buses")
    lines, _ = client.safe("/lines")
    return twin_figure(net, buses, lines)


def render_bus_details(client: APIClient, bus_no):
    if bus_no is None:
        return bus_details(None)
    buses, err = client.safe("/buses")
    if buses is None:
        return html.Div(err, className="banner banner-warn")
    return bus_details(next((b for b in buses if b["bus"] == bus_no), None))


# ---------------------------------------------------------- measurements
def render_measurements(client: APIClient, hist_buses, hist_lines, points: int):
    buses, err = client.safe("/buses")
    if buses is None:
        e = charts.empty(err)
        return [html.Div(err, className="banner banner-warn"), None] + [e] * 13
    lines, _ = client.safe("/lines")
    net, _ = client.safe("/network")
    latest, _ = client.safe("/latest")
    h, _ = client.safe("/history", groups="system,voltage,angle,line_p,loading,current", limit=points)
    v = [b["received"]["V_pu"] for b in buses]
    vt = [b["ground_truth"]["V_pu"] for b in buses]
    ang = [b["received"]["Ang_deg"] for b in buses]
    viol = [b["bus"] for b in buses if b["voltage_violation"]]
    vwarn = html.Div(f"⚠ Voltage limit violation at bus(es) {viol}", className="banner banner-warn") if viol else \
        html.Div("All 30 bus voltages within limits.", className="ok-line")
    hb = hist_buses or []
    hl = hist_lines or []
    figs = [
        charts.bus_bar(v, "Voltage magnitude, all buses (received vs ground truth)", "V (pu)",
                       limits=[0.95, 1.05], truth=vt),
        charts.multi_line(h, [f"V_meas_bus_{b}" for b in hb], "Historical voltage profile (received)", "V (pu)",
                          [f"Bus {b}" for b in hb]) if h else charts.empty("no history"),
        charts.bus_bar(ang, "Voltage angle, all buses (received)", "angle (deg)",
                       truth=[b["ground_truth"]["Ang_deg"] for b in buses]),
        charts.angle_diff_bar(net, ang) if net else charts.empty("network unavailable"),
        charts.multi_line(h, [f"Ang_meas_bus_{b}" for b in hb], "Temporal angle variation (received)",
                          "angle (deg)", [f"Bus {b}" for b in hb]) if h else charts.empty("no history"),
        charts.bus_bar([b["received"]["P_mw"] for b in buses], "Active power injection P = Pg − Pd (received)", "MW",
                       truth=[b["ground_truth"]["P_mw"] for b in buses]),
        charts.bus_bar([b["received"]["Q_mvar"] for b in buses], "Reactive power injection Q = Qg − Qd (received)",
                       "MVAr", truth=[b["ground_truth"]["Q_mvar"] for b in buses]),
        charts.generator_bar(latest["system"]["generators"]) if latest else charts.empty("no data"),
        charts.bus_bar([b["ground_truth"]["Pd_mw"] for b in buses], "Bus loads Pd (simulator demand)", "MW"),
    ]
    if lines:
        loading = [ln["received"]["loading"] for ln in lines]
        over = [ln["line"] for ln in lines if ln["overloaded"]]
        lwarn = html.Div(f"⚠ Line overload on line(s) {over}", className="banner banner-error") if over else \
            html.Div("No line overloads.", className="ok-line")
        figs += [
            charts.line_bar(loading, "Line loading (received, % of RATE_A)", "%", ref=100),
            charts.two_end_bar([ln["received"]["P_from"] for ln in lines], [ln["received"]["P_to"] for ln in lines],
                               "P sending end", "P receiving end", "Sending vs receiving-end active power", "MW"),
            charts.two_end_bar([ln["received"]["I_from"] for ln in lines], [ln["received"]["I_to"] for ln in lines],
                               "I sending end", "I receiving end", "Sending vs receiving-end current", "A"),
            charts.multi_line(h, [f"loading_line_{k}" for k in hl] + [f"P_from_line_{k}" for k in hl],
                              "Selected lines: loading % and P_from (MW)", "",
                              [f"L{k} loading" for k in hl] + [f"L{k} P_from" for k in hl]) if h else
            charts.empty("no history"),
        ]
    else:
        lwarn = html.Div("Line data unavailable", className="banner banner-warn")
        figs += [charts.empty("no line data")] * 4
    return [vwarn, lwarn] + figs


# ---------------------------------------------------------------- attacks
def _card(title, items, cls=""):
    return html.Div([html.Div(title, className=f"panel-title {cls}"),
                     html.Table([html.Tr([html.Td(k, className="k"), html.Td(v, className="v")])
                                 for k, v in items], className="kv")], className="panel")


def render_attacks(client: APIClient):
    stats, err = client.safe("/attack-statistics")
    if stats is None:
        e = charts.empty(err)
        return html.Div(err, className="banner banner-warn"), e, e, e, e, e, e, None
    pred, perr = client.safe("/predictions")
    latest, _ = client.safe("/latest")
    gt = (latest or {}).get("ground_truth", {})
    events = stats["ground_truth"]["events"]
    cur_event = next((e for e in events if e["attack_id"] == gt.get("attack_id")), None)
    gt_items = [("Attack active", "YES" if gt.get("attack_active") else "no"),
                ("Attack type", gt.get("attack_type", "–")), ("Attack ID", gt.get("attack_id") or "–"),
                ("Start step / time", f"{cur_event['start_step']} / {cur_event['start_time'][11:19]}" if cur_event else "–"),
                ("Duration (planned)", f"{gt.get('attack_duration_steps')} steps" if gt.get("attack_active") else "–"),
                ("Attacked buses", str(gt.get("attacked_buses", []))),
                ("Channels", gt.get("attack_channels", "–")),
                ("Severity (σ units)", _fmt(gt.get("attack_severity"))),
                ("Total events", str(stats["ground_truth"]["total_events"]))]
    ai = (pred or {}).get("ai")
    ai_items = [("Detection", ai["prediction"] if ai else "n/a"),
                ("Attack probability", _fmt(ai["attack_probability"], "{:.3f}") if ai else "n/a"),
                ("Predicted type", (ai or {}).get("predicted_attack_type") or "–"),
                ("Predicted compromised buses", str(ai["predicted_compromised_buses"]) if ai else "n/a"),
                ("Estimated severity", _fmt((ai or {}).get("estimated_severity"))),
                ("Combined (AI + physics)", (pred or {}).get("combined", {}).get("decision", "–"))] if ai else \
        [("AI", perr or "unavailable")]
    current = [_card("Simulator ground truth (actual attack)", gt_items, "tag-gt"),
               _card("AI prediction", ai_items, "tag-ai")]
    tl = [charts.attack_timeline(events, t) for t in ("additive", "scaling", "replay", "coordinated")]
    table = dash_table.DataTable(
        data=[{"attack_id": e["attack_id"].split("-")[-1], "type": e["attack_type"], "start": e["start_step"],
               "end": e["end_step"], "duration": e["duration_steps"], "buses": str(e["buses"]),
               "channels": ",".join(e["channels"]), "mean severity": round(e["mean_severity"] or 0, 2),
               "start time": (e["start_time"] or "")[11:19]} for e in events[:200]],
        columns=[{"name": c, "id": c} for c in ["attack_id", "type", "start", "end", "duration", "buses", "channels",
                                               "mean severity", "start time"]],
        page_size=12, style_table={"overflowX": "auto"}, **_table_style())
    return current, charts.event_type_counts(stats), *tl, charts.per_bus_counts(stats), table


def _table_style():
    return dict(style_header={"backgroundColor": "#1e293b", "color": "#e2e8f0", "fontWeight": "600"},
                style_cell={"backgroundColor": "#0f172a", "color": "#cbd5e1", "border": "1px solid #1e293b",
                            "fontSize": 12, "padding": "4px 8px", "textAlign": "left"})


# --------------------------------------------------------------------- AI
def render_ai(client: APIClient):
    mm, err = client.safe("/model-metrics")
    pred, perr = client.safe("/predictions")
    off_kpis, live_kpis = [], []
    if mm is None:
        e = charts.empty(err)
        f1_fig = cm_type = cm_det = e
        regime = html.Div(err, className="banner banner-warn")
    else:
        det = mm["detection"]["splits"]["test"]
        loc = mm["localization"]
        sel = loc["selected_regime"]
        lt = loc["regimes"][sel]["splits"]["test"]
        typ = mm["attack_type"]["splits"]["test"]["attacked_snapshots"]
        for label, val in (("Detection accuracy", det["accuracy"]), ("Precision", det["precision"]),
                           ("Recall", det["recall"]), ("F1", det["f1"]), ("FPR", det["false_positive_rate"]),
                           ("Loc. micro-F1 (oracle)", lt["oracle_attacked_snapshots"]["micro_f1"]),
                           ("Loc. macro-F1 (oracle)", lt["oracle_attacked_snapshots"]["macro_f1"]),
                           ("Loc. micro-F1 (two-stage)", lt["two_stage_all_snapshots"]["micro_f1"]),
                           ("Type accuracy", typ["accuracy"]), ("Type macro-F1", typ["macro_f1"])):
            off_kpis.append(html.Div([html.Div(label, className="kpi-label"),
                                      html.Div(f"{val:.3f}", className="kpi-value")], className="kpi"))
        f1_fig = charts.per_bus_f1(lt["oracle_attacked_snapshots"]["per_bus"],
                                   f"Per-bus localisation F1 (test runs, regime '{sel}', oracle)")
        cm = typ["confusion_matrix"]
        cm_type = charts.confusion(cm["matrix"], cm["labels"], "Attack-type confusion matrix (test)")
        c = det["confusion_matrix"]
        cm_det = charts.confusion([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], ["NORMAL", "ATTACK"],
                                  "Detection confusion matrix (test)")
        rows = []
        for regime_name, r in loc["regimes"].items():
            for split, m in r["splits"].items():
                rows.append({"regime": regime_name + (" (selected)" if regime_name == sel else ""), "split": split,
                             "oracle micro-F1": round(m["oracle_attacked_snapshots"]["micro_f1"], 4),
                             "two-stage micro-F1": round(m["two_stage_all_snapshots"]["micro_f1"], 4)})
        regime = html.Div([html.Div("Localisation training regimes (attack-only vs all snapshots)",
                                    className="panel-title"),
                           dash_table.DataTable(data=rows, columns=[{"name": k, "id": k} for k in rows[0]] if rows else [],
                                                **_table_style())])
    if pred is None:
        live_kpis = [html.Div(perr, className="banner banner-warn")]
        busprob = charts.empty(perr or "no prediction")
    else:
        lp = pred["live_performance"]
        if lp.get("window_steps"):
            d = lp["detection"]
            for label, val in (("Live window", f"{lp['window_steps']} steps"), ("Live accuracy", f"{d['accuracy']:.3f}"),
                               ("Live F1", f"{d['f1']:.3f}"), ("Live FPR", f"{d['false_positive_rate']:.3f}"),
                               ("Live loc. micro-F1", f"{lp['localization_micro_f1']:.3f}"),
                               ("Live type acc.", _fmt(lp["attack_type_accuracy_on_detected"], "{:.3f}")),
                               ("Mean processing", f"{lp['mean_processing_ms']:.0f} ms")):
                live_kpis.append(html.Div([html.Div(label, className="kpi-label"),
                                           html.Div(val, className="kpi-value")], className="kpi"))
        busprob = charts.bus_probabilities(pred["ai"], pred["ground_truth"]["attacked_buses"])
    return off_kpis, f1_fig, cm_type, cm_det, regime, live_kpis, busprob


# ---------------------------------------------------------------- physics
def render_physics(client: APIClient, points: int):
    pv, err = client.safe("/physics-verification")
    if pv is None or pv["physics"].get("status") != "ok":
        msg = err or (pv or {}).get("physics", {}).get("message", "unavailable")
        e = charts.empty(msg)
        return html.Div(msg, className="banner banner-warn"), e, e, e, None
    ph = pv["physics"]
    buses, _ = client.safe("/buses")
    h, _ = client.safe("/history", groups="system", limit=points)
    ai = pv.get("ai_detection")
    cards = [
        _card("Physics verification (WLS)", [
            ("Converged / observable", f"{ph['converged']} / {ph['observable']}"),
            ("χ² statistic J(x̂)", f"{ph['chi2_statistic']:.1f} (threshold {ph['chi2_threshold']:.1f})"),
            ("Max normalised residual", f"{ph['max_normalized_residual']:.2f}"),
            ("Result", "BAD DATA" if ph["bad_data_detected"] else "consistent"),
            ("Suspicious buses (LNR)", str(ph.get("suspicious_buses", [])))], "tag-phys"),
        _card("AI detection", [("Prediction", ai["prediction"] if ai else "n/a"),
                               ("Probability", _fmt((ai or {}).get("attack_probability"), "{:.3f}")),
                               ("Predicted buses", str((ai or {}).get("predicted_compromised_buses", "n/a")))], "tag-ai"),
        _card("Agreement", [("AI vs physics", pv["combined"]["agreement"]),
                            ("Combined decision", pv["combined"]["decision"]),
                            ("Ground truth (simulator)", "attack" if pv["ground_truth"]["attack_active"] else "normal"),
                            ("Limitation", "stealthy a = Hc attacks evade residual tests")], "tag-gt"),
    ]
    if buses:
        est_v = charts.bus_bar([b["received"]["V_pu"] for b in buses], "Voltage: received vs WLS estimate", "V (pu)",
                               limits=[0.95, 1.05], est=ph["estimated_vm"],
                               highlight=set(ph.get("suspicious_buses", [])))
        est_a = charts.bus_bar([b["received"]["Ang_deg"] for b in buses], "Angle: received vs WLS estimate", "deg",
                               est=ph["estimated_va_deg"], highlight=set(ph.get("suspicious_buses", [])))
    else:
        est_v = est_a = charts.empty("bus data unavailable")
    chi = charts.chi2_history(h, ph["chi2_threshold"]) if h else charts.empty("no history")
    rows = [{"measurement": r["measurement"], "bus": r["bus"], "r_N": round(r["normalized_residual"], 2)}
            for r in ph.get("top_residuals", [])]
    ident = [r["measurement"] for r in ph.get("identified_bad_measurements", [])]
    table = html.Div([html.Div("Largest normalised residuals (|r_N| > 3)", className="panel-title"),
                      dash_table.DataTable(data=rows, columns=[{"name": k, "id": k} for k in ("measurement", "bus", "r_N")],
                                           page_size=10, **_table_style()) if rows else
                      html.Div("No residual above the threshold.", className="ok-line"),
                      html.Div(f"LNR-identified bad measurements: {ident or 'none'}", className="muted")])
    return cards, est_v, est_a, chi, table


# ---------------------------------------------------------------- alerts
def render_alerts(client: APIClient):
    al, err = client.safe("/alerts", limit=200)
    if al is None:
        e = charts.empty(err)
        return html.Div(err, className="banner banner-warn"), None, e, e
    rows = [{"time": (a["start_time"] or "")[11:19], "steps": f"{a['start_step']}–{a['end_step']}",
             "status": a["status"], "AI type": a["predicted_type"], "compromised buses (AI)": str(a["predicted_buses"]),
             "max probability": round(a["max_probability"] or 0, 3),
             "est. severity": round(a["est_severity"], 2) if a["est_severity"] is not None else "",
             "verification": a["physics_status"],
             "truth attack id": ",".join(x.split("-")[-1] for x in a["ground_truth_attack_ids"]) or "none (false alarm)",
             "recommended action": a["recommended_action"]} for a in al["alerts"]]
    table = dash_table.DataTable(data=rows, columns=[{"name": k, "id": k} for k in (rows[0] if rows else
                                 {"time": 0, "status": 0})], page_size=10,
                                 style_table={"overflowX": "auto"},
                                 style_data={"whiteSpace": "normal", "height": "auto"}, **_table_style()) if rows \
        else html.Div("No alerts raised yet.", className="ok-line")
    m, merr = client.safe("/mitigation")
    if m is None or m.get("status") != "ok":
        e = charts.empty(merr or (m or {}).get("message", "unavailable"))
        return table, html.Div(merr or "mitigation unavailable", className="banner banner-warn"), e, e
    recs = [html.Li([html.Span(r["priority"].upper(), className=f"prio prio-{r['priority']}"), " ", r["text"]])
            for r in m["recommendations"]]
    ev = m.get("evaluation_against_ground_truth")
    info = [html.Div(f"Step {m['step']} · combined decision: {m.get('combined_decision')} · "
                     f"excluded measurements: {m['excluded_measurements'] or 'none'}", className="muted"),
            html.Div(f"χ² before {m['before']['chi2_statistic']:.1f} → after {m['after']['chi2_statistic']:.1f} "
                     f"(threshold {m['after']['chi2_threshold']:.1f})", className="muted"),
            html.Ul(recs, className="recs"),
            html.Div(m["note"], className="muted")]
    if ev:
        info.append(html.Div(f"Offline evaluation vs ground truth: voltage RMSE received {ev['received_v_rmse']:.4f} "
                             f"→ recovered {ev['recovered_v_rmse']:.4f} pu; angle RMSE {ev['received_angle_rmse_deg']:.3f}"
                             f" → {ev['recovered_angle_rmse_deg']:.3f}°", className="section-tag tag-gt"))
    return table, info, charts.recovery_chart(m), charts.loading_recovery(m)


# ----------------------------------------------------------- registration
def register_callbacks(app, client: APIClient, cfg: dict) -> None:
    points = int(cfg["dashboard"]["history_points"])
    tab_ids = [v for v, _ in TABS]

    @app.callback([Output(f"tab-{t}", "style") for t in tab_ids], Input("tabs", "value"))
    def _tabs(value):
        return [{"display": "block"} if t == value else {"display": "none"} for t in tab_ids]

    @app.callback([Output("status-badges", "children"), Output("banner", "children")]
                  + [Output(k, "children") for k in ("k-sim", "k-step", "k-buses", "k-load-p", "k-load-q",
                                                    "k-events", "k-active", "k-prob", "k-det", "k-phys")]
                  + [Output("footer", "children")],
                  Input("tick", "n_intervals"))
    def _header(_):
        try:
            badges, banner, kpis = render_header(client, points)
        except Exception as exc:          # never freeze silently: show the error and keep refreshing
            log.exception("header refresh failed")
            badges, banner, kpis = [_badge("Dashboard error", "red")], \
                html.Div(f"Dashboard could not render the latest snapshot: {exc!r}", className="banner banner-error"), \
                ["–"] * N_KPI
        foot = f"Refreshed {datetime.now().strftime('%H:%M:%S')} · API {client.base} · values are read from the " \
               "API (no synthetic dashboard data)"
        return [badges, banner, *kpis, foot]

    def _only(tab):
        def guard(current):
            if current != tab:
                raise PreventUpdate
        return guard

    def _safe(render, *args):
        try:
            return render(*args)
        except PreventUpdate:
            raise
        except Exception:
            log.exception("%s failed", render.__name__)
            raise PreventUpdate

    @app.callback([Output("g-prob", "figure"), Output("g-load", "figure"), Output("g-vheat", "figure")],
                  Input("tick", "n_intervals"), Input("tabs", "value"))
    def _overview(_, tab):
        _only("overview")(tab)
        return _safe(render_overview, client, points)

    @app.callback(Output("g-twin", "figure"), Input("tick-tab", "n_intervals"), Input("tabs", "value"))
    def _twin(_, tab):
        _only("twin")(tab)
        return _safe(render_twin, client)

    @app.callback(Output("bus-pick", "value"), Input("g-twin", "clickData"), prevent_initial_call=True)
    def _click(click):
        if not click:
            raise PreventUpdate
        cd = click["points"][0].get("customdata")
        if isinstance(cd, list) and cd and cd[0] == "bus":
            return int(cd[1])
        raise PreventUpdate

    @app.callback(Output("bus-details", "children"), Input("bus-pick", "value"), Input("tick-tab", "n_intervals"),
                  State("tabs", "value"))
    def _details(bus_no, _, tab):
        if tab != "twin" and ctx.triggered_id == "tick-tab":
            raise PreventUpdate
        return render_bus_details(client, bus_no)

    @app.callback([Output("v-warnings", "children"), Output("line-warnings", "children")]
                  + [Output(g, "figure") for g in ("g-vbar", "g-vhist", "g-abar", "g-adiff", "g-ahist", "g-pbar",
                                                   "g-qbar", "g-gen", "g-load2", "g-loading", "g-pq-ends",
                                                   "g-current", "g-line-hist")],
                  Input("tick-tab", "n_intervals"), Input("tabs", "value"), Input("hist-buses", "value"),
                  Input("hist-lines", "value"))
    def _meas(_, tab, hb, hl):
        _only("measurements")(tab)
        return _safe(render_measurements, client, hb, hl, points)

    @app.callback([Output("attack-current", "children"), Output("g-type-counts", "figure")]
                  + [Output(f"g-tl-{t}", "figure") for t in ("additive", "scaling", "replay", "coordinated")]
                  + [Output("g-perbus-gt", "figure"), Output("event-table", "children")],
                  Input("tick-tab", "n_intervals"), Input("tabs", "value"))
    def _attacks(_, tab):
        _only("attacks")(tab)
        return _safe(render_attacks, client)

    @app.callback([Output("offline-kpis", "children"), Output("g-perbus-f1", "figure"), Output("g-cm-type", "figure"),
                   Output("g-cm-det", "figure"), Output("regime-table", "children"), Output("live-kpis", "children"),
                   Output("g-busprob", "figure")],
                  Input("tick-tab", "n_intervals"), Input("tabs", "value"))
    def _ai(_, tab):
        _only("ai")(tab)
        return _safe(render_ai, client)

    @app.callback([Output("physics-cards", "children"), Output("g-est-v", "figure"), Output("g-est-a", "figure"),
                   Output("g-chi2", "figure"), Output("residual-table", "children")],
                  Input("tick-tab", "n_intervals"), Input("tabs", "value"))
    def _physics(_, tab):
        _only("physics")(tab)
        return _safe(render_physics, client, points)

    @app.callback([Output("alert-table", "children"), Output("mitigation-panel", "children"),
                   Output("g-recovery", "figure"), Output("g-load-recovery", "figure")],
                  Input("tick-tab", "n_intervals"), Input("tabs", "value"))
    def _alerts(_, tab):
        _only("alerts")(tab)
        return _safe(render_alerts, client)
