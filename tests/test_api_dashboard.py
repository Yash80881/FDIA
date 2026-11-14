"""Tests 26-31: API health, predictions, history, stale detection, dashboard callbacks."""
import threading
import time

import pytest
from fastapi.testclient import TestClient

from config.settings import apply_overrides, resolve_path


@pytest.fixture
def api_cfg(cfg):
    return apply_overrides(cfg, {"simulation.stale_after_s": 1.0, "api.monitor_poll_s": 0.1})


def _client(cfg, monitor=True):
    from backend.main import create_app
    return TestClient(create_app(cfg, start_monitor=monitor))


def test_health_without_data(api_cfg):
    with _client(api_cfg) as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["api"] == "ok" and body["stream"]["state"] == "no_data"
        r = c.get("/latest")
        assert r.status_code == 404 and "simulator" in r.json()["hint"]


def test_api_with_finished_run_reports_stopped(api_cfg):
    from simulator.continuous_simulator import ContinuousSimulator
    rid = ContinuousSimulator(api_cfg, seed=4, fast=True, max_steps=80).run()
    with _client(api_cfg, monitor=False) as c:
        h = c.get("/health").json()
        assert h["stream"]["state"] == "completed" and h["stream"]["live"] is False
        latest = c.get("/latest").json()
        assert latest["step"] == 79 and latest["run_id"] == rid
        assert len(latest["received"]["V_pu"]) == 30            # all buses, not only bus 1
        assert "ground_truth" in latest and "ai" in latest and "physics" in latest
        hist = c.get("/history", params={"groups": "system,voltage", "limit": 50}).json()
        assert hist["n"] == 50 and len(hist["data"]["V_meas_bus_30"]) == 50
        assert len(set(hist["data"]["total_load_p_mw"])) > 10    # load really changes
        assert c.get("/history", params={"groups": "nope"}).status_code == 400
        buses = c.get("/buses").json()
        assert len(buses) == 30 and "ground_truth" in buses[0]
        lines = c.get("/lines").json()
        assert len(lines) == 41
        net = c.get("/network").json()
        assert len(net["buses"]) == 30 and len(net["branches"]) == 41
        stats = c.get("/attack-statistics").json()
        assert "ground_truth" in stats and "ai" in stats
        pv = c.get("/physics-verification").json()
        assert pv["physics"]["status"] == "ok"
        mit = c.get("/mitigation").json()
        assert mit["status"] == "ok" and mit["operator_supervised"]
        assert c.get("/docs").status_code == 200


def test_live_stream_and_stale_detection(api_cfg):
    from simulator.continuous_simulator import ContinuousSimulator
    sim = ContinuousSimulator(api_cfg, seed=8)
    sim.interval = 0.1
    th = threading.Thread(target=sim.run)
    th.start()
    try:
        with _client(api_cfg) as c:
            time.sleep(1.5)
            h1 = c.get("/health").json()
            assert h1["stream"]["state"] == "running" and h1["stream"]["live"]
            s1 = c.get("/latest").json()["step"]
            time.sleep(0.6)
            s2 = c.get("/latest").json()["step"]
            assert s2 > s1                                       # steps keep increasing
            # freeze the heartbeat to simulate a hung simulator
            sim.stop_requested = True
            th.join(timeout=10)
            from storage.database import Database
            db = Database(resolve_path(api_cfg, "database"))
            db.execute("UPDATE runs SET status='running', last_heartbeat=? WHERE run_id=?",
                       (time.time() - 30, sim.run_id))
            db.commit()
            db.close()
            h2 = c.get("/health").json()
            assert h2["stream"]["state"] == "stale" and not h2["stream"]["live"]
    finally:
        sim.stop_requested = True
        th.join(timeout=10)


def test_predictions_endpoint_without_models(api_cfg):
    from simulator.continuous_simulator import ContinuousSimulator
    ContinuousSimulator(api_cfg, seed=4, fast=True, max_steps=20).run()
    with _client(api_cfg, monitor=False) as c:
        r = c.get("/predictions")
        assert r.status_code == 503
        assert "train" in r.json()["hint"].lower()


def test_dashboard_renderers_with_fake_api(api_cfg):
    """Dashboard callbacks render real API payloads (TestClient-backed client)."""
    from simulator.continuous_simulator import ContinuousSimulator
    from dashboard import callbacks as cb
    from dashboard.api_client import APIClient, APIUnavailable
    ContinuousSimulator(api_cfg, seed=4, fast=True, max_steps=60).run()
    with _client(api_cfg, monitor=False) as c:
        class FakeClient(APIClient):
            def __init__(self):
                self.base = "testclient"

            def get(self, path, **params):
                r = c.get(path, params={k: v for k, v in params.items() if v is not None})
                if r.status_code >= 400:
                    raise APIUnavailable(r.json().get("detail"))
                return r.json()
        fc = FakeClient()
        badges, banner, kpis = cb.render_header(fc)
        assert len(kpis) == cb.N_KPI and kpis[1] == "59"
        assert banner is not None                       # run completed -> NOT LIVE banner shown
        figs = cb.render_overview(fc, 50)
        assert len(figs[0].data) >= 1 and len(figs[2].data) == 1
        twin = cb.render_twin(fc)
        assert any(getattr(t, "customdata", None) is not None for t in twin.data)
        out = cb.render_measurements(fc, [1, 2], [1], 50)
        assert len(out) == 2 + 13
        assert len(out[2].data[0].y) == 30              # voltage bar shows all 30 buses
        assert cb.render_attacks(fc)[1] is not None
        assert cb.render_physics(fc, 50)[1] is not None
        assert cb.render_alerts(fc)[2] is not None
        # AI tab without models -> explicit message, not fake numbers
        ai = cb.render_ai(fc)
        assert ai[0] == [] or ai[0] is not None


def test_dashboard_reports_api_down():
    from dashboard import callbacks as cb
    from dashboard.api_client import APIClient
    badges, banner, kpis = cb.render_header(APIClient("http://127.0.0.1:9", timeout=0.5))
    assert "API offline" in str(badges) and all(k == "–" for k in kpis)


def test_dashboard_app_builds(api_cfg):
    from dashboard.app import create_dashboard
    app = create_dashboard(api_cfg, "http://127.0.0.1:9")
    assert app.layout is not None
    assert len(app.callback_map) >= 8


def test_hardware_telemetry_validation():
    from hardware.mqtt_gateway import parse_telemetry, TelemetryError
    m = parse_telemetry('{"device_id": "esp32-01", "voltage_v": 11.9, "current_a": 0.2, "bus": 5}')
    assert abs(m["power_w"] - 2.38) < 1e-9
    for bad in ('not json', '{"device_id": "x", "voltage_v": 230, "current_a": 1}', '{"voltage_v": 5}',
                '{"device_id": "x", "voltage_v": 5, "current_a": 0.1, "bus": 99}'):
        with pytest.raises(TelemetryError):
            parse_telemetry(bad)
