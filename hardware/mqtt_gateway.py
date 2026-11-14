"""OPTIONAL hardware telemetry gateway (MQTT -> SQLite ``hardware_telemetry``).

This module is completely separate from the PYPOWER simulation: the platform
runs without it and nothing in the simulator, ML or API depends on it.  It is
meant for a *low-voltage, isolated* demonstration bench (e.g. an ESP32 with an
INA219/INA226 current-voltage sensor on a 5-12 V DC circuit, or a Raspberry
Pi).  NEVER connect experimental hardware to mains voltage or to operational
grid equipment.

Telemetry message (JSON) published by a device:
    {"device_id": "esp32-01", "ts": 1731500000.0, "bus": 5,
     "voltage_v": 11.94, "current_a": 0.231, "power_w": 2.76}

Run:  pip install paho-mqtt   then   python -m hardware.mqtt_gateway
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import load_settings, resolve_path

log = logging.getLogger("gridsentinel.hardware")

REQUIRED = {"device_id": str, "voltage_v": (int, float), "current_a": (int, float)}
LIMITS = {"voltage_v": (0.0, 48.0), "current_a": (-10.0, 10.0)}   # extra-low-voltage bench only


class TelemetryError(ValueError):
    pass


def parse_telemetry(payload: bytes | str) -> dict:
    """Validate one telemetry message; raises TelemetryError on bad input."""
    try:
        msg = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise TelemetryError(f"not JSON: {exc}") from exc
    if not isinstance(msg, dict):
        raise TelemetryError("telemetry must be a JSON object")
    for key, typ in REQUIRED.items():
        if key not in msg:
            raise TelemetryError(f"missing field '{key}'")
        if not isinstance(msg[key], typ) or isinstance(msg[key], bool):
            raise TelemetryError(f"field '{key}' has wrong type")
    for key, (lo, hi) in LIMITS.items():
        if not lo <= float(msg[key]) <= hi:
            raise TelemetryError(f"{key}={msg[key]} outside the safe bench range [{lo}, {hi}]")
    if "bus" in msg and not (isinstance(msg["bus"], int) and 1 <= msg["bus"] <= 30):
        raise TelemetryError("bus must be an integer 1..30")
    msg.setdefault("power_w", float(msg["voltage_v"]) * float(msg["current_a"]))
    msg.setdefault("ts", time.time())
    return msg


class TelemetryStore:
    def __init__(self, cfg: dict):
        from storage.database import Database
        self.db = Database(resolve_path(cfg, "database"))

    def save(self, topic: str, msg: dict) -> None:
        with self.db.transaction() as conn:
            conn.execute("INSERT INTO hardware_telemetry(received_at, device_id, topic, payload) VALUES(?,?,?,?)",
                         (time.time(), msg["device_id"], topic, json.dumps(msg)))


def main() -> int:
    cfg = load_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("paho-mqtt is not installed (optional). Install with: pip install paho-mqtt")
        return 1
    hw = cfg["hardware"]
    store = TelemetryStore(cfg)

    def on_message(_client, _userdata, message):
        try:
            store.save(message.topic, parse_telemetry(message.payload))
        except TelemetryError as exc:
            log.warning("rejected telemetry on %s: %s", message.topic, exc)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2) if hasattr(mqtt, "CallbackAPIVersion") else mqtt.Client()
    client.on_message = on_message
    client.connect(hw["mqtt_host"], int(hw["mqtt_port"]))
    client.subscribe(hw["topic"])
    log.info("Listening on %s:%s topic %s (Ctrl+C to stop)", hw["mqtt_host"], hw["mqtt_port"], hw["topic"])
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()
        store.db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
