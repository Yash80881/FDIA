// GridSentinel AI - OPTIONAL ESP32 telemetry publisher (extra-low-voltage bench only)
//
// Hardware: ESP32 + INA219 (I2C: SDA=21, SCL=22) measuring a 5-12 V DC demo circuit.
// SAFETY: never connect this to mains voltage or to real grid equipment.
// Libraries: WiFi, PubSubClient, Adafruit_INA219.
// Publishes JSON once per second to gridsentinel/telemetry/<device_id>, matching
// hardware/mqtt_gateway.py:parse_telemetry().

#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_INA219.h>

const char* WIFI_SSID = "YOUR_WIFI";
const char* WIFI_PASS = "YOUR_PASSWORD";
const char* MQTT_HOST = "192.168.1.10";   // PC running the MQTT broker (e.g. Mosquitto)
const int   MQTT_PORT = 1883;
const char* DEVICE_ID = "esp32-01";
const int   BUS_ID    = 5;                // which simulated IEEE-30 bus this bench represents

WiFiClient net;
PubSubClient mqtt(net);
Adafruit_INA219 ina219;

void connectAll() {
  while (WiFi.status() != WL_CONNECTED) { WiFi.begin(WIFI_SSID, WIFI_PASS); delay(2000); }
  while (!mqtt.connected()) { mqtt.connect(DEVICE_ID); delay(1000); }
}

void setup() {
  Serial.begin(115200);
  ina219.begin();
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  connectAll();
}

void loop() {
  if (!mqtt.connected()) connectAll();
  mqtt.loop();
  float v = ina219.getBusVoltage_V();
  float i = ina219.getCurrent_mA() / 1000.0;
  char payload[200];
  snprintf(payload, sizeof(payload),
           "{\"device_id\":\"%s\",\"bus\":%d,\"voltage_v\":%.3f,\"current_a\":%.4f,\"power_w\":%.3f}",
           DEVICE_ID, BUS_ID, v, i, v * i);
  char topic[80];
  snprintf(topic, sizeof(topic), "gridsentinel/telemetry/%s", DEVICE_ID);
  mqtt.publish(topic, payload);
  delay(1000);
}
