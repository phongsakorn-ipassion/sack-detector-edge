# Device Telemetry Service

This service runs as a lightweight container to collect system metrics (CPU, Memory, Disk, Network, Temperature) from the host device and publish them to the MQTT broker.

## Features

- **CPU Usage & Temperature**: Monitors load and thermal status (requires volume mount).
- **Memory & Disk**: Tracks RAM and Disk usage.
- **Network Info**: Reports IP and MAC address.
- **MQTT Publishing**: Sends JSON payloads to `device/telemetry` topic every 5 seconds.

## Configuration

The service is configured via environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MQTT_HOST` | `mqtt` | Hostname of the MQTT broker. |
| `MQTT_PORT` | `1883` | Port of the MQTT broker. |
| `MQTT_USER` | - | MQTT Username (optional). |
| `MQTT_PASS` | - | MQTT Password (optional). |

## Running Locally

To run the script directly on the host (e.g. for testing or without Docker):

1.  **Setup**: Run `bash ../scripts/setup.sh` (installs dependencies in `.venv`).
2.  **Activate**: `source ../.venv/bin/activate`
3.  **Run**: `python device_system_stats.py`

*Note: Environment variables (`MQTT_HOST`) must be set manually if not using defaults.*

## Docker Compose

Ensure the following volume is mounted to allow access to thermal sensors on Raspberry Pi / Linux:

```yaml
volumes:
  - /sys/class/thermal:/sys/class/thermal:ro
```

## Payload Example

Topic: `device/telemetry`

```json
{
  "device_id": "10000000abcde",
  "ts": "2026-01-12T12:00:00",
  "cpu": { "percent": 15.2, "temp_c": 45.0 },
  "memory": { "percent": 30.5 },
  "disk": { "percent": 45.0 },
  "uptime_s": 3600
}
```
