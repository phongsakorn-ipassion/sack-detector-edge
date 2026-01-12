-- ===== devices =====
CREATE TABLE IF NOT EXISTS devices (
  device_id     TEXT PRIMARY KEY,
  created_on    TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')),
  updated_on    TEXT,
  mac_address   TEXT,
  inet_last     TEXT,
  inet6_last    TEXT,
  notes         TEXT
);

CREATE INDEX IF NOT EXISTS idx_devices_device_id ON devices(device_id);
CREATE INDEX IF NOT EXISTS idx_devices_created_on ON devices(created_on);

-- ===== device_telemetry =====
CREATE TABLE IF NOT EXISTS device_telemetry (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,

  device_id     TEXT NOT NULL,
  event_id      TEXT NOT NULL UNIQUE,
  ts            TEXT NOT NULL,

  cpu_percent   REAL NOT NULL,
  cpu_temp_c    REAL,
  gpu_temp_c    REAL,

  mem_total_mb  REAL NOT NULL,
  mem_used_mb   REAL NOT NULL,
  mem_percent   REAL NOT NULL,

  disk_total_gb REAL NOT NULL,
  disk_used_gb  REAL NOT NULL,
  disk_percent  REAL NOT NULL,

  serial_id     TEXT NOT NULL,
  mac_address   TEXT,
  inet          TEXT,
  inet6         TEXT,
  ether         TEXT,

  uptime_s      INTEGER NOT NULL,

  raw_json      TEXT NOT NULL,
  created_on    TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')),

  FOREIGN KEY (device_id) REFERENCES devices(device_id)
);

CREATE INDEX IF NOT EXISTS idx_device_telemetry_device_id ON device_telemetry(device_id);
CREATE INDEX IF NOT EXISTS idx_device_telemetry_ts ON device_telemetry(ts);
CREATE INDEX IF NOT EXISTS idx_device_telemetry_device_ts ON device_telemetry(device_id, ts);

-- ===== optional DLQ =====
CREATE TABLE IF NOT EXISTS device_telemetry_dlq (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  created_on  TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')),
  reason       TEXT NOT NULL,
  topic        TEXT,
  raw_payload  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_device_telemetry_dlq_created_on
ON device_telemetry_dlq(created_on);