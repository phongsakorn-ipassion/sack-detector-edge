#!/usr/bin/env python3
import json
import os
import platform
import socket
import subprocess
import time
import uuid
from datetime import datetime, timezone

import psutil
import paho.mqtt.client as mqtt


# ---------------------------
# Hardware / network helpers
# ---------------------------

def get_pi_serial():
    """Raspberry Pi hardware serial (best case)."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("Serial"):
                    serial = line.split(":")[1].strip()
                    if serial and serial != "0000000000000000":
                        return serial
    except Exception:
        pass
    return None


def get_linux_machine_id():
    """Linux machine-id (works on most Linux hosts & containers)."""
    try:
        with open("/etc/machine-id", "r") as f:
            mid = f.read().strip()
            return mid if mid else None
    except Exception:
        return None


def get_mac_address():
    """Stable MAC-based ID (fallback)."""
    try:
        mac = uuid.getnode()
        # If this is a random MAC, uuid sets multicast bit
        if (mac >> 40) % 2 == 0:
            return f"mac-{mac:012x}"
    except Exception:
        pass
    return None


def get_hostname_id():
    """Last-resort identity."""
    try:
        return f"host-{socket.gethostname()}"
    except Exception:
        return None


def resolve_device_id():
    """
    Cross-platform device identity resolver.
    device_id == serial_id by design.
    """
    return (
        get_pi_serial()
        or get_linux_machine_id()
        or get_mac_address()
        or get_hostname_id()
    )


def get_default_interface():
    """Try to find the interface used for default route."""
    try:
        out = subprocess.check_output(["ip", "route", "show", "default"], text=True).strip()
        # Example: "default via 192.168.1.1 dev wlan0 proto dhcp src 192.168.1.50 metric 600"
        parts = out.split()
        if "dev" in parts:
            return parts[parts.index("dev") + 1]
    except Exception:
        pass
    return None


def get_interface_addrs(ifname):
    """
    Return inet/inet6/mac for a specific interface name.
    """
    inet = None
    inet6 = None
    mac = None

    try:
        addrs = psutil.net_if_addrs().get(ifname, [])
        for a in addrs:
            # AF_LINK on Linux is usually psutil.AF_LINK or socket.AF_PACKET
            if getattr(socket, "AF_PACKET", None) == a.family or str(a.family).endswith("AF_LINK"):
                mac = a.address
            elif a.family == socket.AF_INET:
                inet = a.address
            elif a.family == socket.AF_INET6:
                inet6 = a.address.split("%")[0]  # strip scope if present
    except Exception:
        pass

    return inet, inet6, mac


def get_primary_network_info():
    """
    Build the 'network' object exactly as requested:
    {
      serial_id, mac_address, inet, inet6, ether
    }
    """
    serial_id = resolve_device_id()
    ifname = get_default_interface()

    inet = inet6 = mac = None

    if ifname:
        inet, inet6, mac = get_interface_addrs(ifname)

    # Fallback: pick the first interface that has an IPv4
    if inet is None:
        for name, addrs in psutil.net_if_addrs().items():
            tmp_inet, tmp_inet6, tmp_mac = get_interface_addrs(name)
            if tmp_inet:
                inet, inet6, mac = tmp_inet, tmp_inet6, tmp_mac
                break

    return {
        "serial_id": serial_id,
        "mac_address": mac,
        "inet": inet,
        "inet6": inet6,
        "ether": mac  # commonly same as MAC; kept for your structure
    }


# ---------------------------
# System metric helpers
# ---------------------------

def cpu_percent_sampled():
    """
    Reliable CPU percent. Using a small interval avoids 0% outputs.
    """
    return float(psutil.cpu_percent(interval=0.5))

def get_cpu_temp_c():
    """
    Cross-platform best effort:
    - Raspberry Pi / many Linux: thermal_zone0
    - macOS: usually unavailable without 3rd-party tools → None
    """
    # Linux / Pi path
    try:
        path = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(path):
            with open(path) as f:
                return round(int(f.read().strip()) / 1000.0, 1)
    except Exception:
        pass

    # Optional: if user installs 'osx-cpu-temp' on macOS, use it (no harm if missing)
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["osx-cpu-temp"], text=True).strip()
            # e.g. "58.1°C"
            out = out.replace("°C", "").replace("°", "").strip()
            return float(out)
        except Exception:
            return None

    return None


def get_gpu_temp_c():
    """
    Best effort:
    - Raspberry Pi: vcgencmd measure_temp
    - Others: None
    """
    # Pi tool
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], stderr=subprocess.DEVNULL, text=True).strip()
        # temp=54.2'C
        return float(out.split("=")[1].split("'")[0])
    except Exception:
        return None

def get_uptime_s():
    return int(time.time() - psutil.boot_time())


def collect_payload():
    # CPU
    cpu_percent = psutil.cpu_percent(interval=0.2)

    # RAM
    mem = psutil.virtual_memory()

    # Disk (root)
    disk = psutil.disk_usage("/")

    # Network identity/address info
    net_info = get_primary_network_info()

    payload = {
        "device_id": resolve_device_id(),
        "event_id": str(uuid.uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cpu": {
            "percent": cpu_percent_sampled(),
            "temp_c": get_cpu_temp_c()
        },
        "gpu": {
            "temp_c": get_gpu_temp_c()
        },
        "memory": {
            "total_mb": round(mem.total / 1024 / 1024, 1),
            "used_mb": round(mem.used / 1024 / 1024, 1),
            "percent": float(mem.percent)
        },
        "disk": {
            "total_gb": round(disk.total / 1024 / 1024 / 1024, 1),
            "used_gb": round(disk.used / 1024 / 1024 / 1024, 1),
            "percent": float(disk.percent)
        },
        "network": net_info,
        "uptime_s": get_uptime_s()
    }
    return payload


# ---------------------------
# MQTT publishing loop
# ---------------------------


def mqtt_connect():
    # Connect to local MQTT broker
    host = os.getenv("MQTT_HOST", "localhost")
    port = int(os.getenv("MQTT_PORT", "1883"))
    user = os.getenv("MQTT_USER")
    pwd  = os.getenv("MQTT_PASS")

    topic = "device/telemetry"
    # Use fallback if get_primary_network_info fails or returns None for serial
    net_info = get_primary_network_info()
    client_id = net_info.get("serial_id")
    if not client_id:
        client_id = f"device-{uuid.uuid4().hex[:8]}"

    # Use CallbackAPIVersion.VERSION2 if available
    try:
        client = mqtt.Client(client_id=client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client(client_id=client_id)

    if user:
        client.username_pw_set(user, pwd)

    # Last Will: set device status to offline if disconnected unexpectedly
    status_topic = "device/status" 
    client.will_set(status_topic, payload=json.dumps({"status": "offline", "id": client_id}), qos=1, retain=True)

    print(f"📡 Connecting to MQTT at {host}:{port} as {client_id}...")
    
    # Retry logic for initial connection
    while True:
        try:
            client.connect(host, port, keepalive=30)
            break
        except Exception as e:
            print(f"⚠️ Connection failed ({e}), retrying in 5s...")
            time.sleep(5)

    client.loop_start()

    # Announce online status
    client.publish(status_topic, json.dumps({"status": "online", "id": client_id}), qos=1, retain=True)

    return client, topic

def main():
    interval = 5
    client, topic = mqtt_connect()
    
    # Get host again for printing, strictly for display
    host_disp = os.getenv('MQTT_HOST', 'localhost')
    port_disp = os.getenv('MQTT_PORT', '1883')

    print(f"[device_system_stats] Publishing every {interval}s → mqtt://{host_disp}:{port_disp}/{topic}")

    try:
        while True:
            payload = collect_payload()
            client.publish(topic, json.dumps(payload), qos=1, retain=False)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            # Publish offline status before clean exit
            client.publish("device/status", json.dumps({"status": "offline", "id": client._client_id.decode('utf-8') if isinstance(client._client_id, bytes) else client._client_id}), qos=1, retain=True)
            client.loop_stop()
            client.disconnect()
        except Exception:
            pass
        print("👋 Exited.")

if __name__ == "__main__":
    main()