
const p = {
    "device_id": "mac-4a968d44ca0d",
    "event_id": "dbebd025-d5c1-420a-b1ce-0557e6deeb46",
    "ts": "2026-01-12T16:24:44+00:00",
    "cpu": {
        "percent": 0,
        "temp_c": null
    },
    "gpu": {
        "temp_c": null
    },
    "memory": {
        "total_mb": 7936.4,
        "used_mb": 695,
        "percent": 8.8
    },
    "disk": {
        "total_gb": 452.1,
        "used_gb": 47.1,
        "percent": 11
    },
    "network": {
        "serial_id": "mac-4a968d44ca0d",
        "mac_address": "4a:96:8d:44:ca:0d",
        "inet": "172.18.0.5",
        "inet6": null,
        "ether": "4a:96:8d:44:ca:0d"
    },
    "uptime_s": 9479
};

const msg = { payload: p };

function validate(msg) {
    const p = msg.payload;

    // Basic shape checks
    if (!p || typeof p !== 'object') {
        msg.dlq_reason = 'payload_not_object';
        return [null, msg];
    }

    const requiredTop = ['device_id', 'event_id', 'ts', 'cpu', 'memory', 'disk', 'network', 'uptime_s'];
    for (const k of requiredTop) {
        if (p[k] === undefined || p[k] === null) {
            msg.dlq_reason = `missing_${k}`;
            return [null, msg];
        }
    }

    if (!p.cpu || p.cpu.percent === undefined) {
        msg.dlq_reason = 'missing_cpu.percent';
        return [null, msg];
    }

    // Choose a stable device key for routing
    const serial = p.network?.serial_id;
    const mac = p.network?.mac_address;
    msg.device_key = serial || mac || 'unknown';

    // Normalize numeric types
    p.cpu.percent = Number(p.cpu.percent);
    if (p.cpu.temp_c !== null && p.cpu.temp_c !== undefined) p.cpu.temp_c = Number(p.cpu.temp_c);
    if (p.gpu?.temp_c !== null && p.gpu?.temp_c !== undefined) p.gpu.temp_c = Number(p.gpu.temp_c);

    p.memory.total_mb = Number(p.memory.total_mb);
    p.memory.used_mb = Number(p.memory.used_mb);
    p.memory.percent = Number(p.memory.percent);

    p.disk.total_gb = Number(p.disk.total_gb);
    p.disk.used_gb = Number(p.disk.used_gb);
    p.disk.percent = Number(p.disk.percent);

    p.uptime_s = Number(p.uptime_s);

    msg.payload = p;
    return [msg, null];
}

const result = validate(msg);
if (result[0]) {
    console.log("VALIDATION PASSED");
    console.log("Payload:", JSON.stringify(result[0].payload, null, 2));
} else {
    console.log("VALIDATION FAILED");
    console.log("Reason:", result[1].dlq_reason);
}
