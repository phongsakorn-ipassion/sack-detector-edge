# Sack Detector Edge - Usage Guide

This guide explains how to run the `detector.py` and `detector_hailo.py` scripts for object detection and counting.

## Project Structure

```text
.
├── detector.py         # Core detection engine (YOLO + MQTT + Camera)
├── detector_hailo.py   # Hailo-8L inference path (HEF + MQTT + Camera)
├── Dockerfile          # ARM64/Pi-optimized Docker image build
├── models/             # YOLO weights / ONNX / HEF models
├── records/            # Saved video logs (mp4) + count logs (txt)
├── requirements.txt    # Shared Python dependencies (Opencv, YOLO)
└── requirements-pi.txt # Raspberry Pi specific drivers (Picamera2)
```

## 1. Setup

Run the setup script to create a virtual environment (`.venv`) and install dependencies:
```bash
# From project root
bash scripts/setup.sh
```

## 2. Basic Usage

**Important:** Always activate the virtual environment before running:
```bash
source ../.venv/bin/activate
```

### On Desktop (Testing)
Run using a USB webcam (0):
```bash
# Web Camera
python detector.py --source usb0

# Video File
python detector.py --source /path/to/video.mp4 --save
```

### On Raspberry Pi (Production)
Run using the native camera module (`picamera2`) and disable GUI for performance:
```bash
python detector.py --source picamera0 --headless --save
```

### Hailo-8L (Raspberry Pi 5 + Hailo-8L)
```bash
python detector_hailo.py --hef models/detection.hef --source picamera0 --headless
```

## 3. Arguments Reference (detector.py)

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--source` | `picamera0` | Input source. Options: `usbX`, `picameraX`, `/path/to/video.mp4` |
| `--model` | `models/detection.onnx` | Path to the YOLO/ONNX model file. |
| `--conf` | `0.35` | Confidence threshold (0.0 - 1.0). |
| `--iou` | `0.7` | IOU threshold for NMS. |
| `--resolution` | `640x480` | Capture resolution (e.g., `1280x720`). |
| `--tracker` | `bytetrack.yaml` | Tracker config file (`botsort.yaml` or `bytetrack.yaml`). |
| `--max_det` | `50` | Maximum objects to detect per frame. |
| `--save` | `False` | Enable video recording to `records/`. |
| `--headless` | `False` | Run without GUI window (Recommended for Pi). |
| `--line` | `vertical` | Region line orientation. Options: `vertical`, `horizontal`. |

## 4. Arguments Reference (detector_hailo.py)

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--hef` | `models/detection.hef` | Path to the HEF model. |
| `--source` | `picamera0` | Input source. Options: `usbX`, `picameraX`, `/path/to/video.mp4` |
| `--conf` | `0.35` | Confidence threshold (0.0 - 1.0). |
| `--resolution` | `640x480` | Capture resolution (e.g., `1280x720`). |
| `--save` | `False` | Enable video recording to `records/`. |
| `--headless` | `False` | Run without GUI window (Recommended for Pi). |
| `--line` | `vertical` | Region line orientation. Options: `vertical`, `horizontal`. |
| `--classes` | `1` | Comma-separated class IDs to count (e.g., `1` or `0,1`). |

## 5. Hailo Counting + Logs

- Counting is line-crossing based, not raw detection count.
- Default class mapping: `0` = Person, `1` = Sack (change via `--classes`).
- A count log is written to `records/Count_<timestamp>.txt` with rows:
  - `timestamp,current_count,stacked_count`
- MQTT publishes updates to `sack/stats` at most every 3 seconds and only when counts change.

## 6. Examples

**Run with custom model and high confidence:**
```bash
python detector.py --model models/custom.pt --conf 0.6
```

**Run with specific tracker and limit detections:**
```bash
python detector.py --tracker botsort.yaml --max_det 10
```

**Run as a background service (Headless):**
```bash
python detector.py --source picamera0 --headless
```

**Hailo-8L with sack-only counting (default):**
```bash
python detector_hailo.py --hef models/detection.hef --source picamera0 --headless
```

**Hailo-8L counting both classes:**
```bash
python detector_hailo.py --hef models/detection.hef --source picamera0 --classes 0,1 --headless
```
