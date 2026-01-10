# Sack Detector Edge - Usage Guide

This guide explains how to run the `detector.py` script for object detection and counting.

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

## 3. Arguments Reference

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

## 4. Examples

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
