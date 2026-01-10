# Sack Detection Edge Image

This is a minimal Docker image scaffold based on `ultralytics/ultralytics:latest-arm64`.

## Project Structure

```text
.
├── detector/           # Edge Detection Service (AI counting, MQTT)
│   ├── detector.py     # Main application logic
│   ├── Dockerfile      # Deployment container definition
│   ├── models/         # YOLO detection models (.onnx / .pt)
│   ├── requirements.txt# Python dependencies
│   └── requirements-pi.txt # Pi-specific dependencies (e.g. Picamera2)
├── mosquitto/          # Local MQTT Broker (Mosquitto)
├── nodered/            # Dashboard and Workflows (Node-RED)
├── scripts/            # Environment setup and utility scripts
├── sqlite/             # Database storage for counting events
├── docker-compose.yml  # Root orchestrator for all services
└── .env                # Generated environment config
```

## Docker Compose
```bash
bash scripts/setup.sh
docker compose up -d --build
```

## Build
```bash
docker build -t sack-detector-edge:latest .
```

## Run (interactive shell)
```bash
docker run --rm -it -v "$PWD":/app sack-detector-edge:latest
```

## Run your app
1) Copy your detection code into this folder.
2) Update `Dockerfile` to set the correct command, for example:
```Dockerfile
CMD ["python", "main.py"]
```
3) Rebuild the image.
