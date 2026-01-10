# Sack Detection Edge Image

This is a minimal Docker image scaffold based on `ultralytics/ultralytics:latest-arm64`.

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
