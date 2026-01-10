FROM ultralytics/ultralytics:latest-arm64

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt /tmp/requirements.txt
# Install system dependencies needed for building wheels (e.g. picamera2, python-prctl)
RUN apt-get update && apt-get install -y libcap-dev && rm -rf /var/lib/apt/lists/*
RUN pip install -r /tmp/requirements.txt --break-system-packages

# RUN pip install --upgrade pip setuptools wheel --break-system-packages

ARG BUILD_ENV=desktop
COPY requirements-pi.txt /tmp/requirements-pi.txt
RUN if [ "$BUILD_ENV" = "pi" ]; then pip install -r /tmp/requirements-pi.txt --break-system-packages; fi

# Copy your detection code into /app before building.
COPY . /app

# Change this to your real entrypoint when code is ready.
CMD ["python", "-u", "detector.py", "--source", "picamera0", "--headless"]
