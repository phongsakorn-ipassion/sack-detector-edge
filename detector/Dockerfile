FROM ultralytics/ultralytics:latest-arm64

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt /tmp/requirements.txt
# Install system dependencies needed for building wheels (specifically picamera2)
# We need libcamera-dev, libdrm-dev, and build tools.
RUN apt-get update && apt-get install -y \
    libcap-dev \
    libcamera-dev \
    libdrm-dev \
    libgnutls28-dev \
    # FFmpeg dependencies for PyAV (required by picamera2)
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    libavdevice-dev \
    libavfilter-dev \
    libswresample-dev \
    python3-dev \
    build-essential \
    pkg-config \
    meson \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Fix for "canonicalize_version" error in PiDNG/picamera2 build (setuptools>=70 issues)
RUN pip install "setuptools<70.0.0" wheel --break-system-packages

RUN pip install -r /tmp/requirements.txt --break-system-packages

ARG BUILD_ENV=desktop
COPY requirements-pi.txt /tmp/requirements-pi.txt
RUN if [ "$BUILD_ENV" = "pi" ]; then pip install -r /tmp/requirements-pi.txt --break-system-packages; fi

# Copy your detection code into /app before building.
COPY . /app

# Change this to your real entrypoint when code is ready.
CMD ["python", "-u", "detector.py", "--source", "picamera0", "--headless"]
