FROM ultralytics/ultralytics:latest

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /tmp/requirements.txt

ARG BUILD_ENV=desktop
COPY requirements-pi.txt /tmp/requirements-pi.txt
RUN if [ "$BUILD_ENV" = "pi" ]; then pip install -r /tmp/requirements-pi.txt; fi

# Copy your detection code into /app before building.
COPY . /app

# Change this to your real entrypoint when code is ready.
CMD ["python", "-u", "main.py"]
