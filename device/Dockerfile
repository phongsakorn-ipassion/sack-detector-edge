FROM python:3.11-alpine

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Install build dependencies for psutil (linux-headers, gcc, musl-dev)
RUN apk add --no-cache gcc musl-dev linux-headers

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY . /app

CMD ["python", "-u", "device_system_stats.py"]
