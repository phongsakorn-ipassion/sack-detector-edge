#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Detect environment
OS="$(uname -s)"
OVERRIDE_FILE="$root_dir/docker-compose.override.yml"

# Function to update or add a key-value pair in .env
update_env() {
  local key="$1"
  local value="$2"
  local env_file="$root_dir/.env"

  if [ ! -f "$env_file" ]; then
    echo "$key=$value" > "$env_file"
  else
    if grep -q "^$key=" "$env_file"; then
      # Key exists, replace it
      # Accessing matching line, replacing value
      # Using temp file for compatibility
      sed "s|^$key=.*|$key=$value|" "$env_file" > "$env_file.tmp" && mv "$env_file.tmp" "$env_file"
    else
      # Key missing, append it
      # Ensure newline before appending if file is not empty
      [ -s "$env_file" ] && [ "$(tail -c 1 "$env_file" | wc -l)" -eq 0 ] && echo "" >> "$env_file"
      echo "$key=$value" >> "$env_file"
    fi
  fi
}

if [ "$OS" = "Darwin" ]; then
  echo "Detected macOS. Configured for DESKTOP environment."
  update_env "BUILD_ENV" "desktop"
  # Ensure no override file exists on Mac to keep it clean
  rm -f "$OVERRIDE_FILE"
else
  echo "Detected Linux. Configured for PI environment (default)."
  update_env "BUILD_ENV" "pi"
  
  # Create override file for Linux-specific hardware access
  cat > "$OVERRIDE_FILE" <<EOF
services:
  detector:
    privileged: true
    volumes:
      - /run/udev:/run/udev:ro
  device:
    volumes:
      - /sys/class/thermal:/sys/class/thermal:ro
EOF

  # Install system dependencies on Linux (required for building some wheels)
  PYTHON_PKG="python3-dev" # Default
  # Check if python3-dev is available, otherwise might be python3-devel (Fedora) or just implied.
  # Assuming Debian/PiOS based on "Pi" context.
  echo "Checking system dependencies..."
  if command -v apt-get &> /dev/null; then
      echo "Installing build dependencies (libcap-dev, etc.)..."
      # Suppress output, auto-yes
      sudo apt-get update -qq || true
      sudo apt-get install -y libcap-dev libcamera-dev build-essential python3-dev libatlas-base-dev || echo "⚠️  Failed to install system packages. You may need entering password or manual install."
  fi

  # Check/Install Docker on Linux
  if ! command -v docker &> /dev/null; then
      echo "🐳 Docker not found. Installing Docker..."
      curl -fsSL https://get.docker.com -o get-docker.sh
      sudo sh get-docker.sh
      rm get-docker.sh
      
      echo "🐳 Adding user to docker group..."
      sudo usermod -aG docker "$USER"
      echo "⚠️  You must restart your session (logout/login) for Docker permissions to take effect!"
  else
      echo "✅ Docker is already installed."
  fi
fi

# --- DEPENDENCY INSTALLATION ---
VENV_DIR="$root_dir/.venv"
echo "Checking Python environment..."

# Create venv if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    if [ "$OS" != "Darwin" ]; then
        # On Linux/Pi, we use --system-site-packages to access libcamera/picamera2 drivers
        python3 -m venv --system-site-packages "$VENV_DIR"
    else
        python3 -m venv "$VENV_DIR"
    fi
fi

# Function to run pip inside venv
run_pip() {
    "$VENV_DIR/bin/pip" install --upgrade pip
    # Fix for PiDNG/Picamera2 build issues in recent setuptools
    "$VENV_DIR/bin/pip" install "setuptools<70.0.0" wheel
    "$VENV_DIR/bin/pip" install "$@"
}

echo "Installing Python dependencies into virtual environment..."

# Install common requirement
if [ -f "$root_dir/detector/requirements.txt" ]; then
    echo "Installing base requirements (detector)..."
    run_pip -r "$root_dir/detector/requirements.txt"
fi

# Install Pi specific requirements if on Linux
if [ "$OS" != "Darwin" ] && [ -f "$root_dir/detector/requirements-pi.txt" ]; then
    echo "Installing Pi requirements..."
    run_pip -r "$root_dir/detector/requirements-pi.txt"
fi

# Install Device Service requirements (for local running)
if [ -f "$root_dir/device/requirements.txt" ]; then
    echo "Installing Device Service requirements..."
    run_pip -r "$root_dir/device/requirements.txt"
fi

mkdir -p \
  "$root_dir/mosquitto/config" \
  "$root_dir/mosquitto/data" \
  "$root_dir/mosquitto/log" \
  "$root_dir/nodered/data" \
  "$root_dir/sqlite/data"

if [ ! -f "$root_dir/mosquitto/config/mosquitto.conf" ]; then
cat > "$root_dir/mosquitto/config/mosquitto.conf" <<'EOF'
persistence true
persistence_location /mosquitto/data/
log_dest stdout

listener 1883
protocol mqtt
allow_anonymous true

listener 9001
protocol websockets
allow_anonymous true
EOF
fi

echo "Compose scaffold ready."
echo "--------------------------------------------------------"
echo "✅ Setup Complete"
echo ""
echo "To run the detector locally:"
echo "  1. source .venv/bin/activate"
echo "  2. python detector/detector.py --source usb0"
echo ""
echo "To run via Docker:"
echo "  docker compose up -d --build"
echo "--------------------------------------------------------"
