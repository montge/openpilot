#!/usr/bin/env bash
# DGX Spark Quick Start Script
#
# Usage:
#   ./tools/dgx/quickstart.sh          # Setup and verify
#   ./tools/dgx/quickstart.sh --check  # Check only
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=== DGX Spark Quick Start ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo ""

# Check/create venv
VENV_PATH="$PROJECT_ROOT/.venv"

if [ -L "$VENV_PATH" ]; then
    echo "Removing .venv symlink..."
    rm "$VENV_PATH"
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
fi

echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

if [ "$1" != "--check" ]; then
    echo "Installing/upgrading dependencies..."
    pip install --upgrade pip wheel setuptools -q
    pip install numpy tinygrad onnx pytest -q
fi

echo ""
echo "=== Environment ==="
echo "Python: $(python --version)"
echo "Venv: $VIRTUAL_ENV"
echo ""

# Verify tinygrad CUDA
echo "=== Testing tinygrad CUDA ==="
python3 << 'EOF'
from tinygrad import Tensor, Device
Device.DEFAULT = "CUDA"
a = Tensor([1.0, 2.0, 3.0, 4.0])
b = Tensor([5.0, 6.0, 7.0, 8.0])
c = (a + b).numpy()
print(f"CUDA test: {c}")
print("tinygrad CUDA backend: OK")
EOF

echo ""
echo "=== Quick Start Complete ==="
echo ""
echo "Environment is ready. To activate manually:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "To run the GPU detection test:"
echo "  python -c \"from openpilot.system.hardware.nvidia.gpu import *; print(get_nvidia_gpus())\""
