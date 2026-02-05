#!/bin/bash
# OnePlus 6 Shadow Device - Clone Openpilot Script
# Run this inside proot-distro Ubuntu after ubuntu-setup.sh
set -e

# Configuration - modify these as needed
REPO_URL="${OPENPILOT_REPO:-https://github.com/commaai/openpilot.git}"
BRANCH="${OPENPILOT_BRANCH:-master}"

echo "=== Clone Openpilot ==="
echo "Step 3: Repository Setup"
echo ""
echo "Repository: $REPO_URL"
echo "Branch: $BRANCH"
echo ""

cd ~

if [ -d "openpilot" ]; then
    echo "openpilot directory already exists!"
    echo "Remove it first: rm -rf ~/openpilot"
    exit 1
fi

echo "[1/4] Cloning openpilot (shallow)..."
git clone --depth 1 -b "$BRANCH" "$REPO_URL" openpilot
cd openpilot

echo "[2/4] Initializing submodules..."
git submodule update --init --depth 1

echo "[3/4] Setting up Python virtual environment..."
# Use user's global venv at ~/.venv (preferred for dev workflow)
if [ ! -d "$HOME/.venv" ]; then
    echo "Creating ~/.venv..."
    python3 -m venv "$HOME/.venv"
fi
# Create symlink for convenience
ln -sf "$HOME/.venv" .venv
source "$HOME/.venv/bin/activate"

echo "[4/4] Installing base Python packages..."
pip install --upgrade pip wheel
pip install numpy pycapnp

echo ""
echo "=== Clone Complete ==="
echo ""
echo "Openpilot is installed at: ~/openpilot"
echo ""
echo "To activate the environment:"
echo "  source ~/.venv/bin/activate && cd ~/openpilot"
echo ""
echo "To test shadow mode detection:"
echo "  python3 -c \"from openpilot.system.hardware.shadow_mode import is_shadow_mode; print(f'Shadow mode: {is_shadow_mode()}')\""
echo ""
