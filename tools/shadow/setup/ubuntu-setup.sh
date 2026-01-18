#!/bin/bash
# OnePlus 6 Shadow Device - Ubuntu Setup Script
# Run this inside proot-distro Ubuntu
set -e

echo "=== Ubuntu Environment Setup ==="
echo "Step 2: Build Dependencies"
echo ""

echo "[1/3] Updating Ubuntu packages..."
apt update && apt upgrade -y

echo "[2/3] Installing build dependencies..."
apt install -y \
    build-essential \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-venv \
    clang \
    cmake \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    zlib1g-dev \
    curl \
    wget \
    nano

echo "[3/3] Setting up git-lfs..."
git lfs install

echo ""
echo "=== Ubuntu Setup Complete ==="
echo ""
echo "Next steps:"
echo "  Run clone-openpilot.sh to clone and configure openpilot"
echo ""
