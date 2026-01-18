#!/data/data/com.termux/files/usr/bin/bash
# OnePlus 6 Shadow Device - Termux Setup Script
# Run this in Termux after installing from F-Droid
set -e

echo "=== OnePlus 6 Shadow Device Setup ==="
echo "Step 1: Termux Environment"
echo ""

echo "[1/4] Setting up storage access..."
termux-setup-storage
sleep 2

echo "[2/4] Updating packages..."
pkg update -y
pkg upgrade -y

echo "[3/4] Installing base tools..."
pkg install -y wget curl git python proot-distro openssh

echo "[4/4] Installing Ubuntu via proot-distro..."
proot-distro install ubuntu

echo ""
echo "=== Termux Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Enter Ubuntu: proot-distro login ubuntu"
echo "  2. Run ubuntu-setup.sh inside Ubuntu"
echo ""
echo "To copy scripts to Ubuntu, from Termux run:"
echo "  cp /sdcard/Download/*.sh ~/"
echo ""
