#!/data/data/com.termux/files/usr/bin/bash
# OnePlus 6 Shadow Device - SSH Setup Script
# Run this in Termux (not inside Ubuntu proot)
set -e

echo "=== SSH Server Setup ==="
echo ""

# Check if openssh is installed
if ! command -v sshd &> /dev/null; then
    echo "Installing OpenSSH..."
    pkg install -y openssh
fi

echo "[1/3] Setting SSH password..."
echo "You will be prompted to set a password for SSH access."
echo ""
passwd

echo ""
echo "[2/3] Starting SSH server..."
sshd

echo ""
echo "[3/3] Getting connection info..."
echo ""

# Get IP address
IP=$(ifconfig 2>/dev/null | grep -A1 wlan0 | grep inet | awk '{print $2}')
USER=$(whoami)

echo "=== SSH Setup Complete ==="
echo ""
echo "SSH server is running on port 8022"
echo ""
echo "Connect from your computer with:"
echo "  ssh $USER@$IP -p 8022"
echo ""
echo "For key-based auth, add your public key:"
echo "  mkdir -p ~/.ssh"
echo "  echo 'your-public-key' >> ~/.ssh/authorized_keys"
echo "  chmod 600 ~/.ssh/authorized_keys"
echo ""
echo "To run commands in Ubuntu via SSH:"
echo "  ssh $USER@$IP -p 8022 'proot-distro login ubuntu -- command'"
echo ""
