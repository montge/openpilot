# OnePlus 6 Shadow Device Setup Guide

This guide covers setting up a OnePlus 6 as a shadow device for openpilot development and testing.

## Overview

A shadow device runs the full openpilot pipeline (cameras, models, controls) but **never sends actuator commands** to the vehicle. This enables:

- **Safe development**: Test code changes without risking your production device
- **Parallel testing**: Compare algorithm outputs between devices in real-time
- **Pre-deployment validation**: Verify builds before flashing to comma hardware

## Prerequisites

- OnePlus 6 (codename: enchilada)
- Unlocked bootloader
- Computer with `adb` and `fastboot` installed
- USB cable
- WiFi network (for SSH access)

## Setup Process

### Step 1: Flash LineageOS

See [FLASHING.md](FLASHING.md) for detailed instructions.

**Quick summary:**
1. Verify device is on OxygenOS 11 (Android 11)
2. Download from https://download.lineageos.org/devices/enchilada:
   - `lineage-22.x-*-nightly-enchilada-signed.zip`
   - `boot.img`
   - `dtbo.img`
3. Flash via fastboot:
   ```bash
   fastboot flash dtbo dtbo.img
   fastboot reboot bootloader
   fastboot flash boot boot.img
   # Boot into recovery, factory reset, sideload LineageOS
   adb sideload lineage-*.zip
   ```

### Step 2: Install Termux

1. Complete LineageOS setup wizard (skip Google account)
2. Enable Developer Options (tap Build Number 7 times)
3. Enable USB debugging
4. Download Termux from F-Droid: https://f-droid.org/packages/com.termux/
   - **Do NOT use Play Store version** (outdated)
5. Install via adb: `adb install termux.apk`

### Step 3: Run Setup Scripts

Push scripts to device:
```bash
adb push tools/shadow/setup/*.sh /sdcard/Download/
```

In Termux on the phone:
```bash
# Grant storage permission
termux-setup-storage

# Run Termux setup (installs proot-distro Ubuntu)
cp /sdcard/Download/termux-setup.sh ~/
chmod +x ~/termux-setup.sh
~/termux-setup.sh

# Enter Ubuntu and run setup
proot-distro login ubuntu
cp /sdcard/Download/ubuntu-setup.sh ~/
chmod +x ~/ubuntu-setup.sh
~/ubuntu-setup.sh

# Clone openpilot
cp /sdcard/Download/clone-openpilot.sh ~/
chmod +x ~/clone-openpilot.sh
# Optionally set your fork:
export OPENPILOT_REPO=https://github.com/YOUR_USER/openpilot.git
export OPENPILOT_BRANCH=develop
~/clone-openpilot.sh
```

### Step 4: Setup SSH (Optional but Recommended)

Exit Ubuntu proot (Ctrl+D or `exit`), then in Termux:
```bash
cp /sdcard/Download/setup-ssh.sh ~/
chmod +x ~/setup-ssh.sh
~/setup-ssh.sh
```

Add your SSH public key for passwordless auth:
```bash
mkdir -p ~/.ssh
echo "ssh-ed25519 AAAA... you@computer" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Connect from your computer:
```bash
ssh u0_a191@<phone-ip> -p 8022
```

Run commands in Ubuntu via SSH:
```bash
ssh u0_a191@<phone-ip> -p 8022 "proot-distro login ubuntu -- bash -c 'source ~/.venv/bin/activate && cd ~/openpilot && python3 -c \"from openpilot.system.hardware.shadow_mode import is_shadow_mode; print(is_shadow_mode())\"'"
```

## Verifying Shadow Mode

Inside Ubuntu proot:
```bash
source ~/.venv/bin/activate
cd ~/openpilot

python3 -c "
from openpilot.system.hardware.shadow_mode import is_shadow_mode, is_oneplus6

print(f'OnePlus 6 detected: {is_oneplus6()}')
print(f'Shadow mode active: {is_shadow_mode()}')
"
```

Expected output:
```
OnePlus 6 detected: True
Shadow mode active: True
```

## How It Works

### Device Detection

The shadow mode detection (`system/hardware/shadow_mode.py`) uses two methods:

1. **Device tree** (native Linux): Reads `/sys/firmware/devicetree/base/model`
2. **Android getprop** (proot fallback): Runs `getprop ro.product.device`

In proot environments, the device tree is inaccessible, so the getprop fallback enables detection.

### Shadow Mode Activation

Shadow mode activates when:
1. `SHADOW_MODE=1` environment variable is set, OR
2. OnePlus 6 is detected AND no panda is connected, OR
3. `SHADOW_DEVICE=1` is set AND no panda is connected

### Safety Guarantees

When shadow mode is active:
- `controlsd.py` zeros all actuator values before publishing
- `card.py` blocks CAN send messages entirely
- No panda connection means no CAN bus access

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.

### Quick Fixes

**"proot warning: can't sanitize binding"**
- This is normal and harmless; proot can't access some /proc paths

**SSH connection refused**
- Restart sshd in Termux: `pkill sshd && sshd`
- Check IP hasn't changed: `ifconfig | grep inet`

**Shadow mode not detected**
- Verify you're on OnePlus 6: `getprop ro.product.device` should return `OnePlus6`
- Check for typos in environment variables

**Import errors in Python**
- Activate venv: `source ~/.venv/bin/activate`
- Install missing packages: `pip install <package>`

## File Structure

```
tools/shadow/setup/
├── README.md           # This file
├── FLASHING.md         # LineageOS flashing guide
├── TROUBLESHOOTING.md  # Common issues and solutions
├── termux-setup.sh     # Termux + proot-distro setup
├── ubuntu-setup.sh     # Ubuntu build dependencies
├── clone-openpilot.sh  # Clone and configure openpilot
└── setup-ssh.sh        # SSH server for remote dev
```

## Tested Configuration

- Device: OnePlus 6 (enchilada)
- OS: LineageOS 22.2 (Android 15)
- Termux: F-Droid version 0.118.0+
- proot-distro: Ubuntu 24.04
- Python: 3.12+
