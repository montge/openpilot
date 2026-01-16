# OnePlus 6 Shadow Device Setup Guide

This guide covers setting up a OnePlus 6 (codename: "enchilada") as a shadow device for openpilot algorithm testing and comparison.

## Why OnePlus 6?

| Feature | OnePlus 6 | comma three |
|---------|-----------|-------------|
| SoC | Snapdragon 845 | Snapdragon 845 |
| RAM | 6/8 GB | 4 GB |
| GPU | Adreno 630 | Adreno 630 |
| Price | ~$50-100 used | $999 new |

The OnePlus 6 shares the same Snapdragon 845 as the comma three, making it ideal for running the same ML models and comparing algorithm outputs.

## OS Options

### Option 1: postmarketOS (Recommended for Shadow Device)

**Pros:**
- Mainline Linux kernel (ongoing security patches)
- Native Linux environment (no Android overhead)
- Active community support
- Front camera working, rear camera in development

**Cons:**
- Rear camera support still in progress (front camera works)
- Some features partial (WiFi 2.4GHz stability)

**Status (as of 2025):**
- Display, touch, battery, audio, Bluetooth, 3D acceleration: Working
- Front camera: Working (edge branch)
- Rear camera: In development
- GPS, Modem: Partial

### Option 2: LineageOS (Android-based)

**Pros:**
- Full camera support
- Complete Android functionality
- Well-documented installation

**Cons:**
- Android overhead
- More complex openpilot integration
- Requires maintaining Android compatibility layer

### Option 3: AGNOS Port (Advanced)

**Pros:**
- Closest to production comma three experience
- Same OS as comma devices

**Cons:**
- Requires custom device tree adaptation
- Significant development effort
- Not officially supported

## Recommended Path: postmarketOS

For shadow device use, postmarketOS is recommended because:
1. Native Linux means openpilot runs directly without Android
2. Mainline kernel shares code with AGNOS's SDM845 kernel
3. Active development community
4. Lower overhead = better performance for ML inference

## Installation Guide (postmarketOS)

### Prerequisites

- OnePlus 6 (A6003 Global or A6000 APAC)
- USB-C cable
- Computer with `fastboot` installed
- Backup of any data on the device

### Step 1: Unlock Bootloader

1. Enable Developer Options:
   - Settings > About Phone > Tap "Build Number" 7 times

2. Enable OEM Unlocking:
   - Settings > Developer Options > OEM Unlocking > Enable

3. Boot to fastboot mode:
   ```bash
   adb reboot bootloader
   ```

4. Unlock bootloader (WARNING: This wipes all data):
   ```bash
   fastboot oem unlock
   ```

### Step 2: Install postmarketOS

1. Download pmbootstrap:
   ```bash
   pip install --user pmbootstrap
   ```

2. Initialize pmbootstrap:
   ```bash
   pmbootstrap init
   ```
   - Select `oneplus-enchilada` as device
   - Select `phosh` or `sxmo` as UI (or `none` for headless)
   - Select `edge` channel for latest camera support

3. Install to device:
   ```bash
   pmbootstrap install
   pmbootstrap flasher flash_rootfs
   pmbootstrap flasher flash_kernel
   ```

4. Reboot:
   ```bash
   fastboot reboot
   ```

### Step 3: Initial Setup

1. Connect via USB or SSH:
   ```bash
   ssh user@172.16.42.1  # USB networking
   ```

2. Update system:
   ```bash
   sudo apk update
   sudo apk upgrade
   ```

3. Install development tools:
   ```bash
   sudo apk add python3 py3-pip git clang cmake
   ```

## Installing openpilot

### Clone Repository

```bash
git clone https://github.com/commaai/openpilot.git
cd openpilot
git submodule update --init --recursive
```

### Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# System dependencies (Alpine Linux)
sudo apk add capnproto zeromq-dev opencv-dev
```

### Configure Shadow Mode

Shadow mode auto-activates on OnePlus 6 without a panda:

```bash
# Verify shadow mode detection
python -c "from openpilot.system.hardware.shadow_mode import is_shadow_mode; print(f'Shadow mode: {is_shadow_mode()}')"
```

Or force shadow mode explicitly:

```bash
export SHADOW_MODE=1
./launch_openpilot.sh
```

## Camera Configuration

### Current Status

| Camera | Sensor | Status |
|--------|--------|--------|
| Front | IMX371 | Working (edge) |
| Rear (main) | IMX519 | In development |
| Rear (secondary) | IMX376 | Working (edge) |

### Using Front Camera for Testing

For initial testing, the front camera can be used:

```bash
# Check camera devices
ls /dev/video*

# Test with libcamera
libcamera-hello --camera 0
```

### Camera Calibration

See [camera_calibration.md](camera_calibration.md) for detailed calibration instructions.

## Power Options

### USB-C Power Delivery

The OnePlus 6 supports USB-C PD charging:
- Standard: 5V/4A (20W Dash Charge)
- USB-PD: Up to 15W

### 12V Vehicle Power

For in-vehicle use:
1. Use a quality USB-C car charger (minimum 18W)
2. Consider a USB-C PD car charger for faster charging
3. Avoid cheap chargers that may cause voltage drops

### Battery Management

For continuous operation:
```bash
# Monitor battery status
cat /sys/class/power_supply/battery/capacity
cat /sys/class/power_supply/battery/status

# Check thermal status
cat /sys/class/thermal/thermal_zone*/temp
```

## Thermal Management

The Snapdragon 845 can throttle under sustained load. For continuous operation:

### Passive Cooling

1. Remove case for better heat dissipation
2. Position device with good airflow
3. Consider adhesive heat sink on back panel

### Active Cooling

For sustained high-performance operation:
1. Small USB fan directed at device
2. Peltier cooler pad (requires external power)

### Monitoring

```bash
# Watch CPU temperatures
watch -n 1 'cat /sys/class/thermal/thermal_zone*/temp'

# Check CPU frequency (throttling indicator)
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
```

### Throttling Thresholds

| Zone | Warning | Throttle | Shutdown |
|------|---------|----------|----------|
| CPU | 75C | 85C | 95C |
| GPU | 75C | 85C | 95C |
| Battery | 40C | 45C | 50C |

## Mounting

### Parallel Mount (Recommended)

Mount the OnePlus 6 alongside your production comma device:

```
┌─────────────────────────────────────┐
│           Windshield                │
├─────────────────┬───────────────────┤
│  comma three    │   OnePlus 6       │
│  (production)   │   (shadow)        │
│  [actuates]     │   [observes]      │
└─────────────────┴───────────────────┘
```

### Camera Alignment

Both cameras should view the same scene:
1. Mount at same height
2. Parallel optical axes
3. Minimal lateral offset

### Mount Options

1. **3D Printed Bracket** - See `tools/shadow/mounts/` for STL files
2. **Universal Phone Mount** - Windshield suction mount
3. **Custom Piggyback** - Attach to existing comma mount

## Troubleshooting

### Device Not Detected

```bash
# Check USB connection
lsusb | grep -i oneplus

# Check dmesg for USB errors
dmesg | tail -20
```

### Camera Not Working

```bash
# Check camera subsystem
ls /dev/video*
dmesg | grep -i camera

# Verify libcamera
libcamera-hello --list-cameras
```

### Thermal Throttling

```bash
# Check current CPU frequency
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq

# Compare to max frequency
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
```

### Shadow Mode Not Activating

```bash
# Check device detection
python -c "from openpilot.system.hardware.shadow_mode import is_oneplus6; print(is_oneplus6())"

# Check panda detection
python -c "from openpilot.system.hardware.shadow_mode import panda_connected; print(panda_connected())"

# Force shadow mode
export SHADOW_MODE=1
```

## Next Steps

1. [Camera Calibration](camera_calibration.md) - Calibrate for accurate comparison
2. [Comparison Testing](../tools/shadow/README.md) - Run comparison tests
3. [Algorithm Harness](../selfdrive/controls/lib/tests/algorithm_harness/README.md) - Replay shadow logs

## References

- [postmarketOS OnePlus 6 Wiki](https://wiki.postmarketos.org/wiki/OnePlus_6_(oneplus-enchilada))
- [LineageOS Installation](https://wiki.lineageos.org/devices/enchilada/install/)
- [comma AGNOS Builder](https://github.com/commaai/agnos-builder)
- [Shadow Device README](../../tools/shadow/README.md)
