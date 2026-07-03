# Remote Shadow Device Setup Guide

This guide covers setting up and testing the shadow device when you only have:
- SSH access to your desktop
- ADB access to the OnePlus 6

## Current State (2026-01-19)

### What's Working
- ✅ SSH to device: `ssh -p 8022 10.0.1.62`
- ✅ ADB connected: `adb devices` shows `34f011c6`
- ✅ proot Ubuntu with Python 3.13 and openpilot cloned
- ✅ ZMQ streaming tested and working (0% packet loss)
- ✅ termux-camera-photo captures work (~4.7s/frame, too slow for real-time)
- ✅ Shadow mode detection works (`is_oneplus6()` returns True)

### What's Needed for Real-Time Streaming
- Camera streaming app (IP Webcam or alternative)
- IP Webcam is NOT on F-Droid (proprietary, Play Store only)

## Remote Setup Options

### Option 1: Use Termux Camera Server (Slow but Works)

The `termux_camera_server.py` script uses termux-api for camera access.
This works but is limited to ~0.2 FPS due to termux-camera-photo latency.

```bash
# On desktop - start ZMQ receiver
cd /home/e/Development/openpilot
source .venv/bin/activate
python3 -c "
import zmq, struct, time, cv2, numpy as np
ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.bind('tcp://0.0.0.0:5555')
sock.subscribe(b'frame')
sock.setsockopt(zmq.RCVTIMEO, 120000)
print('Waiting for frames on port 5555...')
for i in range(10):
    try:
        parts = sock.recv_multipart()
        _, meta, jpeg = parts
        fid, w, h = struct.unpack('QII', meta)
        print(f'Frame {i}: {w}x{h}, {len(jpeg)/1024:.1f}KB')
    except zmq.Again:
        print('Timeout')
        break
"

# On device via SSH - run camera test sender
ssh -p 8022 10.0.1.62 "proot-distro login ubuntu -- bash -c '
cd ~/openpilot && source .venv/bin/activate
python3 /data/data/com.termux/files/home/camera_zmq_test.py
'"
```

### Option 2: Install IP Webcam via ADB

IP Webcam is a proprietary app (not on F-Droid). There are several ways to obtain and install it:

**Option A: Extract from another Android device with Play Store**
```bash
# On a device with IP Webcam installed, find the APK path
adb shell pm path com.pas.webcam
# Output: package:/data/app/.../base.apk

# Pull the APK to your desktop
adb pull /data/app/.../base.apk ipwebcam.apk
```

**Option B: Download from an APK mirror site**
Search for "IP Webcam by Pavel Khlebovich" APK. Verify the package name is `com.pas.webcam` and check the SHA256 hash against known versions before installing.

**Option C: Install Aurora Store (F-Droid) to access Play Store apps**
```bash
# Aurora Store is an open-source Play Store client on F-Droid
# Install F-Droid first, then install Aurora Store, then search for IP Webcam
```

**Install and configure via ADB:**
```bash
# Install the APK
adb install ipwebcam.apk

# Grant camera permission
adb shell pm grant com.pas.webcam android.permission.CAMERA

# Grant audio permission (optional, not needed for shadow mode)
adb shell pm grant com.pas.webcam android.permission.RECORD_AUDIO

# Launch IP Webcam
adb shell am start -n com.pas.webcam/.Rolling

# Start the HTTP server programmatically
adb shell am broadcast -a com.pas.webcam.START_SERVER
```

**Recommended settings** (configure via the app UI or scrcpy):
- Resolution: 1280x720 (720p balances quality and performance)
- Quality: 50-80% (higher = more bandwidth, lower latency matters more)
- FPS: 30 (or max available)
- Audio: disabled (not needed)
- Focus mode: Continuous (for driving)

**Verify IP Webcam is working:**
```bash
# Test from device via SSH
ssh -p 8022 10.0.1.62 "curl -s -o /dev/null -w '%{http_code}' http://localhost:8080/shot.jpg"
# Should return 200

# Test from desktop (requires ADB port forward)
adb forward tcp:8080 tcp:8080
curl -s -o /tmp/test.jpg http://localhost:8080/shot.jpg
# Check file size - should be >10KB
ls -la /tmp/test.jpg

# Run MJPEG streamer for shadow mode
ssh -p 8022 10.0.1.62 "proot-distro login ubuntu -- bash -c '
cd ~/openpilot && source .venv/bin/activate
python3 /data/data/com.termux/files/home/mjpeg_zmq_streamer.py \
    --camera http://localhost:8080 \
    --server tcp://10.0.1.123:5555 \
    --fps 15
'"
```

**Auto-start IP Webcam on boot** (optional):
```bash
# Use Termux:Boot (install from F-Droid) to run on device boot
mkdir -p ~/.termux/boot
cat > ~/.termux/boot/start-ipwebcam.sh << 'SCRIPT'
#!/usr/bin/env bash
# NOTE: On Termux, bash is at /data/data/com.termux/files/usr/bin/bash
# but #!/usr/bin/env bash works when Termux is on PATH
sleep 10  # Wait for system to settle
am start -n com.pas.webcam/.Rolling
sleep 3
am broadcast -a com.pas.webcam.START_SERVER
SCRIPT
chmod +x ~/.termux/boot/start-ipwebcam.sh
```

### Option 3: Use scrcpy for Screen/Camera Mirroring

scrcpy can mirror the device screen to desktop. Combined with a camera app:

```bash
# On desktop
scrcpy --no-audio --max-fps 30

# Then manually start any camera app on the mirrored screen
```

### Option 4: Use RTSP from a Camera App

Some camera apps support RTSP. If you find one on F-Droid:

```bash
# Test RTSP stream
ffprobe rtsp://10.0.1.62:8554/stream

# Or use camera_bridge.py with RTSP URL
python camera_bridge.py --url rtsp://10.0.1.62:8554/stream --test
```

## Useful ADB Commands

```bash
# Check device status
adb devices
adb shell getprop ro.product.device  # Should show "OnePlus6"

# List installed packages
adb shell pm list packages | grep -i camera

# Take screenshot
adb exec-out screencap -p > /tmp/screen.png

# Install APK
adb install /path/to/app.apk

# Launch app by package name
adb shell am start -n com.package.name/.MainActivity

# Forward port (access device port from desktop)
adb forward tcp:8080 tcp:8080
# Then access http://localhost:8080 from desktop

# Reverse port (access desktop port from device)
adb reverse tcp:5555 tcp:5555
```

## Useful SSH Commands

```bash
# Quick shadow mode test
ssh -p 8022 10.0.1.62 "proot-distro login ubuntu -- python3 -c '
import subprocess
result = subprocess.run([\"getprop\", \"ro.product.device\"], capture_output=True, text=True)
print(f\"Device: {result.stdout.strip()}\")
print(f\"Is OnePlus 6: {result.stdout.strip().lower() in (\"oneplus6\", \"enchilada\")}\")'
"

# Check if IP Webcam is accessible
ssh -p 8022 10.0.1.62 "curl -s http://localhost:8080/ | head -5"

# Run comparison logger test
ssh -p 8022 10.0.1.62 "proot-distro login ubuntu -- bash -c '
cd ~/openpilot && source .venv/bin/activate
python3 -c \"from tools.shadow.comparison_logger import ComparisonLogger; print(\\\"Logger OK\\\")\"
'"
```

## Network Setup

Desktop IP: 10.0.1.123
Device IP: 10.0.1.62
SSH Port: 8022 (Termux)

### Port Usage
- 5555: ZMQ frame streaming (desktop receives)
- 5556: ZMQ result streaming (device receives)
- 8080: IP Webcam HTTP server (on device)
- 8022: SSH (Termux)

## File Locations

### On Desktop
- `/home/e/Development/openpilot/` - Main repo
- `/home/e/Development/openpilot/tools/shadow/setup/` - Shadow device scripts

### On Device (Termux)
- `/data/data/com.termux/files/home/` - Termux home
- `~/openpilot/` (in proot) - Openpilot clone (proot Ubuntu)
- `.venv/` - Python virtual environment

### Test Scripts on Device
- `/data/data/com.termux/files/home/camera_zmq_test.py` - Camera → ZMQ test
- `/data/data/com.termux/files/home/zmq_test_sender.py` - ZMQ connectivity test
- `/data/data/com.termux/files/home/mjpeg_zmq_streamer.py` - MJPEG → ZMQ streamer

## Next Steps

1. **Get IP Webcam APK** - Download from Play Store on another device, extract APK, transfer via ADB
2. **Test MJPEG streaming** - Once IP Webcam is installed and running
3. **Benchmark real-time performance** - Measure FPS and latency
4. **Test inference server integration** - Run modeld on desktop with streamed frames
