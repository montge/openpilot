# Camera Integration for Shadow Device

This document covers camera access options for the OnePlus 6 shadow device.

## Overview

Openpilot requires continuous camera frames for model inference. On LineageOS with Termux/proot, direct V4L2 access is not available without root. Several alternative approaches are documented here.

## Frame Rate Requirements

Openpilot's road camera runs at **20 FPS** on production hardware. For shadow mode:

| Use Case | Minimum FPS | Recommended Method |
|----------|-------------|-------------------|
| Pipeline validation ("does it work?") | 0.4+ | termux-api server |
| Data logging / driving review | 10-15 | IP Webcam |
| Model inference testing | 15-20 | IP Webcam / RTSP |
| Real-time comparison with comma device | 20+ | IP Webcam / RTSP |

**Important:** The termux-api approach (~0.4 FPS) is only suitable for verifying the pipeline works. For actual shadow mode usage (data collection, model testing), use IP Webcam or similar app to achieve 15-30 FPS.

## Camera Access Methods

### Method 1: Termux-API Camera Server (Pipeline Testing Only)

**Best for**: Verifying the camera→VisionIPC→modeld pipeline works

**Performance**: ~0.4 FPS (too slow for actual shadow mode usage)

```bash
# Install termux-api (in Termux, not proot)
pkg install termux-api
# Also install Termux:API app from F-Droid

# Start the camera server (in Termux)
python3 tools/shadow/setup/termux_camera_server.py --port 8080

# Test from proot Ubuntu
proot-distro login ubuntu
source ~/openpilot/.venv/bin/activate
cd ~/openpilot
python3 tools/shadow/setup/camera_bridge.py --url http://<phone-ip>:8080 --test
```

**Limitations**:
- ~0.4 FPS due to termux-api overhead (1 frame every 2.5 seconds)
- Only suitable for pipeline validation, NOT for data collection
- Requires Termux:API app with camera permission

**Single frame capture** (for quick tests):
```bash
# From Termux
termux-camera-photo -c 0 /sdcard/photo.jpg

# From proot Ubuntu (add Termux to PATH)
export PATH=$PATH:/data/data/com.termux/files/usr/bin
termux-camera-photo -c 0 /sdcard/photo.jpg
```

### Method 2: IP Webcam App (Recommended for Shadow Mode)

**Best for**: Actual shadow mode usage - data collection, model testing

**Performance**: 15-30 FPS at 720p (suitable for shadow mode)

**Installation**:
1. Install "IP Webcam" from Play Store (requires Google Play or APK sideload)
2. Configure: Resolution 1280x720, Quality 50-80%
3. Start server (default port 8080)

**Endpoints**:
- `http://localhost:8080/shot.jpg` - Single JPEG frame
- `http://localhost:8080/video` - MJPEG stream
- `http://localhost:8080/videofeed` - Alternative MJPEG endpoint

**Usage with camera_bridge.py**:
```bash
# Start IP Webcam on the phone, then in proot:
source ~/openpilot/.venv/bin/activate
cd ~/openpilot

# Test connection
python3 tools/shadow/setup/camera_bridge.py --url http://localhost:8080 --test

# Run full bridge (with VisionIPC if msgq is built)
python3 tools/shadow/setup/camera_bridge.py --url http://localhost:8080
```

**Manual Python capture**:
```python
import cv2

cap = cv2.VideoCapture("http://localhost:8080/video")
while True:
    ret, frame = cap.read()
    if ret:
        # Process frame...
        pass
```

**Why IP Webcam is recommended**:
- 15-30 FPS (vs 0.4 FPS with termux-api)
- Hardware-accelerated encoding
- Configurable resolution and quality
- Stable MJPEG streaming

### Method 3: RTSP Camera Server

**Best for**: Lower latency, hardware encoding

1. Install "RTSP Camera Server" from Play Store
2. Start RTSP server
3. Connect via:
   ```bash
   ffplay rtsp://localhost:8554/camera
   ```

**Python with FFmpeg**:
```python
import subprocess
import numpy as np

cmd = [
    'ffmpeg', '-i', 'rtsp://localhost:8554/camera',
    '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'
]
pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE)
# Read frames from pipe.stdout
```

### Method 4: Custom Camera Bridge (Experimental)

See `camera_bridge.py` for a prototype that:
1. Captures from IP Webcam HTTP stream
2. Converts BGR to NV12 (openpilot format)
3. Can publish to VisionIPC (TODO)

## Camera Specifications

OnePlus 6 cameras detected:

| ID | Type | Resolution | Notes |
|----|------|------------|-------|
| 0 | Back (main) | 4608x2592 | Primary, use for road |
| 1 | Front | 4608x2592 | Could be driver camera |
| 2 | Back (depth) | 2592x1940 | Secondary sensor |
| 3 | Back (logical) | 4608x2592 | Combined output |

## Integration with Openpilot

### Frame Format

Openpilot expects frames in **NV12 YUV** format:
- Y plane: Full resolution (width × height)
- UV plane: Half resolution, interleaved (width × height/2)

### Publishing to VisionIPC

```python
# Pseudocode - requires msgq/visionipc module
from msgq.visionipc import VisionIpcServer, VisionStreamType

vipc = VisionIpcServer("camerad")
vipc.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 4, False, width, height)
vipc.start_listener()

# In capture loop:
vipc.send(VisionStreamType.VISION_STREAM_ROAD, nv12_data, frame_id, timestamp)
```

### Cereal Messaging (Metadata)

```python
from cereal import messaging

pm = messaging.PubMaster(['roadCameraState'])

# Per frame:
msg = messaging.new_message('roadCameraState')
msg.roadCameraState.frameId = frame_id
msg.roadCameraState.timestampEof = timestamp
pm.send('roadCameraState', msg)
```

## Performance Considerations

| Method | FPS | Latency | Use Case |
|--------|-----|---------|----------|
| termux-api server | ~0.4 | Very High | Pipeline testing only |
| IP Webcam HTTP | 15-30 | Medium | **Recommended for shadow mode** |
| RTSP | 20-30 | Low | Alternative to IP Webcam |
| Native (rooted) | 30+ | Lowest | Best performance (requires root) |

## Troubleshooting

### "Camera permission denied"
```bash
# Grant permission via adb
adb shell pm grant com.termux.api android.permission.CAMERA
```

### "Stream connection failed"
- Check camera app is running
- Verify IP address (use `ifconfig`)
- Try localhost if on same device

### "Low frame rate"
- Reduce resolution (720p recommended for shadow mode)
- Close other camera apps
- Check thermal throttling (phone may be hot)

## Recommendations for Shadow Mode

### For actual shadow mode usage (data collection, model testing):
1. **Recommended**: IP Webcam at 720p via HTTP MJPEG (15-30 FPS)
2. **Alternative**: RTSP Camera Server for lower latency

### For pipeline validation only:
1. **Quick test**: termux-api camera server (~0.4 FPS)
2. **Single frames**: termux-camera-photo for snapshots

### Building VisionIPC (msgq module)

To enable full VisionIPC publishing, build the msgq module:

```bash
# In proot Ubuntu
source ~/openpilot/.venv/bin/activate

# Install system dependencies
apt-get install -y libzmq3-dev ocl-icd-opencl-dev opencl-headers

# Install Python build dependencies
pip install Cython scons setuptools

# Download catch2 headers (for tests)
cd ~/openpilot/msgq_repo
mkdir -p msgq/catch2
curl -sL https://raw.githubusercontent.com/catchorg/Catch2/v2.13.9/single_include/catch2/catch.hpp \
  -o msgq/catch2/catch.hpp

# Build msgq
scons -j2

# Verify import
python3 -c "from msgq.visionipc import VisionIpcServer; print('VisionIPC OK')"
```

### Next steps for full pipeline:
1. ✅ Build msgq/VisionIPC module in proot environment
2. ✅ Run camera_bridge.py with VisionIPC publishing
3. Test modeld consumption of frames (requires building modeld)

**Note**: Shadow mode doesn't require real-time performance since no vehicle control is involved. However, 10-15 FPS minimum is needed for meaningful data collection and model behavior testing. The termux-api approach (0.4 FPS) is only useful for verifying the pipeline works.
