# Camera Integration for Shadow Device

This document covers camera access options for the OnePlus 6 shadow device.

## Overview

Openpilot requires continuous camera frames for model inference. On LineageOS with Termux/proot, direct V4L2 access is not available without root. Several alternative approaches are documented here.

## Camera Access Methods

### Method 1: Termux-API (Snapshots)

**Best for**: Testing, validation, single-frame capture

```bash
# Install
pkg install termux-api
# Also install Termux:API app from F-Droid

# Capture single frame
termux-camera-photo -c 0 /sdcard/photo.jpg

# From proot Ubuntu (add Termux to PATH)
export PATH=$PATH:/data/data/com.termux/files/usr/bin
termux-camera-photo -c 0 /sdcard/photo.jpg
```

**Limitations**:
- Single frame only (~0.5-1 second per capture)
- Not suitable for continuous streaming
- Requires Termux:API app with camera permission

### Method 2: IP Webcam App (Recommended)

**Best for**: Continuous streaming, integration with openpilot

1. Install "IP Webcam" from Play Store
2. Start server (default port 8080)
3. Access streams:
   - `http://localhost:8080/shot.jpg` - Single JPEG
   - `http://localhost:8080/video` - MJPEG stream

**Python capture example**:
```python
import cv2
import numpy as np

# MJPEG stream
cap = cv2.VideoCapture("http://localhost:8080/video")
while True:
    ret, frame = cap.read()
    if ret:
        # Process frame...
        pass
```

**Performance**: 15-30 FPS at 720p/1080p

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

| Method | FPS | Latency | CPU Usage |
|--------|-----|---------|-----------|
| termux-api | 1-2 | High | Low |
| IP Webcam HTTP | 15-30 | Medium | Medium |
| RTSP | 20-30 | Low | Medium |
| Native (rooted) | 30+ | Lowest | Lowest |

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

For shadow device development/testing:

1. **Primary**: IP Webcam at 720p via HTTP MJPEG
2. **Fallback**: termux-api for single frame validation
3. **Future**: Custom VisionIPC publisher for full integration

Note: Shadow mode doesn't require real-time performance since no vehicle control is involved. 10-15 FPS is adequate for logging and comparison testing.
