# Remote Inference for Shadow Device

Run model inference on a desktop/server GPU while streaming camera frames from the shadow device (OnePlus 6).

## Overview

The shadow device captures camera frames and streams them over ZeroMQ to a remote server that runs modeld. Results are streamed back for logging and comparison.

```
Shadow Device (OnePlus 6)              Desktop/Server (GPU)
─────────────────────────              ────────────────────
camera app (IP Webcam)
       ↓
camera_bridge.py → VisionIPC
       ↓
frame_streamer.py ──── ZMQ:5555 ────▶ inference_server.py
                                              ↓
                                         VisionIPC
                                              ↓
                                           modeld
                                              ↓
                                          modelV2
                                              ↓
result_receiver.py ◀── ZMQ:5556 ────────────┘
       ↓
ComparisonLogger
```

## Prerequisites

**Shadow device (OnePlus 6)**:
- LineageOS with Termux and proot Ubuntu (see `README.md`)
- Camera streaming app (IP Webcam recommended, see `CAMERA.md`)
- ZeroMQ: `pip install pyzmq opencv-python`
- VisionIPC built in proot (see `CAMERA.md` - "Building VisionIPC" section)

**Desktop/server**:
- openpilot built with `scons -u -j$(nproc)`
- GPU with working OpenCL or CUDA for modeld
- ZeroMQ: `pip install pyzmq opencv-python`

## Quick Start

### 1. Start camera on shadow device

```bash
# Option A: IP Webcam (recommended - 15-30 FPS)
# Start IP Webcam app on phone, then:
ssh -p 8022 <device-ip> "proot-distro login ubuntu -- bash -c '
cd ~/openpilot && source .venv/bin/activate
python3 tools/shadow/setup/camera_bridge.py --url http://localhost:8080'"

# Option B: termux-api (slow - ~0.4 FPS, for testing only)
ssh -p 8022 <device-ip> "python3 tools/shadow/setup/termux_camera_server.py"
```

### 2. Start frame streamer on shadow device

```bash
ssh -p 8022 <device-ip> "proot-distro login ubuntu -- bash -c '
cd ~/openpilot && source .venv/bin/activate
python3 tools/shadow/setup/frame_streamer.py --server tcp://<desktop-ip>:5555 --fps 20'"
```

### 3. Start inference server on desktop

```bash
cd /path/to/openpilot
source .venv/bin/activate
python3 tools/shadow/setup/inference_server.py --port 5555 --result-port 5556
```

This will:
- Listen for frames on port 5555
- Start modeld automatically
- Forward modelV2 results on port 5556

### 4. Start result receiver on shadow device

```bash
ssh -p 8022 <device-ip> "proot-distro login ubuntu -- bash -c '
cd ~/openpilot && source .venv/bin/activate
python3 tools/shadow/setup/result_receiver.py --server tcp://<desktop-ip>:5556'"
```

## Configuration

### Inference Server Options

```
--port PORT         Frame receive port (default: 5555)
--result-port PORT  Result send port (default: 5556)
--width WIDTH       Target frame width (default: 1280)
--height HEIGHT     Target frame height (default: 720)
--model PATH        Model variant or path (default: standard modeld)
--no-modeld         Don't start modeld (run it separately)
--test              Test mode - receive one frame and exit
```

### Frame Streamer Options

```
--server URL        Inference server URL (e.g., tcp://192.168.1.100:5555)
--quality N         JPEG quality 1-100 (default: 80)
--fps N             Target FPS (default: 20)
--test-vipc         Test VisionIPC connection only
--test-zmq          Test ZMQ connection only
```

### Result Receiver Options

```
--server URL        Result server URL (e.g., tcp://192.168.1.100:5556)
```

## Network Configuration

### Required Ports

| Port | Direction | Protocol | Purpose |
|------|-----------|----------|---------|
| 5555 | Device → Desktop | ZMQ PUB/SUB | Camera frames |
| 5556 | Desktop → Device | ZMQ PUB/SUB | Model results |
| 8080 | Local on device | HTTP | IP Webcam stream |
| 8022 | Desktop → Device | SSH | Remote access |

### Firewall Rules

```bash
# On desktop - allow incoming ZMQ connections
sudo ufw allow 5555/tcp
sudo ufw allow 5556/tcp

# Or use ADB port forwarding (no firewall changes needed)
adb forward tcp:5555 tcp:5555
adb reverse tcp:5556 tcp:5556
```

### Bandwidth Estimates

| Resolution | JPEG Quality | FPS | Bandwidth |
|------------|-------------|-----|-----------|
| 1280x720 | 80 | 20 | ~2-4 MB/s |
| 1280x720 | 50 | 20 | ~1-2 MB/s |
| 1280x720 | 80 | 10 | ~1-2 MB/s |
| 640x480 | 80 | 20 | ~0.5-1 MB/s |

Results (modelV2 messages) add minimal bandwidth (~10 KB/s).

## Running Without VisionIPC

If VisionIPC is not built on the shadow device, use the MJPEG-to-ZMQ streamer to bypass VisionIPC entirely:

```bash
# On shadow device (no VisionIPC needed)
python3 tools/shadow/setup/mjpeg_zmq_streamer.py \
    --camera http://localhost:8080 \
    --server tcp://<desktop-ip>:5555 \
    --fps 15
```

This streams JPEG frames directly from IP Webcam to the inference server.

## Running modeld Separately

If you prefer to manage modeld yourself (e.g., for debugging):

```bash
# Terminal 1: Start inference server without modeld
python3 tools/shadow/setup/inference_server.py --no-modeld

# Terminal 2: Start modeld manually
cd /path/to/openpilot
./selfdrive/modeld/modeld
```

## Troubleshooting

### "modeld not found"
Build openpilot first:
```bash
cd /path/to/openpilot
scons -u -j$(nproc)
```

### "modeld exited immediately"
- Check GPU drivers are working: `clinfo` or `nvidia-smi`
- Check modeld logs in stderr output
- Try running modeld directly to see the error

### "No frames received"
- Verify ZMQ connectivity: run `frame_streamer.py --test-zmq`
- Check firewall rules on desktop
- Verify IP addresses are correct
- Try ADB port forwarding as alternative

### "High latency (>500ms)"
- Reduce JPEG quality: `--quality 50`
- Reduce FPS: `--fps 10`
- Use wired connection (USB tethering) instead of WiFi
- Check for thermal throttling on device

### "Frame drops"
- ZMQ PUB/SUB drops frames when subscriber is slow
- Reduce FPS or resolution
- Check CPU load on both devices
- Ensure modeld is consuming frames fast enough

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Frame delivery | >95% | ZMQ PUB/SUB may drop under load |
| End-to-end latency | <500ms | Frame capture → result received |
| Bandwidth | <5 MB/s | At 720p, 20fps, JPEG Q80 |
| Stability | 10+ min | Without crashes |
| Recovery | <5s | Auto-reconnect after network glitch |
