# Design: Shadow Device Remote Inference

## Context

### Background

The OnePlus 6 shadow device has been set up with:
- LineageOS 22 + Termux + proot Ubuntu
- Magisk root for GPU device access
- VisionIPC working (camera frames can be published/consumed)
- OpenCL verified working (Adreno 630, OpenCL 2.0) - but only as root

However, modeld cannot run locally because:
1. Android's linker namespace blocks vendor libraries from user processes
2. OpenCL requires running as root (`su -c`)
3. proot Ubuntu uses glibc, incompatible with Android's Bionic-based OpenCL
4. commonmodel.cc has no CPU fallback for frame transforms

### Stakeholders

- Developer using OnePlus 6 as shadow device
- Future shadow device users wanting full pipeline testing

## Goals / Non-Goals

### Goals

1. Run full modeld inference using frames from shadow device
2. Return inference results to shadow device for logging/comparison
3. Minimize latency while accepting shadow mode constraints
4. Reuse existing openpilot infrastructure where possible

### Non-Goals

1. Real-time performance for vehicle control (shadow mode only)
2. On-device GPU inference (blocked by Android restrictions)
3. Support for multiple simultaneous shadow devices (initially)
4. Optimized video codec streaming (JPEG sufficient for prototype)

## Decisions

### Decision 1: Use ZeroMQ for frame streaming

**What**: Use ZeroMQ PUB/SUB for streaming frames from device to server

**Why**:
- Already used extensively in openpilot (msgq, cereal)
- Efficient binary message passing
- Built-in reconnection handling
- Available in both Termux and desktop environments

**Alternatives considered**:
- WebSocket: More complex, unnecessary for binary data
- Raw TCP: No message framing, reconnection handling
- gRPC: Heavy dependency, overkill for this use case
- HTTP chunked: High overhead for streaming

**Trade-offs**:
- Requires ZMQ port forwarding or direct network access
- Not web-browser compatible (not needed)

### Decision 2: JPEG compression for frame transport

**What**: Compress NV12 frames to JPEG before transmission

**Why**:
- Raw NV12 at 1280x720 @ 20fps = ~27 MB/s (too high for WiFi)
- JPEG at quality 80 reduces to ~2-5 MB/s
- OpenCV readily available for encode/decode
- Acceptable quality loss for shadow mode

**Calculation**:
```
Raw NV12: 1280 * 720 * 1.5 bytes * 20 fps = 27.6 MB/s
JPEG Q80: ~100KB/frame * 20 fps = 2 MB/s
```

**Alternatives considered**:
- Raw NV12: Bandwidth too high
- H.264/HEVC: Complex, adds latency, not frame-addressable
- WebP: Less universal support

### Decision 3: Separate frame_streamer from camera_bridge

**What**: Create new `frame_streamer.py` component that consumes VisionIPC and streams to remote

**Why**:
- Separation of concerns (capture vs streaming)
- camera_bridge.py already works and is tested
- Can run frame_streamer independently
- Allows local-only mode (no streaming) to continue working

**Architecture**:
```
camera_bridge.py → VisionIPC → frame_streamer.py → ZMQ → inference_server.py
```

### Decision 4: inference_server.py wraps modeld

**What**: Create server that receives frames, feeds to modeld process, returns results

**Why**:
- Reuses existing modeld code unchanged
- modeld already handles VisionIPC consumption
- Server just bridges network → local VisionIPC

**Implementation approach**:
```
inference_server.py:
  1. Receive JPEG frame via ZMQ
  2. Decode to NV12
  3. Publish to local VisionIPC (as camerad would)
  4. modeld consumes from VisionIPC (unchanged)
  5. Subscribe to modelV2 messages
  6. Send modelV2 back to device via ZMQ
```

### Decision 5: Return serialized modelV2 messages

**What**: Send Cap'n Proto serialized modelV2 messages back to device

**Why**:
- Native openpilot format, no conversion needed
- Device can publish to local messaging for other consumers
- Can log directly in standard format

## Component Design

### frame_streamer.py (Shadow Device)

```python
class FrameStreamer:
    """Streams VisionIPC frames to remote inference server."""

    def __init__(self, server_url: str, stream_type: VisionStreamType):
        self.vipc_client = VisionIpcClient("camerad", stream_type, False)
        self.zmq_socket = zmq.Context().socket(zmq.PUB)
        self.zmq_socket.connect(server_url)

    def run(self):
        while True:
            buf = self.vipc_client.recv()
            if buf:
                # Convert NV12 to JPEG
                jpeg = self._encode_frame(buf)
                # Send with metadata
                self.zmq_socket.send_multipart([
                    b"frame",
                    struct.pack("QII", buf.frame_id, buf.width, buf.height),
                    jpeg
                ])
```

### inference_server.py (Desktop)

```python
class InferenceServer:
    """Receives frames from shadow device, runs modeld, returns results."""

    def __init__(self, listen_port: int):
        self.zmq_socket = zmq.Context().socket(zmq.SUB)
        self.zmq_socket.bind(f"tcp://*:{listen_port}")
        self.zmq_socket.subscribe(b"frame")

        # Local VisionIPC server (feeds modeld)
        self.vipc_server = VisionIpcServer("camerad")
        self.vipc_server.create_buffers(...)

        # Subscribe to modeld output
        self.sm = SubMaster(['modelV2'])

        # ZMQ socket to send results back
        self.result_socket = zmq.Context().socket(zmq.PUB)

    def run(self):
        while True:
            # Receive frame
            topic, metadata, jpeg = self.zmq_socket.recv_multipart()
            frame_id, width, height = struct.unpack("QII", metadata)

            # Decode and publish to VisionIPC
            nv12 = self._decode_frame(jpeg)
            self.vipc_server.send(stream_type, nv12, frame_id, ts, ts)

            # Check for model results
            self.sm.update(0)
            if self.sm.updated['modelV2']:
                # Send back to device
                self.result_socket.send_multipart([
                    b"modelV2",
                    self.sm['modelV2'].to_bytes()
                ])
```

## Data Flow

```
Shadow Device                          Network                    Desktop Server
─────────────────────────────────────────────────────────────────────────────────
IP Webcam → camera_bridge.py
                ↓
           VisionIPC (camerad)
                ↓
         frame_streamer.py
                ↓
           JPEG encode
                ↓
            ZMQ PUB ─────────────────────────────────────▶ ZMQ SUB
                                                              ↓
                                                         JPEG decode
                                                              ↓
                                                      VisionIPC (camerad)
                                                              ↓
                                                          modeld
                                                              ↓
                                                         modelV2
                                                              ↓
            ZMQ SUB ◀───────────────────────────────────── ZMQ PUB
                ↓
         result_receiver.py
                ↓
           Log / Compare
```

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| Network latency (50-200ms) | Delayed inference results | Accept for shadow mode |
| WiFi bandwidth limits | Frame drops | JPEG compression, lower FPS |
| Server must be running | No inference without server | Graceful degradation, local logging |
| JPEG artifacts | Slight quality loss | Quality 80+ acceptable |
| ZMQ port security | Network exposure | Local network only, document firewall |

## Supported Server Hardware

The inference server is designed to run on various hardware:

| Device | Use Case | Notes |
|--------|----------|-------|
| Desktop (any GPU) | Development, bench testing | Recommended starting point |
| RTX 2070 laptop | Portable dev, in-car | Good balance |
| Jetson Orin Nano | In-car deployment | Best for permanent install |
| Jetson Nano (P3450) | Legacy, testing | Works but dated |
| DGX Spark | Heavy development | Overkill but works |

**Recommended progression:**
1. Desktop → develop and test
2. Jetson Orin Nano → in-car deployment
3. Laptop (RTX 2070) → portable development

## Open Questions

1. **Port selection**: What port to use? (suggest 5555 for frames, 5556 for results)
2. **Authentication**: Add any auth? (probably not for local dev network)
3. **Frame rate**: Match camera FPS or allow independent rates?
4. **Buffering**: How many frames to buffer on each side?
