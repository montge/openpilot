# Proposal: Enable Shadow Device Remote Inference

## Summary

Enable the OnePlus 6 shadow device to run full model inference by streaming camera frames to a remote server (desktop/workstation) that runs modeld with GPU acceleration, then receiving inference results back.

## Problem Statement

The OnePlus 6 shadow device cannot run modeld locally due to:

1. **Android linker namespace isolation** - Blocks `/vendor/lib64/libOpenCL.so` from user processes
2. **OpenCL only works as root** - Even with Magisk root, requires `su -c` for every OpenCL call
3. **glibc vs Bionic incompatibility** - proot Ubuntu uses glibc, Android OpenCL uses Bionic
4. **No CPU fallback** - modeld's frame transform pipeline (commonmodel.cc) is 100% OpenCL-dependent

## Proposed Solution

Implement a **remote inference architecture** where:

1. **Shadow device** captures camera frames and streams them to a remote server
2. **Remote server** (desktop with GPU) runs modeld and performs inference
3. **Results** are sent back to the shadow device for logging/comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Shadow Device (OnePlus 6)                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │ IP Webcam /  │───▶│camera_bridge │───▶│    VisionIPC Server      │   │
│  │ Camera App   │    │    .py       │    │    (local frames)        │   │
│  └──────────────┘    └──────────────┘    └───────────┬──────────────┘   │
│                                                      │                   │
│                                          ┌───────────▼──────────────┐   │
│                                          │   frame_streamer.py      │   │
│                                          │   (ZMQ/WebSocket client) │   │
│                                          └───────────┬──────────────┘   │
└──────────────────────────────────────────────────────┼──────────────────┘
                                                       │ Network
                                                       ▼
┌──────────────────────────────────────────────────────┴──────────────────┐
│                         Remote Server (Desktop)                          │
│  ┌──────────────────────────┐    ┌──────────────────────────────────┐   │
│  │   inference_server.py    │───▶│         modeld (GPU)             │   │
│  │   (ZMQ/WebSocket server) │    │   OpenCL/CUDA acceleration       │   │
│  └──────────────────────────┘    └───────────┬──────────────────────┘   │
│                                              │                           │
│                                  ┌───────────▼──────────────────────┐   │
│                                  │   Results sent back to device    │   │
│                                  │   (modelV2 messages)             │   │
│                                  └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Benefits

1. **Full model inference** - Run actual modeld with GPU acceleration
2. **No device modifications** - Works with current proot + rooted setup
3. **Powerful GPU** - Desktop GPUs (RTX, etc.) faster than mobile Adreno
4. **Development flexibility** - Can modify/debug modeld on desktop
5. **Logging/comparison** - Can compare shadow device results with comma device

## Scope

### In Scope
- Frame streaming protocol (device → server)
- Inference result protocol (server → device)
- frame_streamer.py component for shadow device
- inference_server.py component for desktop
- Integration with existing camera_bridge.py and VisionIPC

### Out of Scope
- On-device GPU inference (blocked by linker namespace)
- Real-time vehicle control (shadow mode only)
- Multi-device orchestration (single shadow device initially)

## Dependencies

- Existing: camera_bridge.py, VisionIPC, msgq
- Required: Network connectivity between device and server
- Optional: ZeroMQ for efficient binary streaming

## Risks

| Risk | Mitigation |
|------|------------|
| Network latency | Accept for shadow mode (not safety-critical) |
| Bandwidth (720p @ 20fps ≈ 30MB/s raw) | Use JPEG compression or efficient encoding |
| Server availability | Document setup, provide offline logging mode |

## Success Criteria

1. Frames captured on shadow device appear in modeld on server
2. modelV2 inference results returned to shadow device
3. End-to-end latency < 500ms (acceptable for shadow mode)
4. Can log both frames and inference results for analysis
