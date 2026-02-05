# Design: OnePlus 6 Shadow Device with LineageOS

## Context

This design documents the technical decisions for running openpilot on OnePlus 6 as a shadow device using LineageOS instead of AGNOS.

### Background

- AGNOS is built for comma 3/3X custom hardware, not generic Android phones
- NEOS (comma two OS) only supports OnePlus 3T, not OnePlus 6
- LineageOS provides a stable, maintained Android base for OnePlus 6 (codename: enchilada)
- Shadow mode requires camera access but no vehicle actuation

### Stakeholders

- Developers wanting to test openpilot without production hardware
- Contributors who want parallel testing capability
- Users with OnePlus 6 devices seeking to repurpose them

## Goals / Non-Goals

### Goals

1. Document a working setup for OnePlus 6 as shadow device
2. Provide automated setup scripts for reproducibility
3. Enable camera frame capture for model inference testing
4. Support SSH-based remote development workflow

### Non-Goals

1. Full feature parity with comma hardware
2. Real-time driving capability (shadow mode only)
3. Driver monitoring camera integration (initially)
4. Performance optimization for production use

## Decisions

### Decision 1: Use LineageOS + Termux + proot-distro

**What**: Run openpilot inside a proot Ubuntu environment within Termux on LineageOS

**Why**:
- LineageOS is actively maintained for OnePlus 6 (weekly builds)
- Termux provides Linux environment without rooting
- proot-distro enables running Ubuntu userspace
- Avoids complex custom ROM building (NEOS/AGNOS porting)

**Alternatives considered**:
- Port NEOS to OnePlus 6: High effort, NEOS is deprecated, no active maintainers
- Port AGNOS to OnePlus 6: Extremely high effort, hardware-specific
- Bare Termux without proot: Limited package availability, build difficulties

**Trade-offs**:
- Performance overhead from proot emulation (~10-20%)
- No direct hardware access (V4L2, GPU)
- Camera requires workaround via termux-api or HTTP streaming

### Decision 2: Android getprop fallback for device detection

**What**: Add `getprop ro.product.device` as fallback when device tree is inaccessible

**Why**:
- `/sys/firmware/devicetree/base/model` is not exposed in proot
- Android's getprop command works from within proot
- Maintains backward compatibility with native detection

**Implementation**:
```python
def is_oneplus6():
    # Try device tree first
    model = _get_device_model()
    if "OnePlus" in model:
        return True
    # Fallback to getprop
    device = _get_android_device()  # calls getprop
    return device.lower() in ("oneplus6", "enchilada")
```

### Decision 3: Camera access via HTTP streaming

**What**: Use IP Webcam-style app + HTTP MJPEG capture instead of V4L2

**Why**:
- V4L2 requires root access or custom kernel
- termux-api only provides single-frame capture (termux-camera-photo)
- HTTP MJPEG streaming is well-supported and platform-agnostic
- Can achieve 15-30 FPS depending on resolution and network

**Architecture**:
```
┌─────────────────┐      HTTP MJPEG      ┌─────────────────┐
│ Camera App      │  ─────────────────▶  │ camera_bridge.py│
│ (IP Webcam)     │  localhost:8080      │                 │
└─────────────────┘                      └────────┬────────┘
                                                  │
                                                  ▼ NV12 frames
                                         ┌─────────────────┐
                                         │ VisionIPC       │
                                         │ Server          │
                                         └────────┬────────┘
                                                  │
                                                  ▼
                                         ┌─────────────────┐
                                         │ modeld          │
                                         │ (consumer)      │
                                         └─────────────────┘
```

**Alternatives considered**:
- Root + V4L2: Requires unlocking and rooting, voids warranty
- Custom Android app: High development effort, maintenance burden
- termux-api loop: Too slow for continuous capture (~2-5 FPS)

**Trade-offs**:
- Additional latency (~50-100ms) vs native capture
- Requires running separate camera app
- Network stack overhead (localhost)

### Decision 4: OpenCL not available without root

**What**: GPU compute (OpenCL) is blocked by device access restrictions in the proot environment

**Why this matters**:
- modeld uses OpenCL for frame transforms (resizing, color space conversion)
- Without OpenCL, the full perception pipeline cannot run
- VisionIPC works fine, but modeld cannot consume frames for inference

**Investigation findings**:
```
# OpenCL library exists in proot
/usr/lib/aarch64-linux-gnu/libOpenCL.so.1 → /usr/lib/aarch64-linux-gnu/libOpenCL.so.1.0.0

# But no ICD (Installable Client Driver) vendors configured
/etc/OpenCL/vendors/ is empty

# Android has Adreno OpenCL driver
/vendor/lib64/libOpenCL.so (Adreno GPU)

# GPU hardware is present
GPU: Adreno630v2 (via /sys/class/kgsl/kgsl-3d0/gpu_model)

# BUT device nodes are inaccessible without root
/dev/kgsl* - No such file or directory (not exposed to proot)
/dev/dri/ - Permission denied (even if it existed)
```

**Root cause**: Android restricts GPU device access to system apps and processes with appropriate SELinux contexts. The proot environment runs as a regular user without these privileges.

**Options for full modeld**:
| Option | Effort | Viable? |
|--------|--------|---------|
| Root the device | Low | Yes - enables /dev/kgsl access |
| CPU-only modeld fallback | High | Maybe - modeld is GPU-dependent |
| Remote inference server | Medium | Yes - send frames to desktop GPU |
| Accept partial pipeline | None | Yes - VisionIPC works, just no modeld |

**Recommendation**: For shadow mode development:
1. Use VisionIPC for frame distribution testing (works now)
2. Use remote inference if full modeld is needed
3. Defer rooting decision to user preference

### Decision 5: SSH-based development workflow

**What**: Configure SSH server in Termux for remote development

**Why**:
- On-device development is slow and cumbersome
- SSH enables using desktop IDE and tools
- Can run long processes without keeping screen on
- Enables scripted deployment and testing

**Configuration**:
- Port 8022 (Termux default, avoids conflict with system SSH)
- Key-based authentication recommended
- Access proot via: `ssh user@ip -p 8022 "proot-distro login ubuntu -- command"`

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| proot performance overhead | Accept for shadow/dev use; not for production |
| Camera latency | Document limitations; use for validation not real-time |
| LineageOS updates break setup | Pin tested version; document recovery |
| Termux app killed by Android | Use wakelock; document battery optimization |
| **OpenCL not available** | Root device, use remote inference, or accept partial pipeline |
| GPU device access denied | Root required; fundamental Android security restriction |

## Migration Plan

Not applicable - this is new functionality for shadow device support.

## Open Questions

1. ~~**Best camera streaming app?**~~ **RESOLVED**: IP Webcam recommended (15-30 FPS). Created termux_camera_server.py as fallback (0.4 FPS, pipeline testing only).
2. ~~**VisionIPC integration**~~ **RESOLVED**: camera_bridge.py publishes NV12 frames to VisionIPC. Server/client communication verified working.
3. ~~**Performance benchmarks**~~ **RESOLVED**: termux-api ~0.4 FPS, IP Webcam 15-30 FPS. Documented in CAMERA.md.
4. **Driver camera support** - Can front camera be used for driver monitoring? (Future work)
5. **OpenCL access** - How to enable GPU without rooting? **BLOCKED**: Not possible. Root or alternative approach required.
