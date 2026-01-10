## Context

Testing openpilot changes on real hardware carries risks:
- Bricking the device with bad software
- Safety issues from untested control algorithms
- No way to A/B compare algorithms in real driving conditions

A shadow device running in parallel provides:
- Safe pre-deployment testing
- Real-world algorithm comparison without vehicle control
- Development environment that matches production hardware

**Target Hardware**: OnePlus 6 (Snapdragon 845, same SoC as comma two)

**Stakeholders**: Algorithm developers, contributors without comma hardware, safety engineers

## Goals / Non-Goals

**Goals**:
- Enable safe software validation before production deployment
- Allow real-world A/B testing of algorithms without affecting vehicle control
- Provide development platform with production-equivalent hardware
- Capture synchronized logs for offline comparison analysis

**Non-Goals**:
- Replace comma device for production use
- Enable any form of vehicle control from shadow device
- Support devices other than OnePlus 6 initially
- Real-time comparison (post-hoc analysis is sufficient)

## Decisions

### Decision 1: Shadow Mode Detection

Shadow mode activates when:
1. Device is OnePlus 6 (or explicitly configured)
2. No panda hardware detected (USB or WiFi)
3. `SHADOW_MODE=1` environment variable set (override)

```python
def is_shadow_mode() -> bool:
  if os.environ.get('SHADOW_MODE') == '1':
    return True
  if HARDWARE.get_device_type() == 'oneplus6' and not panda_connected():
    return True
  return False
```

**Rationale**: Safe default - if no panda, assume shadow mode. Explicit override for testing.

### Decision 2: Actuator Lockout

In shadow mode, ALL actuator paths are disabled at multiple levels:

```
Level 1: controlsd.py
  └── if shadow_mode: skip actuator publishing

Level 2: pandad.py
  └── if shadow_mode: refuse CAN writes

Level 3: panda firmware (not present)
  └── No panda = no CAN bus access
```

**Rationale**: Defense in depth. Even if software bug enables control, no hardware path exists.

### Decision 3: Sensor Pipeline

Shadow mode runs the full sensor pipeline:
- Camera capture and encoding
- Model inference (driving model, driver monitoring)
- Localization and calibration
- Path planning and control computation

Control outputs are computed but logged, never sent.

```
┌─────────────────────────────────────────────────────┐
│                   Shadow Device                      │
├─────────────────────────────────────────────────────┤
│  Cameras → modeld → planner → controlsd → [LOG]     │
│                                              ↓       │
│                                         comparison   │
│                                           logger     │
└─────────────────────────────────────────────────────┘
```

**Rationale**: Full pipeline execution ensures realistic comparison with production device.

### Decision 4: Comparison Logging Format

Logs captured in msgpack format with microsecond timestamps:

```python
@dataclass
class ShadowLogEntry:
  timestamp_us: int          # Monotonic microseconds
  frame_id: int              # Camera frame number

  # Model outputs
  model_output: bytes        # Raw model output tensor
  desire: int                # Predicted desire

  # Planned trajectory
  planned_path: list[float]  # x, y, heading over time
  planned_speed: list[float] # Velocity profile

  # Control commands (NOT sent to vehicle)
  steer_cmd: float           # -1 to 1
  accel_cmd: float           # m/s^2

  # Metadata
  lat_active: bool
  long_active: bool
  events: list[str]
```

**Rationale**: Compact binary format, easy to align with primary device logs.

### Decision 5: Time Synchronization

Synchronization strategy:
1. **GPS time**: Both devices use GPS for absolute time reference
2. **Frame timestamps**: Camera frames include monotonic timestamps
3. **Post-hoc alignment**: Comparison tool aligns logs by GPS time + frame matching

```python
def align_logs(shadow_log, primary_log, max_offset_ms=100):
  # Find matching frames by GPS timestamp
  # Interpolate if needed
  # Return aligned pairs
```

**Rationale**: GPS provides ~10ms accuracy, sufficient for control comparison.

### Decision 6: Mount Configuration

Two mounting options:

**Option A: Piggyback Mount**
- Shadow device mounted behind/beside comma device
- Shares approximate camera view
- Simpler mounting, less accurate comparison

**Option B: Split Mount**
- Separate windshield mounts, matched camera angles
- More accurate comparison
- Requires calibration alignment

```
Option A (Piggyback):          Option B (Split):
┌─────────┐                    ┌─────┐ ┌─────┐
│ Comma   │                    │Comma│ │Shadw│
│ [OP6]   │                    └─────┘ └─────┘
└─────────┘                        ↓       ↓
    ↓                          Same road view
Road view
```

**Rationale**: Start with Option A for simplicity; Option B for research-grade comparison.

### Decision 7: Power Supply

OnePlus 6 power options:
1. **USB-C power bank**: 10000mAh = ~4 hours runtime
2. **12V adapter**: Tap into vehicle 12V (like dashcam)
3. **USB from primary**: If comma device has spare USB port

**Rationale**: Power bank for initial testing; 12V for extended sessions.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Shadow device accidentally controls vehicle | Multiple lockout levels; no panda = no CAN |
| Camera view mismatch affects comparison | Document calibration procedure; Option B mount |
| Time sync drift | GPS reference; frame-based alignment |
| OnePlus 6 thermal throttling | Monitor temps; use heat sink if needed |
| Software divergence from comma device | Regular sync with upstream; same AGNOS base |

## Migration Plan

1. **Phase 1**: Shadow mode detection and actuator lockout
2. **Phase 2**: Comparison logger and basic analysis
3. **Phase 3**: Mount hardware design and documentation
4. **Phase 4**: Advanced analysis tools and visualization
5. **Rollback**: Remove shadow mode code; no production impact

## Open Questions

1. Should shadow mode work on comma 3 as well (for users with two devices)?
2. What's the minimum comparison fidelity needed for useful A/B testing?
3. Should we support real-time streaming of shadow logs to laptop for live comparison?
