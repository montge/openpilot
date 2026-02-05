# Change: Add Shadow Device Harness for Parallel Testing

## Why

Developers need a safe way to test experimental algorithms and software changes before deploying to production hardware. A shadow device (OnePlus 6) can serve multiple purposes:

1. **Pre-deployment Validation**: Verify software builds, boots, and runs correctly before flashing the real comma device
2. **Development Surrogate**: Test code changes without risking the production device
3. **Parallel Comparison Testing**: Mount alongside the real device to compare algorithm outputs in real driving conditions without affecting vehicle control

The OnePlus 6 is ideal because:
- Known compatibility (used in comma two)
- Cheap and readily available (~$50-100 used)
- Same Snapdragon 845 as comma two
- Camera capable of running driving models

## What Changes

- **Add Shadow Device Mode** (`system/hardware/shadow_mode.py`):
  - Detect when running as shadow device (not connected to panda)
  - Disable all actuator outputs (steering, gas, brake)
  - Enable full sensor pipeline (cameras, GPS, IMU)
  - Log all control outputs for comparison

- **Add Comparison Logger** (`tools/shadow/comparison_logger.py`):
  - Capture model outputs, planned trajectories, control commands
  - Timestamp synchronization between shadow and primary device
  - Export format for offline analysis

- **Add Shadow Mount Hardware Spec** (`docs/shadow_mount.md`):
  - Mounting bracket design for parallel camera placement
  - Power supply considerations
  - WiFi/USB tethering for log sync

- **Add Comparison Analysis Tools** (`tools/shadow/analyze.py`):
  - Align logs from shadow and primary device
  - Compute divergence metrics (model outputs, control commands)
  - Visualize differences over time

## Impact

- Affected specs: New `shadow-device` capability
- Affected code:
  - `system/hardware/` - Shadow mode detection
  - `selfdrive/controls/` - Output disable for shadow mode
  - `tools/` - New shadow device utilities
- Dependencies: None (uses existing openpilot infrastructure)
- **SAFETY**: Shadow device NEVER sends actuator commands - read-only operation
