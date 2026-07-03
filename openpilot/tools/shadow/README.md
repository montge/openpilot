# Shadow Device Comparison Testing

Tools for running openpilot on a secondary "shadow" device alongside a production device for comparison testing.

## Overview

Shadow mode enables running the full openpilot pipeline on a secondary device (like OnePlus 6) without any actuator output. This allows:

- Comparing algorithm outputs between different hardware/software versions
- Testing changes without affecting the production device
- Capturing detailed logs for offline analysis
- Validating model behavior across different platforms

## Quick Start

### 1. Enable Shadow Mode

Shadow mode is automatically detected when:
- Running on OnePlus 6 hardware without a panda connected
- The `SHADOW_DEVICE=1` environment variable is set without panda

You can also force shadow mode:
```bash
SHADOW_MODE=1 ./launch_openpilot.sh
```

### 2. Capture Logs

Use the ComparisonLogger to capture detailed logs:

```python
from openpilot.tools.shadow import ComparisonLogger, FrameData

logger = ComparisonLogger(output_dir="/data/shadow_logs")
logger.start_segment("my_drive_001")

# In your main loop
frame_data = FrameData(
    frame_id=frame_id,
    timestamp_mono=time.monotonic(),
    timestamp_gps=gps_time,  # For cross-device sync
    model_outputs={"desired_curvature": curvature},
    controls={"steer": steer_cmd, "accel": accel_cmd},
    state={"lat_active": lat_active, "long_active": long_active},
)
logger.log_frame(frame_data)

logger.end_segment()
```

### 3. Align and Analyze Logs

```python
from openpilot.tools.shadow import ComparisonLogger, LogAligner, compute_all_metrics

# Load logs
shadow_frames = ComparisonLogger.load_segment("/data/shadow_logs/drive_001")
prod_frames = ComparisonLogger.load_segment("/data/prod_logs/drive_001")

# Align using GPS timestamps
aligner = LogAligner()
result = aligner.auto_align(shadow_frames, prod_frames)

# Compute metrics
report = compute_all_metrics(result)
print(f"Steer RMSE: {report.control_metrics.steer_rmse:.4f}")
print(f"Accel RMSE: {report.control_metrics.accel_rmse:.4f}")
```

### 4. CLI Analysis Tool

```bash
# Generate comparison report
python tools/shadow/analyze.py \
    --shadow /data/shadow_logs/segment_001 \
    --prod /data/prod_logs/segment_001 \
    --output report.md

# Output as JSON
python tools/shadow/analyze.py \
    --shadow /path/to/shadow \
    --prod /path/to/prod \
    --json --output metrics.json
```

## Components

### Shadow Mode Detection (`system/hardware/shadow_mode.py`)

- `is_shadow_mode()` - Check if running in shadow mode
- `panda_connected()` - Check for panda hardware
- `SHADOW_MODE` - Module-level constant for fast access

### Comparison Logger (`tools/shadow/comparison_logger.py`)

- `ComparisonLogger` - Main logging class
- `FrameData` - Data container for each frame
- Supports JSON serialization with optional gzip compression
- Automatic log rotation for long sessions

### Log Alignment (`tools/shadow/align.py`)

- `LogAligner` - Aligns logs from two devices
- Supports GPS-based, frame ID-based, and timestamp-based alignment
- `AlignmentResult` - Contains paired frames and unmatched frames
- `validate_alignment()` - Check alignment quality

### Metrics (`tools/shadow/metrics.py`)

- `compute_control_metrics()` - Steer/accel divergence
- `compute_trajectory_metrics()` - Path/speed divergence
- `compute_model_metrics()` - Model output divergence
- `format_report_markdown()` - Generate markdown report

## Safety

Shadow mode includes multiple layers of safety:

1. **controlsd.py** - Zeros all actuator values before publishing
2. **card.py** - Blocks sendcan messages (defense in depth)
3. **Startup warnings** - Logs clear "SHADOW MODE" warnings

The safety checks ensure no CAN actuation messages are ever sent, even if one layer fails.

## Supported Devices

### Primary (Shadow Device)
- **OnePlus 6** - Automatically detected, same Snapdragon 845 as comma two
- Any device with `SHADOW_DEVICE=1` environment variable

### Requirements
- No panda connection (shadow mode activates automatically)
- Same camera mounting as production for valid comparison
- GPS for cross-device time synchronization (recommended)

## Running in proot (OnePlus 6 / LineageOS)

On the OnePlus 6, openpilot runs inside a **proot-distro Ubuntu** environment within Termux. This has important implications:

### Shadow Mode Detection in proot

Shadow mode detection uses `getprop` to identify OnePlus 6 hardware. In proot, the Android `getprop` binary is accessible via Termux's PATH:

```bash
# Ensure Termux binaries are on PATH (already set by setup scripts)
export PATH=$PATH:/data/data/com.termux/files/usr/bin

# Verify detection
python3 -c "from openpilot.system.hardware.shadow_mode import is_shadow_mode; print(is_shadow_mode())"
```

If `getprop` is not found, set the environment variable explicitly:
```bash
SHADOW_MODE=1 ./launch_openpilot.sh
```

### proot Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| No GPU access | OpenCL unavailable | Remote inference server (see `enable-shadow-remote-inference`) |
| No `/dev/kgsl-3d0` | modeld can't use GPU | CPU-only or remote inference |
| No V4L2 cameras | Direct camera access unavailable | IP Webcam HTTP stream (see `setup/CAMERA.md`) |
| glibc vs Bionic | Can't load Android `.so` libs | Build native or use remote |
| No kernel modules | Limited hardware access | Use Termux APIs for hardware |

### proot Performance Notes

- **CPU inference**: Snapdragon 845 A75 cores are capable but slow without GPU acceleration
- **Memory**: 6-8 GB RAM on OnePlus 6 is sufficient for the pipeline minus GPU inference
- **Storage**: Use `/data` partition for logs (larger than `/home`)
- **Thermals**: Monitor temperature - sustained inference causes throttling. See `setup/TROUBLESHOOTING.md`

### Setup Reference

See `tools/shadow/setup/README.md` for the complete OnePlus 6 setup guide covering:
- LineageOS flashing (`FLASHING.md`)
- Termux and proot environment setup
- Camera integration (`CAMERA.md`)
- Rooting for GPU access (`ROOTING.md`)
- Remote access (`REMOTE_SETUP.md`)

## Workflow

```
┌─────────────────┐     ┌─────────────────┐
│ Production      │     │ Shadow Device   │
│ (comma three)   │     │ (OnePlus 6)     │
├─────────────────┤     ├─────────────────┤
│ Full openpilot  │     │ Full openpilot  │
│ + actuators     │     │ NO actuators    │
│ Logs to /data   │     │ Logs to /data   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
          ┌─────────────────────┐
          │  Log Alignment      │
          │  (GPS or frame ID)  │
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │  Comparison Metrics │
          │  - Control error    │
          │  - Trajectory error │
          │  - Model divergence │
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │  Report / Analysis  │
          └─────────────────────┘
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SHADOW_MODE=1` | Force shadow mode on |
| `SHADOW_MODE=0` | Force shadow mode off |
| `SHADOW_DEVICE=1` | Mark this as a shadow device |

## Algorithm Harness Integration

Shadow logs can be imported into the algorithm test harness for replaying real-world data:

```python
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.shadow_import import (
    import_shadow_segment,
    compare_shadow_to_harness,
    format_shadow_comparison_report,
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.runner import ScenarioRunner
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.adapters import LatControlPIDAdapter

# Import shadow segment as harness scenario
scenario = import_shadow_segment(
    "/data/shadow_logs/route_001",
    mode="lateral",  # or "longitudinal"
)

# Run algorithm through harness
runner = ScenarioRunner()
adapter = LatControlPIDAdapter()
result = runner.run(adapter, scenario, "LatControlPID")

# Compare outputs
metrics = compare_shadow_to_harness(
    frames=ComparisonLogger.load_segment("/data/shadow_logs/route_001"),
    harness_outputs=result.outputs,
    mode="lateral",
)

# Generate report
report = format_shadow_comparison_report(metrics, "LatControlPID")
print(report)
```

### Shadow Capture to Harness Analysis Workflow

```
Shadow Device (capture)              Desktop (analysis)
───────────────────────              ──────────────────
1. Run openpilot in shadow mode
2. ComparisonLogger captures frames
3. End drive, sync logs
                    ─────────────▶
                                     4. import_shadow_segment()
                                     5. ScenarioRunner.run(algorithm, scenario)
                                     6. compare_shadow_to_harness()
                                     7. Generate comparison report
```

This enables:
- Validating algorithm changes against real-world data
- Regression testing on captured scenarios
- Comparing different algorithm variants on identical inputs

## Troubleshooting

### Shadow mode not activating
- Check panda is disconnected: `lsusb | grep -i panda`
- Verify environment: `echo $SHADOW_MODE`
- Check logs for "SHADOW MODE" warnings

### Alignment quality poor
- Ensure GPS is active on both devices
- Verify cameras have same view
- Check time synchronization

### Missing frames
- Check disk space: `df -h /data`
- Verify logger.end_segment() is called
- Check for exceptions in logs

## File Format

Logs are stored as JSON (optionally gzip compressed):

```
/data/shadow_logs/
  segment_001/
    shadow_0000.json.gz
    shadow_0001.json.gz
    ...
```

Each file contains:
```json
{
  "version": 1,
  "device_id": "shadow",
  "segment_id": "segment_001",
  "file_index": 0,
  "frame_count": 6000,
  "frames": [...]
}
```
