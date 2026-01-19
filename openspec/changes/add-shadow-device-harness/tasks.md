## 1. Shadow Mode Infrastructure

- [x] 1.1 Add shadow mode detection (`system/hardware/shadow_mode.py`)
  - [x] 1.1.1 Detect OnePlus 6 device type
  - [x] 1.1.2 Check for panda connection
  - [x] 1.1.3 Support `SHADOW_MODE` environment override
- [x] 1.2 Add `is_shadow_mode()` to hardware abstraction
- [x] 1.3 Add shadow mode to system startup logging
- [x] 1.4 Add unit tests for shadow mode detection

## 2. Actuator Lockout

- [x] 2.1 Disable actuator publishing in `controlsd.py` when shadow mode
- [x] 2.2 Add actuator lockout in `card.py` (defense in depth)
- [x] 2.3 Log "SHADOW MODE - NO ACTUATION" warning at startup
- [x] 2.4 Add shadow mode indicator to UI (if UI running)
- [x] 2.5 Add integration tests verifying no CAN writes in shadow mode

## 3. Comparison Logger

- [x] 3.1 Create `tools/shadow/` directory structure
- [x] 3.2 Implement `ComparisonLogger` class
  - [x] 3.2.1 Capture model outputs per frame
  - [x] 3.2.2 Capture planned trajectories
  - [x] 3.2.3 Capture control commands (steer, accel)
  - [x] 3.2.4 Capture events and state
- [x] 3.3 Implement msgpack serialization for logs
- [x] 3.4 Add GPS timestamp capture for synchronization
- [x] 3.5 Implement log rotation and compression
- [x] 3.6 Add unit tests for comparison logger

## 4. Log Synchronization

- [x] 4.1 Implement log alignment algorithm (`tools/shadow/align.py`)
  - [x] 4.1.1 GPS-based time alignment
  - [x] 4.1.2 Frame ID matching
  - [x] 4.1.3 Interpolation for mismatched timestamps
- [x] 4.2 Implement log merging for paired logs
- [x] 4.3 Add validation for alignment quality
- [x] 4.4 Add unit tests for alignment

## 5. Analysis Tools

- [x] 5.1 Implement divergence metrics (`tools/shadow/metrics.py`)
  - [x] 5.1.1 Model output divergence (cosine similarity, RMSE)
  - [x] 5.1.2 Trajectory divergence (path error, speed error)
  - [x] 5.1.3 Control command divergence (steer error, accel error)
- [x] 5.2 Implement visualization (`tools/shadow/visualize.py`)
  - [x] 5.2.1 Time-series plots of divergence
  - [x] 5.2.2 Heatmaps of control differences
  - [x] 5.2.3 Event timeline comparison
- [x] 5.3 Create CLI tool (`tools/shadow/analyze.py`)
  - [x] 5.3.1 Load and align logs
  - [x] 5.3.2 Compute metrics
  - [x] 5.3.3 Generate report (markdown/HTML)
- [x] 5.4 Add integration tests with sample logs

## 6. OnePlus 6 Setup

> **Note**: Moved to separate change `document-oneplus6-lineageos-setup`

- [x] 6.1 Document OnePlus 6 flashing procedure for LineageOS
- [x] 6.2 Document proot Ubuntu setup for openpilot
- [x] 6.3 Document camera integration options
- [x] 6.4 Document thermal management (monitoring via sensors)
- [x] 6.5 Test and document power options (USB-C)

## 7. Mount Hardware

- [ ] 7.1 Design piggyback mount bracket (STL files)
- [ ] 7.2 Document mounting procedure
- [ ] 7.3 Design split mount option (research-grade)
- [ ] 7.4 Document camera alignment verification

## 8. Integration with Algorithm Harness

- [x] 8.1 Add shadow log import to algorithm harness scenarios
  - [x] `shadow_import.py` with frame_to_lateral_state, frame_to_longitudinal_state
  - [x] `import_shadow_log`, `import_shadow_segment`, `import_shadow_segments`
- [x] 8.2 Enable replay of shadow logs through harness
  - [x] Convert FrameData to LateralAlgorithmState/LongitudinalAlgorithmState
  - [x] Create Scenario objects with ground truth from actual outputs
- [x] 8.3 Add comparison metrics to harness reporting
  - [x] `compare_shadow_to_harness` function with RMSE, MAE, correlation
  - [x] `format_shadow_comparison_report` for markdown output
- [x] 8.4 Document workflow: shadow capture → harness analysis

## 9. Documentation

- [x] 9.1 Add README.md to tools/shadow/
- [x] 9.2 Document complete shadow device setup guide (in tools/shadow/setup/)
- [x] 9.3 Document comparison testing workflow
- [x] 9.4 Add troubleshooting guide (in tools/shadow/setup/TROUBLESHOOTING.md)
- [x] 9.5 Add example analysis notebook

## Summary

**Completed**: Shadow mode detection, actuator lockout (including UI indicator), comparison logging, log alignment, metrics, CLI tools, visualization tools (including event timeline), OnePlus 6 setup documentation, algorithm harness integration, example notebook.

**Remaining**: Mount hardware design (7.1-7.4).
