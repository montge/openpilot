## 1. Shadow Mode Infrastructure

- [ ] 1.1 Add shadow mode detection (`system/hardware/shadow_mode.py`)
  - [ ] 1.1.1 Detect OnePlus 6 device type
  - [ ] 1.1.2 Check for panda connection
  - [ ] 1.1.3 Support `SHADOW_MODE` environment override
- [ ] 1.2 Add `is_shadow_mode()` to hardware abstraction
- [ ] 1.3 Add shadow mode to system startup logging
- [ ] 1.4 Add unit tests for shadow mode detection

## 2. Actuator Lockout

- [ ] 2.1 Disable actuator publishing in `controlsd.py` when shadow mode
- [ ] 2.2 Add actuator lockout in `pandad.py` (defense in depth)
- [ ] 2.3 Log "SHADOW MODE - NO ACTUATION" warning at startup
- [ ] 2.4 Add shadow mode indicator to UI (if UI running)
- [ ] 2.5 Add integration tests verifying no CAN writes in shadow mode

## 3. Comparison Logger

- [ ] 3.1 Create `tools/shadow/` directory structure
- [ ] 3.2 Implement `ComparisonLogger` class
  - [ ] 3.2.1 Capture model outputs per frame
  - [ ] 3.2.2 Capture planned trajectories
  - [ ] 3.2.3 Capture control commands (steer, accel)
  - [ ] 3.2.4 Capture events and state
- [ ] 3.3 Implement msgpack serialization for logs
- [ ] 3.4 Add GPS timestamp capture for synchronization
- [ ] 3.5 Implement log rotation and compression
- [ ] 3.6 Add unit tests for comparison logger

## 4. Log Synchronization

- [ ] 4.1 Implement log alignment algorithm (`tools/shadow/align.py`)
  - [ ] 4.1.1 GPS-based time alignment
  - [ ] 4.1.2 Frame ID matching
  - [ ] 4.1.3 Interpolation for mismatched timestamps
- [ ] 4.2 Implement log merging for paired logs
- [ ] 4.3 Add validation for alignment quality
- [ ] 4.4 Add unit tests for alignment

## 5. Analysis Tools

- [ ] 5.1 Implement divergence metrics (`tools/shadow/metrics.py`)
  - [ ] 5.1.1 Model output divergence (cosine similarity, RMSE)
  - [ ] 5.1.2 Trajectory divergence (path error, speed error)
  - [ ] 5.1.3 Control command divergence (steer error, accel error)
- [ ] 5.2 Implement visualization (`tools/shadow/visualize.py`)
  - [ ] 5.2.1 Time-series plots of divergence
  - [ ] 5.2.2 Heatmaps of control differences
  - [ ] 5.2.3 Event timeline comparison
- [ ] 5.3 Create CLI tool (`tools/shadow/analyze.py`)
  - [ ] 5.3.1 Load and align logs
  - [ ] 5.3.2 Compute metrics
  - [ ] 5.3.3 Generate report (markdown/HTML)
- [ ] 5.4 Add integration tests with sample logs

## 6. OnePlus 6 Setup

- [ ] 6.1 Document OnePlus 6 flashing procedure for AGNOS
- [ ] 6.2 Document camera calibration for shadow mount
- [ ] 6.3 Create calibration validation tool
- [ ] 6.4 Document thermal management (heat sink, throttling)
- [ ] 6.5 Test and document power options (USB-C, 12V adapter)

## 7. Mount Hardware

- [ ] 7.1 Design piggyback mount bracket (STL files)
- [ ] 7.2 Document mounting procedure
- [ ] 7.3 Design split mount option (research-grade)
- [ ] 7.4 Document camera alignment verification

## 8. Integration with Algorithm Harness

- [ ] 8.1 Add shadow log import to algorithm harness scenarios
- [ ] 8.2 Enable replay of shadow logs through harness
- [ ] 8.3 Add comparison metrics to harness reporting
- [ ] 8.4 Document workflow: shadow capture → harness analysis

## 9. Documentation

- [ ] 9.1 Add README.md to tools/shadow/
- [ ] 9.2 Document complete shadow device setup guide
- [ ] 9.3 Document comparison testing workflow
- [ ] 9.4 Add troubleshooting guide
- [ ] 9.5 Add example analysis notebook
