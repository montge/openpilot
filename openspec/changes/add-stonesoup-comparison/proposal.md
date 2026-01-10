# Change: Add Stone Soup Algorithm Comparison Framework

## Why

openpilot uses custom implementations of tracking and filtering algorithms (KF1D, EKF via rednose). The Stone Soup framework provides a comprehensive library of state-of-the-art tracking algorithms that could potentially improve:

1. **Lead vehicle tracking** - Current KF1D may underperform vs multi-hypothesis trackers
2. **Sensor fusion** - Stone Soup's track-to-track fusion (Covariance Intersection) could improve radar/vision fusion
3. **Multi-object tracking** - Current approach tracks leads independently; JPDA/MHT could improve
4. **Localization** - UKF/Cubature KF may handle nonlinearities better than EKF

Additionally, emerging algorithms not yet in Stone Soup warrant investigation:
- **Voxel-based tracking** - 3D occupancy grids for spatial reasoning
- **Viterbi-based tracking** - HMM decoding for track association
- **Octree approaches** - Hierarchical spatial partitioning for efficient 3D queries

## What Changes

- **Add Stone Soup Integration** (`tools/stonesoup/`):
  - Adapter layer to convert openpilot messages to Stone Soup types
  - Benchmarking harness using algorithm test framework
  - Side-by-side comparison utilities

- **Add Algorithm Implementations** (`selfdrive/controls/lib/trackers/`):
  - Voxel grid tracker for 3D occupancy
  - Viterbi decoder for track association
  - Octree spatial index for efficient queries

- **Add Comparison Benchmarks** (`tools/stonesoup/benchmarks/`):
  - EKF vs UKF vs Cubature KF for localization
  - KF1D vs JPDA vs MHT for multi-target tracking
  - Current radar fusion vs Covariance Intersection

- **Documentation**:
  - Algorithm comparison results
  - Recommendations for production integration

## Impact

- Affected specs: New `algorithm-comparison` capability
- Affected code:
  - `tools/stonesoup/` - New comparison framework
  - `selfdrive/controls/lib/trackers/` - New algorithm implementations
- Dependencies: stonesoup (new), scipy (existing), numpy (existing)
- **No changes to production code paths** (research/comparison only)
