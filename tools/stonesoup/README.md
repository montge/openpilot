# Stone Soup Tracking Algorithm Comparison

Tools for comparing openpilot's tracking algorithms against the [Stone Soup](https://stonesoup.readthedocs.io/) library and implementing experimental tracking approaches.

## Overview

The Stone Soup comparison framework provides:

1. **Filter Comparison** - Compare KF1D against Stone Soup's Kalman variants
2. **Multi-Target Tracking** - JPDA and GNN trackers with MOTA/MOTP metrics
3. **Track Fusion** - Covariance Intersection for radar + vision fusion
4. **Voxel Tracking** - Log-odds occupancy grid with sparse representation
5. **Viterbi Association** - HMM-based track association for occlusion handling
6. **Octree Index** - Efficient spatial queries for large point sets

## Installation

Stone Soup is an optional development dependency:

```bash
pip install stonesoup
```

## Quick Start

```python
# Filter comparison
from openpilot.tools.stonesoup.comparison import (
    create_constant_velocity_scenario,
    compare_filters,
    format_comparison_report
)

scenario = create_constant_velocity_scenario(dt=0.05, duration=5.0)
metrics = compare_filters(scenario)
print(format_comparison_report(scenario, metrics))

# Multi-target tracking
from openpilot.tools.stonesoup.multi_target import (
    create_highway_scenario,
    compare_multi_target_trackers,
    format_tracking_report
)

scenario = create_highway_scenario(dt=0.1, duration=10.0, n_vehicles=3)
metrics = compare_multi_target_trackers(scenario)
print(format_tracking_report(scenario, metrics))

# Track fusion
from openpilot.tools.stonesoup.track_fusion import (
    create_fusion_scenario,
    compare_fusion_methods,
    format_fusion_report
)

scenario = create_fusion_scenario(dt=0.05, duration=5.0)
metrics = compare_fusion_methods(scenario)
print(format_fusion_report(scenario, metrics))
```

## Modules

### adapters.py

Type conversion between openpilot and Stone Soup data structures:

```python
from openpilot.tools.stonesoup.adapters import OpenpilotAdapter

adapter = OpenpilotAdapter()
detection = adapter.radar_point_to_detection(radar_point, timestamp)
lead_dict = adapter.gaussian_state_to_lead_dict(state)
```

### comparison.py

Single-target filter comparison harness:

- `KF1DWrapper` - openpilot's production Kalman filter
- `StoneSoupKalmanWrapper` - Stone Soup filters (KF, EKF, UKF, CKF)
- `ParticleFilterWrapper` - Particle filter with resampling

### multi_target.py

Multi-object tracking comparison:

- `JPDATrackerWrapper` - Joint Probabilistic Data Association
- `GNNTrackerWrapper` - Global Nearest Neighbor
- Metrics: MOTA, MOTP, ID switches, false positives/negatives

### track_fusion.py

Sensor fusion using Covariance Intersection:

```python
from openpilot.tools.stonesoup.track_fusion import TrackFusionEngine, RadarTrack, VisionTrack

engine = TrackFusionEngine()
fused = engine.fuse_tracks(radar_track, vision_track)
```

### voxel_tracker.py

Occupancy grid tracking:

```python
from openpilot.tools.stonesoup.voxel_tracker import VoxelTracker, VoxelGridConfig

config = VoxelGridConfig(resolution=0.5)
tracker = VoxelTracker(config, use_sparse=True)
result = tracker.process_detections(timestamp, points)
```

- `VoxelGrid` - Dense 3D grid with log-odds updates
- `SparseVoxelGrid` - Memory-efficient dict-based representation (99.95% memory reduction)
- `GPUVoxelGrid` - CuPy-accelerated batch updates

### viterbi_tracker.py

HMM-based track association:

```python
from openpilot.tools.stonesoup.viterbi_tracker import ViterbiTracker, Detection

tracker = ViterbiTracker()
tracks = tracker.process_detections(timestamp, detections)
```

- Sliding window Viterbi decoding
- Better handling of temporary occlusions than frame-by-frame matching

### octree.py

Spatial index for 3D point queries:

```python
from openpilot.tools.stonesoup.octree import Octree, BoundingBox
import numpy as np

bounds = BoundingBox(
    min_corner=np.array([0.0, -30.0, -5.0]),
    max_corner=np.array([100.0, 30.0, 5.0])
)
tree = Octree(bounds)
tree.insert_points(points)

neighbors = tree.k_nearest(query_point, k=5)
in_range = tree.radius_query(query_point, radius=10.0)
```

- 7.6x speedup for radius queries vs brute force
- 4.2x speedup for K-NN queries

### scenarios.py

Standardized benchmark scenarios:

- `create_highway_following()` - Steady-state tracking
- `create_cut_in()` - Lane change handling
- `create_cut_out()` - Revealed vehicle tracking
- `create_multi_vehicle()` - Cluttered environment
- `create_occlusion()` - Detection gaps
- `create_adverse_weather()` - Degraded sensor performance

## Running Benchmarks

```bash
# Run filter comparison
python -m openpilot.tools.stonesoup.comparison

# Run multi-target tracking benchmark
python -m openpilot.tools.stonesoup.multi_target

# Run track fusion benchmark
python -m openpilot.tools.stonesoup.track_fusion

# Run voxel grid benchmark
python -m openpilot.tools.stonesoup.voxel_tracker

# Run octree benchmark
python -m openpilot.tools.stonesoup.octree

# Run scenario summary
python -m openpilot.tools.stonesoup.scenarios
```

## Algorithm Selection Guide

| Use Case | Recommended Algorithm |
|----------|----------------------|
| Single lead vehicle | KF1D (production) |
| Multiple vehicles | JPDA or GNN |
| Radar + vision fusion | Covariance Intersection |
| Dense point clouds | Sparse Voxel Grid |
| Long occlusions | Viterbi Tracker |
| Spatial queries | Octree |

## Running Tests

```bash
pytest tools/stonesoup/tests/
```

## Dependencies

Required:
- numpy
- scipy

Optional:
- stonesoup (for Stone Soup filter comparison)
- cupy (for GPU voxel grid acceleration)
