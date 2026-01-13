# Stone Soup Comparison Tools

Tools for comparing openpilot's tracking algorithms against Stone Soup and other baseline implementations.

## Overview

This module provides:

1. **Stone Soup Adapters** - Type conversions between openpilot and Stone Soup
2. **Alternative Trackers** - Voxel grid, Viterbi tracker, and octree implementations
3. **Benchmark Scenarios** - Standardized driving scenarios for algorithm comparison

## Installation

Stone Soup is an optional dependency:

```bash
pip install stonesoup
```

The adapters work without Stone Soup installed for basic openpilot types.

## Components

### Adapters (`adapters.py`)

Convert between openpilot and Stone Soup data types:

```python
from openpilot.tools.stonesoup.adapters import (
    OpenpilotAdapter,
    RadarDetection,
    LeadData,
    PoseState,
)

adapter = OpenpilotAdapter()

# Convert radar detection to Stone Soup
radar = RadarDetection(d_rel=50.0, v_rel=-5.0, a_rel=0.0, y_rel=0.0, timestamp=100.0)
ss_detection = adapter.radar_to_stonesoup(radar)

# Convert Stone Soup state back to openpilot
lead_data = adapter.stonesoup_to_lead(ss_state)
```

### Alternative Trackers (`selfdrive/controls/lib/trackers/`)

#### VoxelGrid

3D occupancy grid for environment representation:

```python
from openpilot.selfdrive.controls.lib.trackers import VoxelGrid, VoxelGridConfig

config = VoxelGridConfig(resolution=0.5)
grid = VoxelGrid(config)

# Update with point cloud
points = np.array([[10, 0, 0], [20, 1, 0], ...])
grid.update_with_points(points, origin=np.zeros(3))

# Query occupancy
if grid.is_occupied(10.0, 0.0, 0.0):
    print("Occupied!")

# Get all occupied voxels
occupied = grid.get_occupied_voxels()
```

#### ViterbiTracker

Multi-object tracker using HMM and Viterbi decoding for globally optimal track association:

```python
from openpilot.selfdrive.controls.lib.trackers import (
    ViterbiTracker,
    ViterbiConfig,
    Detection,
)

config = ViterbiConfig(window_size=5, min_hits_to_confirm=3)
tracker = ViterbiTracker(config)

# Update with detections each frame
for frame_detections in detection_stream:
    dets = [Detection(measurement=np.array([d.x, d.y])) for d in frame_detections]
    confirmed_tracks = tracker.update(dets)

    for track in confirmed_tracks:
        print(f"Track {track.id}: pos=({track.state[0]:.1f}, {track.state[2]:.1f})")
```

#### Octree

Spatial index for efficient 3D point queries:

```python
from openpilot.selfdrive.controls.lib.trackers import (
    Octree,
    BoundingBox,
    create_octree_from_bounds,
)

# Create octree
tree = create_octree_from_bounds(
    x_min=-50, x_max=150,
    y_min=-25, y_max=25,
    z_min=-2, z_max=8,
)

# Insert points
for point in points:
    tree.insert(point)

# Range query
query = BoundingBox(min_corner=np.array([0, -5, -1]), max_corner=np.array([50, 5, 3]))
found = tree.query_range(query)

# K-nearest neighbors
neighbors = tree.query_knn(np.array([10, 0, 1]), k=5)

# Radius search
nearby = tree.query_radius(np.array([10, 0, 1]), radius=10.0)
```

### Benchmark Scenarios (`benchmarks/`)

Standardized driving scenarios for testing tracking algorithms:

```python
from openpilot.tools.stonesoup.benchmarks import (
    create_highway_scenario,
    create_cut_in_scenario,
    create_cut_out_scenario,
    create_multi_vehicle_scenario,
    create_occlusion_scenario,
    create_noisy_scenario,
)

# Highway following
scenario = create_highway_scenario(
    duration=30.0,
    lead_distance=50.0,
    ego_velocity=28.0,
    noise_std=0.5,
)

# Access data
for frame_idx in range(scenario.n_frames):
    detections = scenario.get_all_detections_at(frame_idx)
    ego_state = scenario.ego_trajectory[frame_idx]

    for det in detections:
        print(f"Detection: d={det.d_rel:.1f}m, v={det.v_rel:.1f}m/s")
```

Available scenarios:
- **Highway Following**: Simple constant-distance lead tracking
- **Cut-In**: Vehicle enters from adjacent lane
- **Cut-Out**: Lead leaves lane, revealing slower vehicle
- **Multi-Vehicle**: Multiple vehicles in sensor view
- **Occlusion**: Lead temporarily occluded
- **Noisy**: High noise simulating adverse weather

## Algorithm Selection Guide

| Scenario | Recommended Algorithm |
|----------|----------------------|
| Single lead, low noise | KF1D (openpilot default) |
| Multi-target | ViterbiTracker or JPDA |
| High noise / false alarms | Particle filter or IMM |
| Occlusions | ViterbiTracker (temporal consistency) |
| Dense point clouds | VoxelGrid + Octree for queries |

## Running Tests

```bash
# Adapter tests
pytest tools/stonesoup/tests/

# Tracker tests
pytest selfdrive/controls/lib/trackers/tests/

# Benchmark tests
pytest tools/stonesoup/benchmarks/tests/
```

## Integration with Existing Code

The trackers are designed for experimentation and comparison, not production use. They provide:

1. Reference implementations for algorithm comparison
2. Baseline metrics for evaluating improvements
3. Test infrastructure for tracking algorithm development

For production, openpilot continues to use optimized implementations in `selfdrive/controls/`.
