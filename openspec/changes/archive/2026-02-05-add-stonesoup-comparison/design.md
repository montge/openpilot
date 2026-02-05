## Context

openpilot's current tracking and filtering algorithms:

| Component | Algorithm | Location | Limitations |
|-----------|-----------|----------|-------------|
| Lead tracking | KF1D | radard.py | Single hypothesis, no multi-target |
| Pose estimation | EKF (rednose) | locationd.py | Linearization errors for large errors |
| Vehicle dynamics | EKF (rednose) | paramsd.py | Fixed process noise |
| Radar/vision fusion | Probabilistic matching | radard.py | No track-to-track fusion |

Stone Soup offers alternatives worth benchmarking, plus emerging approaches (voxel, Viterbi, octree) that address 3D spatial reasoning.

**Stakeholders**: Algorithm researchers, safety engineers, performance optimizers

## Goals / Non-Goals

**Goals**:
- Quantitatively compare openpilot algorithms vs Stone Soup alternatives
- Implement and evaluate voxel, Viterbi, and octree approaches
- Identify algorithms that improve tracking accuracy or robustness
- Provide clear recommendations for production integration paths

**Non-Goals**:
- Replace production algorithms immediately (research first)
- Achieve real-time performance on comma device (DGX Spark target)
- Certify Stone Soup for automotive safety

## Decisions

### Decision 1: Stone Soup Adapter Layer

Create bidirectional conversion between openpilot and Stone Soup types:

```python
# tools/stonesoup/adapters.py
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from cereal import log

class OpenpilotAdapter:
    def radar_track_to_detection(self, track: log.RadarState.RadarTrack) -> Detection:
        """Convert openpilot radar track to Stone Soup Detection."""
        return Detection(
            state_vector=StateVector([track.dRel, track.vRel]),
            timestamp=datetime.fromtimestamp(track.t),
            metadata={'yRel': track.yRel, 'aRel': track.aRel}
        )

    def gaussian_state_to_lead(self, state: GaussianState) -> log.RadarState.LeadData:
        """Convert Stone Soup state estimate to openpilot LeadData."""
        lead = log.RadarState.LeadData.new_message()
        lead.dRel = float(state.state_vector[0])
        lead.vRel = float(state.state_vector[1])
        return lead
```

**Rationale**: Clean separation allows testing Stone Soup algorithms with real openpilot data.

### Decision 2: Algorithm Comparison Matrix

Benchmark these algorithm pairs:

| openpilot | Stone Soup | Metric Focus |
|-----------|------------|--------------|
| KF1D | KalmanFilter | Baseline validation |
| KF1D | ExtendedKalmanFilter | Nonlinearity handling |
| KF1D | UnscentedKalmanFilter | Large uncertainty |
| KF1D | CubatureKalmanFilter | High-dimensional |
| KF1D | ParticleFilter | Multi-modal distributions |
| Probabilistic matching | JPDATracker | Multi-target association |
| Probabilistic matching | MHT (GNNTracker) | Track hypothesis management |
| N/A | CovarianceIntersection | Track-to-track fusion |

**Rationale**: Cover spectrum from simple to sophisticated to find sweet spot.

### Decision 3: Voxel Grid Tracker

Implement 3D occupancy grid for spatial reasoning:

```python
# selfdrive/controls/lib/trackers/voxel_tracker.py
@dataclass
class VoxelGrid:
    resolution: float = 0.5  # meters per voxel
    x_range: tuple = (-10, 100)  # meters
    y_range: tuple = (-20, 20)
    z_range: tuple = (-2, 4)

    def __post_init__(self):
        self.grid = np.zeros(self._compute_shape(), dtype=np.float32)

    def update_from_detections(self, detections: list[Detection3D]):
        """Update occupancy probabilities from sensor detections."""
        for det in detections:
            voxel_idx = self._world_to_voxel(det.position)
            self.grid[voxel_idx] = self._log_odds_update(
                self.grid[voxel_idx], det.confidence
            )

    def get_occupied_voxels(self, threshold: float = 0.5) -> np.ndarray:
        """Return world coordinates of occupied voxels."""
        occupied_idx = np.argwhere(self.grid > np.log(threshold / (1 - threshold)))
        return self._voxel_to_world(occupied_idx)
```

**Use cases**:
- Detecting stopped vehicles in blind spots
- Reasoning about occluded areas
- Path planning with spatial constraints

### Decision 4: Viterbi Track Association

Use Hidden Markov Model for track-detection association:

```python
# selfdrive/controls/lib/trackers/viterbi_tracker.py
class ViterbiTracker:
    """HMM-based track association using Viterbi decoding."""

    def __init__(self, n_tracks: int, n_detections: int):
        self.transition_prob = self._compute_transition_matrix(n_tracks)
        self.emission_prob = np.zeros((n_tracks, n_detections))

    def associate(self, tracks: list[Track], detections: list[Detection]) -> list[tuple]:
        """Find optimal track-detection association via Viterbi."""
        # Compute emission probabilities (likelihood of detection given track)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                self.emission_prob[i, j] = self._mahalanobis_likelihood(track, det)

        # Viterbi decoding
        path = self._viterbi_decode(self.transition_prob, self.emission_prob)
        return [(tracks[i], detections[path[i]]) for i in range(len(tracks))]
```

**Advantages over current approach**:
- Considers temporal consistency of associations
- Handles occlusions and missed detections naturally
- Provides globally optimal association

### Decision 5: Octree Spatial Index

Hierarchical spatial partitioning for efficient 3D queries:

```python
# selfdrive/controls/lib/trackers/octree.py
class Octree:
    """Octree for efficient spatial queries on 3D objects."""

    def __init__(self, bounds: BoundingBox, max_depth: int = 8):
        self.bounds = bounds
        self.max_depth = max_depth
        self.children = [None] * 8  # 8 octants
        self.objects = []

    def insert(self, obj: SpatialObject) -> None:
        """Insert object into appropriate octant."""
        if self._is_leaf() or self.depth >= self.max_depth:
            self.objects.append(obj)
        else:
            octant = self._get_octant(obj.position)
            if self.children[octant] is None:
                self.children[octant] = Octree(self._octant_bounds(octant))
            self.children[octant].insert(obj)

    def query_range(self, query_box: BoundingBox) -> list[SpatialObject]:
        """Find all objects intersecting query box."""
        results = []
        if not self.bounds.intersects(query_box):
            return results
        results.extend([o for o in self.objects if query_box.contains(o.position)])
        for child in self.children:
            if child:
                results.extend(child.query_range(query_box))
        return results
```

**Use cases**:
- Fast nearest-neighbor queries for radar/vision matching
- Collision checking for path planning
- Level-of-detail rendering for visualization

### Decision 6: Benchmark Scenarios

Use algorithm test harness scenarios for comparison:

| Scenario | Focus | Key Metrics |
|----------|-------|-------------|
| Highway following | Steady state tracking | RMSE position/velocity |
| Cut-in | Track initialization | Time to track, false positives |
| Cut-out | Track termination | Track lifetime, ghost tracks |
| Multi-vehicle | Association | MOTA, ID switches |
| Occlusion | Track persistence | Re-ID accuracy, track fragmentation |
| Adverse weather | Robustness | Performance degradation |

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Stone Soup API changes | Pin specific version; abstract adapter layer |
| Compute overhead of sophisticated algorithms | Benchmark on DGX Spark; optimize for comma later |
| Voxel memory consumption | Adaptive resolution; sparse representations |
| Octree insertion overhead | Batch insertions; lazy updates |

## Migration Plan

1. **Phase 1**: Stone Soup integration + baseline comparison
2. **Phase 2**: Voxel grid implementation + benchmarks
3. **Phase 3**: Viterbi tracker + octree implementation
4. **Phase 4**: Recommendations document with integration paths

## Open Questions

1. What Stone Soup version to target (latest vs LTS)?
2. Should voxel grid use GPU acceleration (via DGX Spark)?
3. How to handle Stone Soup's datetime-based timestamps vs openpilot's monotonic time?
