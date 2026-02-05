# Algorithm Comparison

**Purpose:** Define a framework for comparing openpilot's tracking algorithms against Stone Soup reference implementations, including filter variants, multi-target trackers, voxel-based 3D tracking, HMM-based association, and track fusion methods.

## Requirements

### Requirement: Stone Soup Integration
The system SHALL provide an adapter layer to interface with the Stone Soup tracking framework.

#### Scenario: Convert openpilot radar track to Stone Soup detection
- **GIVEN** a radar track from openpilot's RadarState message
- **WHEN** the adapter converts to Stone Soup format
- **THEN** a Detection object is created with position and velocity
- **AND** timestamp is correctly mapped
- **AND** metadata preserves additional track attributes

#### Scenario: Convert Stone Soup state estimate to openpilot format
- **GIVEN** a GaussianState from Stone Soup tracker
- **WHEN** the adapter converts to openpilot format
- **THEN** a LeadData message is created with dRel, vRel, aRel
- **AND** uncertainty information is preserved where applicable

### Requirement: Algorithm Comparison Framework
The system SHALL provide a framework for comparing tracking algorithms.

#### Scenario: Compare Kalman filter variants
- **GIVEN** the same input scenario
- **WHEN** KF1D, EKF, UKF, CKF, and ParticleFilter are executed
- **THEN** metrics are collected for each algorithm
- **AND** a comparison report shows relative performance

#### Scenario: Compare multi-target tracking algorithms
- **GIVEN** a multi-vehicle scenario
- **WHEN** JPDA and MHT trackers are compared against current approach
- **THEN** MOTA, MOTP, and ID switch metrics are computed
- **AND** computational cost is measured

### Requirement: Voxel Grid Tracker
The system SHALL provide a voxel-based 3D occupancy tracker.

#### Scenario: Update voxel grid from detections
- **GIVEN** a set of 3D detection points
- **WHEN** the voxel grid is updated
- **THEN** occupancy probabilities increase for detected voxels
- **AND** probabilities decay for unobserved voxels (free space)

#### Scenario: Query occupied regions
- **GIVEN** a voxel grid with updated occupancy
- **WHEN** occupied voxels are queried above threshold
- **THEN** world coordinates of occupied regions are returned
- **AND** query time is O(n) where n is number of occupied voxels

#### Scenario: Sparse representation for memory efficiency
- **GIVEN** a large spatial extent with sparse occupancy
- **WHEN** the voxel grid is stored
- **THEN** only occupied voxels consume significant memory
- **AND** empty regions use constant overhead

### Requirement: Viterbi Track Association
The system SHALL provide HMM-based track-detection association.

#### Scenario: Associate tracks with detections
- **GIVEN** N existing tracks and M new detections
- **WHEN** Viterbi association is computed
- **THEN** globally optimal assignment is returned
- **AND** association considers temporal consistency

#### Scenario: Handle missed detections
- **GIVEN** a track with no matching detection in current frame
- **WHEN** Viterbi decoding considers history
- **THEN** track is maintained based on motion model prediction
- **AND** track is terminated only after sustained misses

#### Scenario: Handle false alarms
- **GIVEN** detections with no matching tracks
- **WHEN** Viterbi decoding evaluates new track hypotheses
- **THEN** false alarms are filtered based on persistence criteria
- **AND** genuine new objects are tracked

### Requirement: Octree Spatial Index
The system SHALL provide octree-based spatial indexing for 3D queries.

#### Scenario: Insert objects into octree
- **GIVEN** a set of 3D objects with positions
- **WHEN** objects are inserted into the octree
- **THEN** each object is stored in the appropriate octant
- **AND** tree depth adapts to object density

#### Scenario: Range query on octree
- **GIVEN** an octree with inserted objects
- **WHEN** a bounding box query is executed
- **THEN** all objects intersecting the box are returned
- **AND** query time is O(log n + k) where k is result size

#### Scenario: Nearest neighbor query
- **GIVEN** an octree with inserted objects
- **WHEN** k-nearest-neighbor query is executed
- **THEN** k closest objects to query point are returned
- **AND** distance metric is configurable (Euclidean, Manhattan)

### Requirement: Track-to-Track Fusion
The system SHALL support Covariance Intersection for track fusion.

#### Scenario: Fuse radar and vision tracks
- **GIVEN** a radar track estimate and vision track estimate for same object
- **WHEN** Covariance Intersection fusion is applied
- **THEN** fused estimate is consistent (covariance is valid)
- **AND** fused estimate is at least as accurate as better input

#### Scenario: Handle unknown correlation
- **GIVEN** two track estimates with unknown cross-correlation
- **WHEN** Covariance Intersection fusion is applied
- **THEN** result is guaranteed consistent regardless of true correlation
- **AND** optimization finds optimal fusion weight

### Requirement: Benchmark Scenarios
The system SHALL provide standardized scenarios for algorithm comparison.

#### Scenario: Highway following benchmark
- **GIVEN** the highway following scenario
- **WHEN** tracking algorithms are evaluated
- **THEN** steady-state RMSE for position and velocity is computed
- **AND** algorithms are ranked by accuracy

#### Scenario: Cut-in benchmark
- **GIVEN** a vehicle cut-in scenario
- **WHEN** tracking algorithms are evaluated
- **THEN** time-to-track metric is computed
- **AND** false positive rate during cut-in is measured

#### Scenario: Multi-vehicle benchmark
- **GIVEN** a scenario with multiple vehicles
- **WHEN** tracking algorithms are evaluated
- **THEN** MOTA (Multi-Object Tracking Accuracy) is computed
- **AND** ID switches are counted

### Requirement: Comparison Reporting
The system SHALL generate comparison reports for algorithm evaluation.

#### Scenario: Generate metrics table
- **GIVEN** completed benchmark runs
- **WHEN** report generation is requested
- **THEN** a table with all algorithms and metrics is produced
- **AND** statistical significance indicators are included

#### Scenario: Generate recommendation
- **GIVEN** comparison results across scenarios
- **WHEN** recommendation is generated
- **THEN** best algorithm per use case is identified
- **AND** trade-offs (accuracy vs compute) are documented
