## 1. Stone Soup Integration

- [x] 1.1 Add stonesoup to development dependencies
- [x] 1.2 Create `tools/stonesoup/` directory structure
- [x] 1.3 Implement `OpenpilotAdapter` for type conversion
- [x] 1.4 Implement radar track → Stone Soup Detection conversion
- [x] 1.5 Implement Stone Soup state → openpilot LeadData conversion
- [x] 1.6 Implement pose state conversions for localization comparison
- [x] 1.7 Add unit tests for adapters

## 2. Baseline Algorithm Comparison

- [x] 2.1 Create comparison harness using algorithm test framework
- [x] 2.2 Implement KF1D vs Stone Soup KalmanFilter comparison
- [x] 2.3 Implement KF1D vs ExtendedKalmanFilter comparison
- [x] 2.4 Implement KF1D vs UnscentedKalmanFilter comparison
- [x] 2.5 Implement KF1D vs CubatureKalmanFilter comparison
- [x] 2.6 Implement KF1D vs ParticleFilter comparison
- [x] 2.7 Generate comparison report with metrics

## 3. Multi-Target Tracking Comparison

- [x] 3.1 Implement JPDATracker integration
- [x] 3.2 Implement GNNTracker (MHT) integration
- [x] 3.3 Create multi-vehicle tracking scenarios
- [x] 3.4 Benchmark MOTA, MOTP, ID switches metrics
- [x] 3.5 Compare vs current probabilistic matching

## 4. Track-to-Track Fusion

- [x] 4.1 Implement Covariance Intersection adapter
- [x] 4.2 Create radar + vision fusion test scenarios
- [x] 4.3 Benchmark fusion accuracy vs current approach
- [x] 4.4 Document fusion latency characteristics

## 5. Voxel Grid Tracker

- [x] 5.1 Create `tools/stonesoup/voxel_tracker.py`
- [x] 5.2 Implement `VoxelGrid` class
- [x] 5.3 Implement log-odds occupancy update
- [x] 5.4 Implement voxel-to-world coordinate conversion
- [x] 5.5 Add sparse voxel representation for memory efficiency
- [x] 5.6 Add GPU acceleration option (for DGX Spark)
- [x] 5.7 Create benchmark scenarios for voxel tracking
- [x] 5.8 Add unit tests for voxel grid

## 6. Viterbi Track Association

- [ ] 6.1 Implement `ViterbiTracker` class
- [ ] 6.2 Implement HMM transition matrix computation
- [ ] 6.3 Implement emission probability (Mahalanobis likelihood)
- [ ] 6.4 Implement Viterbi decoding algorithm
- [ ] 6.5 Benchmark vs Hungarian algorithm baseline
- [ ] 6.6 Create occlusion handling scenarios
- [ ] 6.7 Add unit tests for Viterbi tracker

## 7. Octree Spatial Index

- [ ] 7.1 Implement `Octree` class
- [ ] 7.2 Implement recursive insert/query operations
- [ ] 7.3 Implement range query optimization
- [ ] 7.4 Implement k-nearest-neighbor query
- [ ] 7.5 Benchmark query performance vs brute force
- [ ] 7.6 Add unit tests for octree operations

## 8. Benchmark Scenarios

- [ ] 8.1 Create highway following scenario
- [ ] 8.2 Create cut-in scenario
- [ ] 8.3 Create cut-out scenario
- [ ] 8.4 Create multi-vehicle scenario
- [ ] 8.5 Create occlusion scenario
- [ ] 8.6 Create adverse weather scenario (simulated sensor noise)

## 9. Documentation and Recommendations

- [ ] 9.1 Generate benchmark results report
- [ ] 9.2 Create algorithm selection decision tree
- [ ] 9.3 Document production integration recommendations
- [ ] 9.4 Add README with usage examples
- [ ] 9.5 Create visualization notebooks for results
