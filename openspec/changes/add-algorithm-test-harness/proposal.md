# Change: Add Algorithm Test Harness for Hardware-Free Development

## Why

Developers without comma device hardware need a comprehensive test harness to experiment with, validate, and benchmark control algorithms. While openpilot has excellent existing test infrastructure (OpenpilotPrefix, process_replay, SimulatedCar), there's no unified framework specifically designed for algorithm experimentation that provides:
- Standardized algorithm benchmarking with synthetic and recorded scenarios
- Easy A/B comparison between algorithm variants
- Deterministic replay for regression testing
- Scenario-based testing with edge cases (high curvature, adverse conditions)

## What Changes

- **Add Algorithm Test Framework** (`selfdrive/controls/lib/tests/algorithm_harness/`):
  - Scenario runner for deterministic algorithm evaluation
  - Metrics collection (latency, smoothness, tracking error, safety margins)
  - A/B comparison utilities for algorithm variants
  - Synthetic scenario generators (curves, lane changes, emergency stops)

- **Add Recorded Scenario Library** (`tools/lib/test_scenarios/`):
  - Curated set of challenging driving scenarios from route logs
  - Ground truth annotations for algorithm validation
  - Scenario categories: curves, lane changes, traffic, weather conditions

- **Add Algorithm Benchmarking CLI** (`tools/algo_bench.py`):
  - Run algorithms against scenario library
  - Generate comparison reports (tables, plots)
  - Track algorithm performance over commits

- **Enhance Existing Infrastructure**:
  - Extend `SimulatedCar` with configurable vehicle dynamics
  - Add deterministic mode to process_replay for reproducible tests
  - Create pytest fixtures for algorithm testing patterns

## Impact

- Affected specs: New `algorithm-test-harness` capability
- Affected code:
  - `selfdrive/controls/lib/tests/` - New harness module
  - `tools/` - New benchmarking CLI
  - `tools/sim/lib/simulated_car.py` - Extended dynamics
  - `selfdrive/test/process_replay/` - Deterministic mode
- Dependencies: numpy, matplotlib (existing), hypothesis (existing)
- **No changes to safety-critical code paths**
