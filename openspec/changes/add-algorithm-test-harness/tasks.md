## 1. Core Framework

- [ ] 1.1 Create `selfdrive/controls/lib/tests/algorithm_harness/` directory structure
- [ ] 1.2 Implement `AlgorithmInterface` protocol and data classes (`interface.py`)
- [ ] 1.3 Implement `ScenarioRunner` class (`runner.py`)
- [ ] 1.4 Implement `MetricsCollector` class (`metrics.py`)
- [ ] 1.5 Create wrapper adapters for existing controllers (LatControlPID, LatControlTorque, LongControl)
- [ ] 1.6 Add unit tests for harness framework (>90% coverage)

## 2. Scenario Infrastructure

- [ ] 2.1 Define Parquet schema for scenarios (`scenario_schema.py`)
- [ ] 2.2 Implement scenario loader and validator (`scenarios.py`)
- [ ] 2.3 Create `tools/lib/test_scenarios/` directory structure
- [ ] 2.4 Create 5 seed scenarios:
  - [ ] 2.4.1 Highway straight driving (baseline)
  - [ ] 2.4.2 Tight S-curve
  - [ ] 2.4.3 Highway lane change
  - [ ] 2.4.4 Low-speed parking maneuver
  - [ ] 2.4.5 Emergency stop scenario
- [ ] 2.5 Add scenario generation utilities from route logs

## 3. Vehicle Dynamics

- [ ] 3.1 Create `VehicleDynamicsConfig` dataclass
- [ ] 3.2 Extend `SimulatedCar` with configurable dynamics
- [ ] 3.3 Add preset configurations (sedan, SUV, truck)
- [ ] 3.4 Add unit tests for vehicle dynamics model

## 4. Deterministic Replay

- [ ] 4.1 Add `--deterministic` flag to process_replay
- [ ] 4.2 Implement fixed random seed injection
- [ ] 4.3 Implement monotonic fake time source
- [ ] 4.4 Add regression tests for determinism

## 5. CLI and Reporting

- [ ] 5.1 Create `tools/algo_bench.py` CLI entry point
- [ ] 5.2 Implement `run` subcommand (run algorithm against scenarios)
- [ ] 5.3 Implement `compare` subcommand (A/B comparison)
- [ ] 5.4 Implement `report` subcommand (generate HTML/markdown reports)
- [ ] 5.5 Add tabular output with metrics summary
- [ ] 5.6 Add optional matplotlib visualization

## 6. Pytest Integration

- [ ] 6.1 Create pytest fixtures for algorithm harness (`conftest.py`)
- [ ] 6.2 Add `@pytest.mark.algorithm_benchmark` marker
- [ ] 6.3 Document usage patterns in docstrings
- [ ] 6.4 Add example tests demonstrating harness usage

## 7. Documentation

- [ ] 7.1 Add README.md to algorithm_harness directory
- [ ] 7.2 Document scenario format specification
- [ ] 7.3 Add contributing guidelines for new scenarios
- [ ] 7.4 Add example notebook for algorithm analysis
