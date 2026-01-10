## 1. Core Framework

- [x] 1.1 Create `selfdrive/controls/lib/tests/algorithm_harness/` directory structure
- [x] 1.2 Implement `AlgorithmInterface` protocol and data classes (`interface.py`)
- [x] 1.3 Implement `ScenarioRunner` class (`runner.py`)
- [x] 1.4 Implement `MetricsCollector` class (`metrics.py`)
- [x] 1.5 Create wrapper adapters for existing controllers (LatControlPID, LatControlTorque, LongControl)
- [x] 1.6 Add unit tests for harness framework (>90% coverage)

## 2. Scenario Infrastructure

- [x] 2.1 Define Parquet schema for scenarios (`scenario_schema.py`)
- [x] 2.2 Implement scenario loader and validator (`scenarios.py`)
- [x] 2.3 Create `tools/lib/test_scenarios/` directory structure
- [x] 2.4 Create 5 seed scenarios:
  - [x] 2.4.1 Highway straight driving (baseline)
  - [x] 2.4.2 Tight S-curve
  - [x] 2.4.3 Highway lane change
  - [x] 2.4.4 Low-speed parking maneuver
  - [x] 2.4.5 Emergency stop scenario
- [x] 2.5 Add scenario generation utilities from route logs

## 3. Vehicle Dynamics

- [x] 3.1 Create `VehicleDynamicsConfig` dataclass
- [x] 3.2 Create `BicycleModel` for vehicle dynamics simulation
- [x] 3.3 Add preset configurations (sedan, SUV, truck, compact, sports)
- [x] 3.4 Add unit tests for vehicle dynamics model

## 4. Deterministic Replay

- [ ] 4.1 Add `--deterministic` flag to process_replay
- [ ] 4.2 Implement fixed random seed injection
- [ ] 4.3 Implement monotonic fake time source
- [ ] 4.4 Add regression tests for determinism

## 5. CLI and Reporting

- [x] 5.1 Create `tools/algo_bench.py` CLI entry point
- [x] 5.2 Implement `run` subcommand (run algorithm against scenarios)
- [x] 5.3 Implement `compare` subcommand (A/B comparison)
- [ ] 5.4 Implement `report` subcommand (generate HTML/markdown reports)
- [x] 5.5 Add tabular output with metrics summary
- [ ] 5.6 Add optional matplotlib visualization

## 6. Pytest Integration

- [x] 6.1 Create pytest fixtures for algorithm harness (`conftest.py`)
- [x] 6.2 Add `@pytest.mark.algorithm_benchmark` marker
- [x] 6.3 Document usage patterns in docstrings
- [x] 6.4 Add example tests demonstrating harness usage

## 7. Documentation

- [x] 7.1 Add README.md to algorithm_harness directory
- [x] 7.2 Document scenario format specification
- [ ] 7.3 Add contributing guidelines for new scenarios
- [ ] 7.4 Add example notebook for algorithm analysis

## 8. Coverage Enforcement

- [x] 8.1 Add coverage check script (`check_coverage.py`)
- [x] 8.2 Add algorithm_benchmark pytest markers to pyproject.toml
- [x] 8.3 Create OpenSpec for test coverage requirements
- [ ] 8.4 Add coverage gate to CI workflow for algorithm harness
