# algorithm-test-harness Specification

## Purpose
Define the algorithm test harness framework for standardized testing, benchmarking, and comparison of control algorithms in openpilot.
## Requirements
### Requirement: Algorithm Interface Protocol
All control algorithms tested via the harness SHALL implement the `AlgorithmInterface` protocol.

#### Scenario: Algorithm implements required methods
- **GIVEN** a control algorithm class
- **WHEN** it is registered with the test harness
- **THEN** it MUST implement `reset()` and `update(state) -> output` methods
- **AND** the `update` method MUST accept `AlgorithmState` and return `AlgorithmOutput`

#### Scenario: Existing controllers wrapped for compatibility
- **GIVEN** an existing controller (LatControlPID, LatControlTorque, LongControl)
- **WHEN** tested via the algorithm harness
- **THEN** a wrapper adapter translates between native interface and `AlgorithmInterface`

### Requirement: Scenario Runner
The test harness SHALL provide a scenario runner that executes algorithms against recorded or synthetic scenarios.

#### Scenario: Run algorithm against scenario file
- **GIVEN** a Parquet scenario file and an algorithm implementing `AlgorithmInterface`
- **WHEN** the scenario runner executes
- **THEN** the algorithm's `update()` is called for each timestep in the scenario
- **AND** outputs and metrics are collected for analysis

#### Scenario: Deterministic execution
- **GIVEN** the same scenario and algorithm
- **WHEN** the scenario runner executes multiple times with deterministic mode enabled
- **THEN** the outputs are bit-exact identical across runs

### Requirement: Metrics Collection
The test harness SHALL collect standardized metrics for algorithm evaluation.

#### Scenario: Tracking error metrics calculated
- **GIVEN** a completed scenario run with ground truth data
- **WHEN** metrics are computed
- **THEN** `tracking_error_rmse` and `tracking_error_max` are calculated
- **AND** values are reported in the algorithm's native units

#### Scenario: Smoothness metrics calculated
- **GIVEN** a completed scenario run
- **WHEN** metrics are computed
- **THEN** `output_smoothness` is calculated as the RMS of output rate-of-change
- **AND** lower values indicate smoother control

#### Scenario: Latency metrics calculated
- **GIVEN** a completed scenario run
- **WHEN** metrics are computed
- **THEN** `latency_p50` and `latency_p99` are calculated
- **AND** values are reported in milliseconds

#### Scenario: Safety metrics calculated
- **GIVEN** a completed scenario run
- **WHEN** metrics are computed
- **THEN** `saturation_ratio` (fraction of time output was clipped) is calculated
- **AND** `safety_margin_min` (minimum distance to safety limits) is calculated

### Requirement: Scenario Library
The test harness SHALL provide a curated library of test scenarios.

#### Scenario: Scenario categories available
- **GIVEN** a developer wants to test an algorithm
- **WHEN** they list available scenarios
- **THEN** scenarios are organized by category:
  - Baseline (straight highway driving)
  - Curves (S-curves, highway exits)
  - Lane changes (merges, lane departures)
  - Edge cases (low speed, adverse conditions)

#### Scenario: Scenario validation
- **GIVEN** a scenario Parquet file
- **WHEN** loaded by the scenario runner
- **THEN** the schema is validated for required columns
- **AND** invalid scenarios are rejected with clear error messages

### Requirement: A/B Comparison
The test harness SHALL support comparing two algorithm variants.

#### Scenario: Compare algorithm variants
- **GIVEN** two algorithms implementing `AlgorithmInterface`
- **WHEN** the `compare` command is executed with a scenario set
- **THEN** both algorithms are run against all scenarios
- **AND** a comparison report shows metric differences (delta, percentage)

#### Scenario: Comparison identifies regressions
- **GIVEN** a baseline algorithm and a candidate algorithm
- **WHEN** the candidate performs worse on any metric beyond threshold
- **THEN** the comparison report flags the regression
- **AND** affected scenarios are listed

### Requirement: CLI Tooling
The test harness SHALL provide a command-line interface for algorithm benchmarking.

#### Scenario: Run benchmarks via CLI
- **GIVEN** a developer at the command line
- **WHEN** they execute `python tools/algo_bench.py run --algorithm lateral_pid --scenarios curves/`
- **THEN** the algorithm runs against all scenarios in the specified directory
- **AND** metrics are printed in tabular format

#### Scenario: Generate reports via CLI
- **GIVEN** completed benchmark results
- **WHEN** the developer executes `python tools/algo_bench.py report --format html`
- **THEN** an HTML report is generated with metrics tables and optional plots

### Requirement: Vehicle Dynamics Configuration
The test harness SHALL support configurable vehicle dynamics for simulation.

#### Scenario: Configure vehicle parameters
- **GIVEN** a `VehicleDynamicsConfig` with custom parameters
- **WHEN** a scenario is run with this configuration
- **THEN** the simulated vehicle responds according to the specified dynamics
- **AND** parameters include mass, wheelbase, steering ratio, and tire stiffness

#### Scenario: Preset vehicle configurations
- **GIVEN** a developer testing across vehicle types
- **WHEN** they select a preset (sedan, SUV, truck)
- **THEN** appropriate dynamics parameters are applied
- **AND** no manual configuration is required

### Requirement: Pytest Integration
The test harness SHALL integrate with pytest for automated testing.

#### Scenario: Algorithm benchmark as pytest test
- **GIVEN** a test function decorated with `@pytest.mark.algorithm_benchmark`
- **WHEN** pytest runs
- **THEN** the test uses harness fixtures for scenario loading and metrics collection
- **AND** failures are reported with metric values

#### Scenario: Harness fixtures available
- **GIVEN** a pytest test file
- **WHEN** the `algorithm_harness` fixture is requested
- **THEN** a configured `ScenarioRunner` instance is provided
- **AND** cleanup is handled automatically after the test

### Requirement: Code Quality Standards

The algorithm test harness code SHALL pass all ruff linting checks without violations.

#### Scenario: Pre-commit validation passes
- **WHEN** running `ruff check` on algorithm harness code
- **THEN** no linting errors are reported

#### Scenario: No banned imports
- **WHEN** the code is analyzed for import violations
- **THEN** no `unittest` module imports are present (use pytest instead)

#### Scenario: Strict zip usage
- **WHEN** `zip()` is used to iterate over multiple iterables
- **THEN** `strict=True` parameter is provided to catch length mismatches
