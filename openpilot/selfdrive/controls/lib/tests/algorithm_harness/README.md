# Algorithm Test Harness

A standardized framework for testing and benchmarking openpilot control algorithms without requiring comma device hardware.

## Features

- **AlgorithmInterface Protocol**: Standardized interface for algorithm implementations
- **ScenarioRunner**: Execute algorithms against deterministic test scenarios
- **MetricsCollector**: Comprehensive metrics (tracking error, smoothness, latency, safety)
- **Controller Adapters**: Wrappers for existing openpilot controllers (LatControlPID, LatControlTorque, LongControl)
- **Scenario Infrastructure**: Parquet-based scenario storage with 5 seed scenarios
- **CLI Tool**: Command-line benchmarking and comparison

## Quick Start

```python
from openpilot.selfdrive.controls.lib.tests.algorithm_harness import (
    ScenarioRunner,
    generate_synthetic_scenario,
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.adapters import (
    LatControlPIDAdapter,
)

# Create a scenario
scenario = generate_synthetic_scenario(
    name="test",
    duration_s=5.0,
    scenario_type="sine",
    amplitude=0.01,
)

# Run algorithm
runner = ScenarioRunner(deterministic=True)
adapter = LatControlPIDAdapter()
result = runner.run(adapter, scenario)

# Check metrics
print(f"Tracking RMSE: {result.metrics.tracking_error_rmse:.4f}")
print(f"Mean Latency: {result.metrics.latency_mean_ms:.2f}ms")
```

## CLI Usage

```bash
# Generate seed scenarios
python tools/algo_bench.py generate-scenarios --output ./scenarios

# List available scenarios
python tools/algo_bench.py list --scenarios ./scenarios

# Run benchmark
python tools/algo_bench.py run --algorithm lateral_pid --scenarios ./scenarios

# Compare algorithms
python tools/algo_bench.py compare \
    --baseline lateral_pid \
    --candidate lateral_torque \
    --scenarios ./scenarios
```

## Seed Scenarios

| Name | Type | Description |
|------|------|-------------|
| `highway_straight_baseline` | Lateral | Steady-state lane keeping at highway speed |
| `tight_s_curve` | Lateral | S-curve with varying curvature |
| `highway_lane_change` | Lateral | Planned lane change maneuver |
| `low_speed_maneuver` | Lateral | Parking lot with tight turns |
| `emergency_stop` | Longitudinal | Hard braking scenario |

## Metrics

| Metric | Description |
|--------|-------------|
| `tracking_error_rmse` | Root mean square error vs ground truth |
| `tracking_error_max` | Maximum absolute error |
| `output_smoothness` | RMS of output rate-of-change (jerk) |
| `latency_mean_ms` | Mean update latency |
| `latency_p99_ms` | 99th percentile latency |
| `saturation_ratio` | Fraction of time output was saturated |
| `safety_margin_min` | Minimum margin to safety limits |

## Implementing Custom Algorithms

```python
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.interface import (
    AlgorithmInterface,
    LateralAlgorithmState,
    LateralAlgorithmOutput,
)

class MyLateralAlgorithm:
    def __init__(self, gain: float = 0.5):
        self.gain = gain
        self.integrator = 0.0

    def reset(self) -> None:
        self.integrator = 0.0

    def update(self, state: LateralAlgorithmState) -> LateralAlgorithmOutput:
        error = state.desired_curvature * 100  # Simplified
        self.integrator += error * 0.01
        output = self.gain * error + 0.1 * self.integrator

        return LateralAlgorithmOutput(
            output=max(-1.0, min(1.0, output)),
            saturated=abs(output) >= 1.0,
        )
```

## Pytest Integration

```python
import pytest

@pytest.mark.algorithm_benchmark
def test_my_algorithm(scenario_runner, highway_straight_scenario):
    scenario, metadata = highway_straight_scenario
    algorithm = MyLateralAlgorithm()

    result = scenario_runner.run(algorithm, scenario)

    assert result.success
    assert result.metrics.tracking_error_rmse < 0.1
    assert result.metrics.latency_mean_ms < 10.0
```

## Coverage Requirements

This module maintains **90%+ test coverage**. Run coverage locally:

```bash
pytest selfdrive/controls/lib/tests/algorithm_harness/ \
    --cov=selfdrive.controls.lib.tests.algorithm_harness \
    --cov-report=html
```

## Contributing Scenarios

We welcome contributions of new test scenarios. Follow these guidelines to ensure your scenarios integrate well with the harness.

### Scenario Requirements

1. **Use Parquet format** with the schema defined in `scenario_schema.py`
2. **Include metadata** with all required fields (name, type, difficulty, duration)
3. **Provide ground truth** values for tracking error calculation
4. **Keep scenarios focused** - one driving situation per scenario

### Creating a New Scenario

#### From Synthetic Data

```python
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenario_generator import (
    create_lateral_scenario,
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenario_schema import (
    ScenarioType,
    DifficultyLevel,
)

# Generate scenario data
scenario = create_lateral_scenario(
    name="my_custom_curve",
    description="Sharp right curve at moderate speed",
    scenario_type=ScenarioType.HIGHWAY_CURVE,
    difficulty=DifficultyLevel.MEDIUM,
    duration_s=15.0,
    # Add custom parameters...
)
```

#### From Route Logs

```python
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenarios import (
    extract_scenario_from_route,
)

# Extract from a recorded drive
scenario = extract_scenario_from_route(
    route_id="a]abc123def456|2024-01-15--12-30-00",
    start_s=120.0,
    end_s=150.0,
    scenario_type=ScenarioType.HIGHWAY_LANE_CHANGE,
    difficulty=DifficultyLevel.HARD,
)
```

### Scenario Naming Convention

Use descriptive, lowercase names with underscores:

| Pattern | Example | Use For |
|---------|---------|---------|
| `{road}_{maneuver}_{variant}` | `highway_curve_tight` | Standard scenarios |
| `{condition}_{situation}` | `wet_emergency_stop` | Weather/condition tests |
| `{source}_{segment}` | `route_abc123_lane_change` | Route-extracted scenarios |

### Required Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Unique identifier (snake_case) |
| `description` | str | Human-readable description |
| `scenario_type` | ScenarioType | Category from enum |
| `difficulty` | DifficultyLevel | easy/medium/hard/extreme |
| `duration_s` | float | Total duration in seconds |
| `dt_s` | float | Time step (typically 0.01s) |
| `source` | str | "synthetic", "route_log", or "simulation" |

### Validation Checklist

Before submitting a scenario:

- [ ] Scenario loads without errors: `load_scenario(path, scenario_class)`
- [ ] Metadata is complete and accurate
- [ ] Ground truth values are reasonable (not NaN, within expected ranges)
- [ ] Duration matches actual data length
- [ ] At least one existing algorithm runs successfully against it
- [ ] Scenario name is unique and descriptive

### Testing Your Scenario

```bash
# Validate scenario file
python -c "
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenarios import load_scenario
scenario = load_scenario('path/to/your_scenario.parquet', 'lateral')
print(f'Loaded: {scenario.name}, {len(scenario)} steps')
"

# Run benchmark
python tools/algo_bench.py run \
    --algorithm lateral_pid \
    --scenarios path/to/scenarios/
```

### Submitting Scenarios

1. Place scenario files in `tools/lib/test_scenarios/`
2. Add an entry to the scenario catalog (if applicable)
3. Include a test that validates the scenario loads correctly
4. Document any special characteristics in the PR description

## Module Structure

```
algorithm_harness/
├── __init__.py           # Package exports
├── interface.py          # AlgorithmInterface protocol, state/output classes
├── metrics.py            # MetricsCollector, comparison utilities
├── runner.py             # ScenarioRunner, Scenario class
├── adapters.py           # Controller adapters (PID, Torque, Long)
├── scenario_schema.py    # Parquet schema definitions
├── scenarios.py          # Load/save/validate scenarios
├── scenario_generator.py # Seed scenario generators
├── conftest.py           # Pytest fixtures and markers
├── test_harness.py       # Core framework tests
├── test_scenarios.py     # Scenario infrastructure tests
├── .coveragerc           # Coverage configuration
└── README.md             # This file
```

## Test Coverage

The algorithm harness maintains a **90%+ code coverage** target for source files.

### Running Coverage Locally

```bash
# Navigate to harness directory
cd selfdrive/controls/lib/tests/algorithm_harness

# Run tests with coverage (basic - no pandas/pyarrow)
pytest . -n0 --cov=. --cov-config=.coveragerc --cov-report=term-missing

# Run with HTML report
pytest . -n0 --cov=. --cov-config=.coveragerc --cov-report=html

# Full coverage (requires pandas and pyarrow for Parquet I/O tests)
pip install pandas pyarrow
pytest . -n0 --cov=. --cov-config=.coveragerc --cov-report=term-missing --cov-fail-under=90
```

### Coverage Requirements

| Module | Target | Notes |
|--------|--------|-------|
| `interface.py` | 90%+ | Protocol and data classes |
| `metrics.py` | 90%+ | Metrics collection and comparison |
| `runner.py` | 90%+ | Scenario execution |
| `adapters.py` | 90%+ | Controller wrappers |
| `scenario_generator.py` | 90%+ | Seed scenario generation |
| `scenarios.py` | 90%+ | Requires pandas/pyarrow |
| `scenario_schema.py` | 90%+ | Requires pyarrow |
| `vehicle_dynamics.py` | 90%+ | Vehicle simulation |

### Contributing Coverage

When adding new features:
1. Write tests first (TDD encouraged)
2. Run coverage locally before submitting PR
3. Ensure no decrease in overall coverage
4. Document any deliberately excluded code with `# pragma: no cover`

## GPU and Parallel Acceleration

For large-scale scenario testing on NVIDIA GPUs (including DGX Spark), use the GPU-accelerated runner:

```python
from openpilot.tools.dgx.algorithm_harness_gpu import GPUScenarioRunner

# Create parallel runner
runner = GPUScenarioRunner(max_workers=8)

# Run scenarios in parallel
results = runner.run_batch_parallel(
    algorithm_factory=lambda: MyAlgorithm(),
    scenarios=scenarios,
)

print(f"Processed {results.num_scenarios} scenarios")
print(f"Throughput: {results.scenarios_per_second:.1f} scenarios/second")
```

### When Parallel Execution Helps

Parallel execution provides speedup when:
- **Many scenarios**: 50+ independent scenarios to run
- **Complex algorithms**: Per-step computation takes >1ms (Kalman filters, model inference)
- **Batch experiments**: Running same algorithm with different parameters

For lightweight algorithms (<1ms per step), sequential execution is often faster due to process pool overhead.

### Benchmarking Parallel Performance

```bash
# Run parallel benchmark
python tools/dgx/benchmark_parallel.py --scenarios 100 --steps 5000

# With algorithm harness scenarios
python tools/dgx/algorithm_harness_gpu.py
```

### GPU Metrics Acceleration

For large datasets, use GPU-accelerated percentile computation:

```python
from openpilot.tools.dgx.algorithm_harness_gpu import GPUMetricsAccelerator

accelerator = GPUMetricsAccelerator()
latencies = np.array([...])  # 10000+ samples
p50, p95, p99 = accelerator.percentiles(latencies, [50, 95, 99])
```

See `tools/dgx/README.md` for full GPU acceleration documentation.
