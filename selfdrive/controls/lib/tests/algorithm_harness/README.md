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
└── README.md             # This file
```
