"""
Pytest fixtures and markers for algorithm test harness.

This module provides pytest integration for the algorithm harness,
including fixtures for creating scenarios, runners, and adapters,
as well as custom markers for benchmark tests.

Usage:
  @pytest.mark.algorithm_benchmark
  def test_lateral_pid_tracking(algorithm_harness, lateral_scenario):
    result = algorithm_harness.run(LatControlPIDAdapter(), lateral_scenario)
    assert result.metrics.tracking_error_rmse < 0.1
"""

import pytest
from typing import Generator

from openpilot.selfdrive.controls.lib.tests.algorithm_harness.interface import (
  LateralAlgorithmState,
  LongitudinalAlgorithmState,
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.runner import (
  Scenario,
  ScenarioRunner,
  generate_synthetic_scenario,
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.metrics import (
  MetricsCollector,
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.adapters import (
  LatControlPIDAdapter,
  LatControlTorqueAdapter,
  LongControlAdapter,
  LateralControlConfig,
  LongitudinalControlConfig,
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenario_generator import (
  generate_highway_straight,
  generate_tight_s_curve,
  generate_highway_lane_change,
  generate_low_speed_maneuver,
  generate_emergency_stop,
)


def pytest_configure(config):
  """Register custom markers."""
  config.addinivalue_line(
    "markers",
    "algorithm_benchmark: mark test as an algorithm benchmark test"
  )
  config.addinivalue_line(
    "markers",
    "slow_benchmark: mark test as a slow benchmark (skipped with -m 'not slow_benchmark')"
  )


# ============================================================================
# Core Fixtures
# ============================================================================

@pytest.fixture
def scenario_runner() -> ScenarioRunner:
  """Create a deterministic scenario runner."""
  return ScenarioRunner(deterministic=True, random_seed=42)


@pytest.fixture
def metrics_collector() -> MetricsCollector:
  """Create a metrics collector with standard safety limits."""
  return MetricsCollector(safety_limits=(-1.0, 1.0))


# ============================================================================
# Algorithm Fixtures
# ============================================================================

@pytest.fixture
def lateral_pid_adapter() -> LatControlPIDAdapter:
  """Create a LatControlPID adapter with default config."""
  return LatControlPIDAdapter()


@pytest.fixture
def lateral_pid_adapter_custom() -> Generator[LatControlPIDAdapter, None, None]:
  """Factory fixture for custom LatControlPID configurations."""
  adapters = []

  def _factory(kp: float = 0.5, ki: float = 0.1, kf: float = 1.0) -> LatControlPIDAdapter:
    config = LateralControlConfig(kp_v=[kp], ki_v=[ki], kf=kf)
    adapter = LatControlPIDAdapter(config)
    adapters.append(adapter)
    return adapter

  yield _factory

  # Cleanup
  for adapter in adapters:
    adapter.reset()


@pytest.fixture
def lateral_torque_adapter() -> LatControlTorqueAdapter:
  """Create a LatControlTorque adapter with default config."""
  return LatControlTorqueAdapter()


@pytest.fixture
def long_control_adapter() -> LongControlAdapter:
  """Create a LongControl adapter with default config."""
  return LongControlAdapter()


# ============================================================================
# Scenario Fixtures
# ============================================================================

@pytest.fixture
def simple_lateral_scenario() -> Scenario:
  """Simple lateral scenario for quick tests."""
  return generate_synthetic_scenario(
    name="simple_lateral",
    duration_s=1.0,
    dt=0.01,
    scenario_type="sine",
    amplitude=0.005,
    frequency=0.5,
  )


@pytest.fixture
def simple_longitudinal_scenario() -> Scenario:
  """Simple longitudinal scenario for quick tests."""
  states = []
  ground_truth = []

  for i in range(100):
    t = i * 0.01
    state = LongitudinalAlgorithmState(
      timestamp_ns=int(t * 1e9),
      v_ego=20.0,
      a_ego=0.0,
      active=True,
      a_target=0.5 if t > 0.5 else 0.0,
      should_stop=False,
    )
    states.append(state)
    ground_truth.append(state.a_target)

  return Scenario(
    name="simple_longitudinal",
    description="Simple acceleration step",
    states=states,
    ground_truth=ground_truth,
  )


@pytest.fixture
def highway_straight_scenario() -> tuple[Scenario, dict]:
  """Highway straight driving scenario."""
  scenario, metadata = generate_highway_straight(duration_s=5.0)
  return scenario, metadata.to_dict()


@pytest.fixture
def tight_curve_scenario() -> tuple[Scenario, dict]:
  """Tight S-curve scenario."""
  scenario, metadata = generate_tight_s_curve(duration_s=5.0)
  return scenario, metadata.to_dict()


@pytest.fixture
def lane_change_scenario() -> tuple[Scenario, dict]:
  """Highway lane change scenario."""
  scenario, metadata = generate_highway_lane_change(duration_s=5.0)
  return scenario, metadata.to_dict()


@pytest.fixture
def low_speed_scenario() -> tuple[Scenario, dict]:
  """Low-speed maneuver scenario."""
  scenario, metadata = generate_low_speed_maneuver(duration_s=10.0)
  return scenario, metadata.to_dict()


@pytest.fixture
def emergency_stop_scenario() -> tuple[Scenario, dict]:
  """Emergency stop scenario."""
  scenario, metadata = generate_emergency_stop(duration_s=5.0)
  return scenario, metadata.to_dict()


@pytest.fixture
def all_lateral_scenarios(
  highway_straight_scenario,
  tight_curve_scenario,
  lane_change_scenario,
  low_speed_scenario,
) -> list[Scenario]:
  """All lateral test scenarios."""
  return [
    highway_straight_scenario[0],
    tight_curve_scenario[0],
    lane_change_scenario[0],
    low_speed_scenario[0],
  ]


# ============================================================================
# Parameterized Test Helpers
# ============================================================================

LATERAL_ALGORITHMS = [
  pytest.param("lateral_pid", id="PID"),
  pytest.param("lateral_torque", id="Torque"),
]

LONGITUDINAL_ALGORITHMS = [
  pytest.param("longitudinal", id="LongControl"),
]

LATERAL_SCENARIOS = [
  pytest.param("highway_straight", id="highway_straight"),
  pytest.param("tight_curve", id="tight_curve"),
  pytest.param("lane_change", id="lane_change"),
  pytest.param("low_speed", id="low_speed"),
]


@pytest.fixture
def get_algorithm():
  """Factory fixture to get algorithm by name."""
  algorithm_map = {
    'lateral_pid': LatControlPIDAdapter,
    'lateral_torque': LatControlTorqueAdapter,
    'longitudinal': LongControlAdapter,
  }

  def _get(name: str):
    if name not in algorithm_map:
      raise ValueError(f"Unknown algorithm: {name}")
    return algorithm_map[name]()

  return _get


@pytest.fixture
def get_scenario():
  """Factory fixture to get scenario by name."""
  scenario_map = {
    'highway_straight': lambda: generate_highway_straight(duration_s=2.0)[0],
    'tight_curve': lambda: generate_tight_s_curve(duration_s=2.0)[0],
    'lane_change': lambda: generate_highway_lane_change(duration_s=2.0)[0],
    'low_speed': lambda: generate_low_speed_maneuver(duration_s=5.0)[0],
    'emergency_stop': lambda: generate_emergency_stop(duration_s=2.0)[0],
  }

  def _get(name: str):
    if name not in scenario_map:
      raise ValueError(f"Unknown scenario: {name}")
    return scenario_map[name]()

  return _get


# ============================================================================
# Assertion Helpers
# ============================================================================

class BenchmarkAssertions:
  """Helper class for benchmark assertions."""

  @staticmethod
  def assert_tracking_error(result, max_rmse: float = 0.1, max_error: float = 0.5):
    """Assert tracking error is within bounds."""
    assert result.metrics.tracking_error_rmse <= max_rmse, \
      f"RMSE {result.metrics.tracking_error_rmse:.4f} > {max_rmse}"
    assert result.metrics.tracking_error_max <= max_error, \
      f"Max error {result.metrics.tracking_error_max:.4f} > {max_error}"

  @staticmethod
  def assert_smoothness(result, max_jerk: float = 0.5):
    """Assert output smoothness is acceptable."""
    assert result.metrics.output_smoothness <= max_jerk, \
      f"Smoothness (jerk) {result.metrics.output_smoothness:.4f} > {max_jerk}"

  @staticmethod
  def assert_latency(result, max_mean_ms: float = 10.0, max_p99_ms: float = 50.0):
    """Assert latency is acceptable."""
    assert result.metrics.latency_mean_ms <= max_mean_ms, \
      f"Mean latency {result.metrics.latency_mean_ms:.2f}ms > {max_mean_ms}ms"
    assert result.metrics.latency_p99_ms <= max_p99_ms, \
      f"P99 latency {result.metrics.latency_p99_ms:.2f}ms > {max_p99_ms}ms"

  @staticmethod
  def assert_safety(result, max_saturation: float = 0.1):
    """Assert safety metrics are acceptable."""
    assert result.metrics.saturation_ratio <= max_saturation, \
      f"Saturation ratio {result.metrics.saturation_ratio:.2%} > {max_saturation:.2%}"


@pytest.fixture
def benchmark_assertions() -> BenchmarkAssertions:
  """Provide benchmark assertion helpers."""
  return BenchmarkAssertions()
