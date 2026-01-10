"""
Unit tests for the algorithm test harness framework.

Tests cover:
- Interface data classes
- MetricsCollector
- ScenarioRunner
- Controller adapters
"""

import time
import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.tests.algorithm_harness.interface import (
  AlgorithmInterface,
  AlgorithmState,
  AlgorithmOutput,
  LateralAlgorithmState,
  LateralAlgorithmOutput,
  LongitudinalAlgorithmState,
  LongitudinalAlgorithmOutput,
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.metrics import (
  MetricsCollector,
  AlgorithmMetrics,
  compare_metrics,
  format_metrics_table,
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.runner import (
  ScenarioRunner,
  Scenario,
  ScenarioResult,
  generate_synthetic_scenario,
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.adapters import (
  LatControlPIDAdapter,
  LongControlAdapter,
  LateralControlConfig,
  LongitudinalControlConfig,
)


# ============================================================================
# Test Fixtures
# ============================================================================

class SimpleAlgorithm:
  """Simple algorithm for testing that returns input scaled by factor."""

  def __init__(self, scale: float = 1.0):
    self.scale = scale
    self.call_count = 0

  def reset(self) -> None:
    self.call_count = 0

  def update(self, state: AlgorithmState) -> AlgorithmOutput:
    self.call_count += 1
    output = state.v_ego * self.scale
    return AlgorithmOutput(output=output, saturated=abs(output) > 10.0)


class LateralSimpleAlgorithm:
  """Simple lateral algorithm for testing."""

  def __init__(self, gain: float = 0.1):
    self.gain = gain

  def reset(self) -> None:
    pass

  def update(self, state: LateralAlgorithmState) -> LateralAlgorithmOutput:
    # Simple proportional control on curvature
    output = state.desired_curvature * self.gain
    return LateralAlgorithmOutput(
      output=output,
      saturated=abs(output) > 1.0,
      steering_angle_desired_deg=output * 15.0,
      angle_error_deg=state.steering_angle_deg - output * 15.0,
    )


# ============================================================================
# Interface Tests
# ============================================================================

class TestAlgorithmState:
  """Tests for AlgorithmState data class."""

  def test_default_values(self):
    """Test default values are set correctly."""
    state = AlgorithmState(timestamp_ns=0, v_ego=0.0, a_ego=0.0)
    assert state.active is True

  def test_all_values(self):
    """Test all values can be set."""
    state = AlgorithmState(
      timestamp_ns=1000000,
      v_ego=25.0,
      a_ego=1.5,
      active=False,
    )
    assert state.timestamp_ns == 1000000
    assert state.v_ego == 25.0
    assert state.a_ego == 1.5
    assert state.active is False


class TestLateralAlgorithmState:
  """Tests for LateralAlgorithmState data class."""

  def test_inherits_base_fields(self):
    """Test lateral state includes base state fields."""
    state = LateralAlgorithmState(
      timestamp_ns=0,
      v_ego=20.0,
      a_ego=0.0,
      desired_curvature=0.01,
    )
    assert state.v_ego == 20.0
    assert state.desired_curvature == 0.01

  def test_default_lateral_fields(self):
    """Test default lateral-specific fields."""
    state = LateralAlgorithmState(timestamp_ns=0, v_ego=0.0, a_ego=0.0)
    assert state.steering_angle_deg == 0.0
    assert state.yaw_rate == 0.0
    assert state.steering_pressed is False


class TestAlgorithmOutput:
  """Tests for AlgorithmOutput data class."""

  def test_default_metadata(self):
    """Test metadata defaults to empty dict."""
    output = AlgorithmOutput(output=0.5)
    assert output.metadata == {}
    assert output.saturated is False

  def test_metadata_mutable(self):
    """Test metadata can be modified."""
    output = AlgorithmOutput(output=0.5, metadata={'key': 'value'})
    assert output.metadata['key'] == 'value'


class TestAlgorithmInterface:
  """Tests for AlgorithmInterface protocol."""

  def test_simple_algorithm_implements_protocol(self):
    """Test that SimpleAlgorithm implements AlgorithmInterface."""
    algo = SimpleAlgorithm()
    assert isinstance(algo, AlgorithmInterface)

  def test_protocol_methods_callable(self):
    """Test protocol methods are callable."""
    algo = SimpleAlgorithm()
    algo.reset()
    state = AlgorithmState(timestamp_ns=0, v_ego=10.0, a_ego=0.0)
    output = algo.update(state)
    assert isinstance(output, AlgorithmOutput)


# ============================================================================
# MetricsCollector Tests
# ============================================================================

class TestMetricsCollector:
  """Tests for MetricsCollector."""

  def test_empty_metrics(self):
    """Test metrics with no data."""
    collector = MetricsCollector()
    metrics = collector.compute_metrics()
    assert metrics.total_steps == 0

  def test_collect_outputs(self):
    """Test collecting outputs."""
    collector = MetricsCollector()
    collector.start_run()
    for i in range(10):
      collector.start_step()
      collector.end_step(output=float(i))

    metrics = collector.compute_metrics()
    assert metrics.total_steps == 10

  def test_tracking_error_calculation(self):
    """Test tracking error metrics."""
    collector = MetricsCollector()
    collector.start_run()

    # Output = target + 1 (constant error of 1)
    for i in range(10):
      collector.start_step()
      collector.end_step(output=float(i) + 1.0, target=float(i))

    metrics = collector.compute_metrics()
    assert abs(metrics.tracking_error_rmse - 1.0) < 0.001
    assert abs(metrics.tracking_error_max - 1.0) < 0.001
    assert abs(metrics.tracking_error_mean - 1.0) < 0.001

  def test_smoothness_calculation(self):
    """Test output smoothness metric."""
    collector = MetricsCollector()
    collector.start_run()

    # Constant output = perfectly smooth
    for _ in range(10):
      collector.start_step()
      collector.end_step(output=5.0)

    metrics = collector.compute_metrics()
    assert metrics.output_smoothness == 0.0

  def test_smoothness_with_variation(self):
    """Test smoothness with varying output."""
    collector = MetricsCollector()
    collector.start_run()

    # Alternating output
    for i in range(10):
      collector.start_step()
      collector.end_step(output=0.0 if i % 2 == 0 else 1.0)

    metrics = collector.compute_metrics()
    assert metrics.output_smoothness > 0.0

  def test_latency_measurement(self):
    """Test latency is measured."""
    collector = MetricsCollector()
    collector.start_run()

    for _ in range(5):
      collector.start_step()
      time.sleep(0.001)  # 1ms delay
      collector.end_step(output=0.0)

    metrics = collector.compute_metrics()
    assert metrics.latency_mean_ms > 0.0
    assert metrics.latency_p50_ms > 0.0

  def test_saturation_ratio(self):
    """Test saturation ratio calculation."""
    collector = MetricsCollector()
    collector.start_run()

    # 5 saturated, 5 not
    for i in range(10):
      collector.start_step()
      collector.end_step(output=0.0, saturated=(i < 5))

    metrics = collector.compute_metrics()
    assert abs(metrics.saturation_ratio - 0.5) < 0.001

  def test_safety_margin(self):
    """Test safety margin calculation."""
    collector = MetricsCollector(safety_limits=(-1.0, 1.0))
    collector.start_run()

    collector.start_step()
    collector.end_step(output=0.5)  # margin = 0.5
    collector.start_step()
    collector.end_step(output=0.8)  # margin = 0.2
    collector.start_step()
    collector.end_step(output=-0.9)  # margin = 0.1

    metrics = collector.compute_metrics()
    assert abs(metrics.safety_margin_min - 0.1) < 0.001

  def test_reset(self):
    """Test reset clears all data."""
    collector = MetricsCollector()
    collector.start_run()
    collector.start_step()
    collector.end_step(output=1.0)

    collector.reset()
    metrics = collector.compute_metrics()
    assert metrics.total_steps == 0

  def test_get_raw_data(self):
    """Test raw data retrieval."""
    collector = MetricsCollector()
    collector.start_run()
    collector.start_step()
    collector.end_step(output=1.0, target=0.5, saturated=True)

    raw = collector.get_raw_data()
    assert raw['outputs'] == [1.0]
    assert raw['targets'] == [0.5]
    assert raw['saturated'] == [True]


class TestCompareMetrics:
  """Tests for compare_metrics function."""

  def test_improvement_detection(self):
    """Test improved metrics are detected."""
    baseline = AlgorithmMetrics(tracking_error_rmse=1.0)
    candidate = AlgorithmMetrics(tracking_error_rmse=0.5)

    comparison = compare_metrics(baseline, candidate)
    assert comparison['tracking_error_rmse']['improved'] is True
    assert comparison['tracking_error_rmse']['delta'] == -0.5

  def test_regression_detection(self):
    """Test regression is detected."""
    baseline = AlgorithmMetrics(tracking_error_rmse=0.5)
    candidate = AlgorithmMetrics(tracking_error_rmse=1.0)

    comparison = compare_metrics(baseline, candidate)
    assert comparison['tracking_error_rmse']['improved'] is False

  def test_percentage_change(self):
    """Test percentage change calculation."""
    baseline = AlgorithmMetrics(latency_mean_ms=10.0)
    candidate = AlgorithmMetrics(latency_mean_ms=8.0)

    comparison = compare_metrics(baseline, candidate)
    assert abs(comparison['latency_mean_ms']['pct_change'] - (-20.0)) < 0.001


class TestFormatMetricsTable:
  """Tests for format_metrics_table function."""

  def test_format_produces_string(self):
    """Test formatting produces non-empty string."""
    metrics = AlgorithmMetrics(
      tracking_error_rmse=0.5,
      latency_mean_ms=5.0,
    )
    table = format_metrics_table(metrics, "Test Algorithm")
    assert len(table) > 0
    assert "Test Algorithm" in table
    assert "0.5" in table


# ============================================================================
# ScenarioRunner Tests
# ============================================================================

class TestScenario:
  """Tests for Scenario data class."""

  def test_len(self):
    """Test scenario length."""
    states = [AlgorithmState(timestamp_ns=i, v_ego=0.0, a_ego=0.0) for i in range(5)]
    scenario = Scenario(name="test", states=states)
    assert len(scenario) == 5

  def test_iter(self):
    """Test scenario iteration."""
    states = [AlgorithmState(timestamp_ns=i, v_ego=float(i), a_ego=0.0) for i in range(3)]
    scenario = Scenario(name="test", states=states)

    v_egos = [s.v_ego for s in scenario]
    assert v_egos == [0.0, 1.0, 2.0]

  def test_with_ground_truth(self):
    """Test iteration with ground truth."""
    states = [AlgorithmState(timestamp_ns=i, v_ego=0.0, a_ego=0.0) for i in range(3)]
    ground_truth = [1.0, 2.0, 3.0]
    scenario = Scenario(name="test", states=states, ground_truth=ground_truth)

    pairs = list(scenario.with_ground_truth())
    assert len(pairs) == 3
    assert pairs[0][1] == 1.0
    assert pairs[2][1] == 3.0


class TestScenarioRunner:
  """Tests for ScenarioRunner."""

  def test_run_basic(self):
    """Test basic scenario run."""
    runner = ScenarioRunner()
    algo = SimpleAlgorithm(scale=0.1)
    scenario = generate_synthetic_scenario("test", duration_s=1.0, dt=0.1)

    result = runner.run(algo, scenario, "simple")

    assert result.success is True
    assert result.scenario_name == "test"
    assert result.algorithm_name == "simple"
    assert len(result.outputs) == 10

  def test_run_collects_metrics(self):
    """Test that run collects metrics."""
    runner = ScenarioRunner()
    algo = SimpleAlgorithm()
    scenario = generate_synthetic_scenario("test", duration_s=0.5, dt=0.01)

    result = runner.run(algo, scenario)

    assert result.metrics.total_steps == 50
    assert result.metrics.steps_per_second > 0

  def test_deterministic_mode(self):
    """Test deterministic mode produces same results."""
    runner = ScenarioRunner(deterministic=True, random_seed=42)
    algo = SimpleAlgorithm()
    scenario = generate_synthetic_scenario("test", duration_s=0.1, dt=0.01)

    result1 = runner.run(algo, scenario)
    algo.reset()
    result2 = runner.run(algo, scenario)

    assert result1.outputs == result2.outputs

  def test_run_batch(self):
    """Test batch scenario execution."""
    runner = ScenarioRunner()
    algo = SimpleAlgorithm()
    scenarios = [
      generate_synthetic_scenario(f"test_{i}", duration_s=0.1, dt=0.01)
      for i in range(3)
    ]

    results = runner.run_batch(algo, scenarios)

    assert len(results) == 3
    assert all(r.success for r in results)

  def test_compare_algorithms(self):
    """Test algorithm comparison."""
    runner = ScenarioRunner()
    baseline = SimpleAlgorithm(scale=1.0)
    candidate = SimpleAlgorithm(scale=0.5)
    scenarios = [generate_synthetic_scenario("test", duration_s=0.1, dt=0.01)]

    comparison = runner.compare(baseline, candidate, scenarios)

    assert 'per_scenario' in comparison
    assert 'aggregate' in comparison
    assert len(comparison['per_scenario']) == 1

  def test_handles_algorithm_error(self):
    """Test graceful handling of algorithm errors."""

    class FailingAlgorithm:
      def reset(self):
        pass

      def update(self, state):
        raise ValueError("Intentional failure")

    runner = ScenarioRunner()
    algo = FailingAlgorithm()
    scenario = generate_synthetic_scenario("test", duration_s=0.1, dt=0.01)

    result = runner.run(algo, scenario)

    assert result.success is False
    assert "Intentional failure" in result.error_message


class TestGenerateSyntheticScenario:
  """Tests for generate_synthetic_scenario function."""

  def test_constant_scenario(self):
    """Test constant scenario generation."""
    scenario = generate_synthetic_scenario(
      "constant_test",
      duration_s=1.0,
      dt=0.1,
      scenario_type="constant",
      target=5.0,
    )

    assert len(scenario) == 10
    assert all(gt == 5.0 for gt in scenario.ground_truth)

  def test_ramp_scenario(self):
    """Test ramp scenario generation."""
    scenario = generate_synthetic_scenario(
      "ramp_test",
      duration_s=1.0,
      dt=0.1,
      scenario_type="ramp",
      rate=1.0,
    )

    # Ground truth should increase linearly
    assert scenario.ground_truth[0] == 0.0
    assert abs(scenario.ground_truth[-1] - 0.9) < 0.01

  def test_sine_scenario(self):
    """Test sine scenario generation."""
    scenario = generate_synthetic_scenario(
      "sine_test",
      duration_s=2.0,
      dt=0.1,
      scenario_type="sine",
      amplitude=1.0,
      frequency=0.5,
    )

    # Should complete one cycle
    assert len(scenario) == 20
    assert abs(scenario.ground_truth[0]) < 0.01  # sin(0) = 0

  def test_step_scenario(self):
    """Test step scenario generation."""
    scenario = generate_synthetic_scenario(
      "step_test",
      duration_s=2.0,
      dt=0.1,
      scenario_type="step",
      step_time=1.0,
      step_value=1.0,
    )

    # First half should be 0, second half should be 1
    first_half = scenario.ground_truth[:10]
    second_half = scenario.ground_truth[10:]
    assert all(gt == 0.0 for gt in first_half)
    assert all(gt == 1.0 for gt in second_half)


# ============================================================================
# Adapter Tests
# ============================================================================

class TestLatControlPIDAdapter:
  """Tests for LatControlPIDAdapter."""

  def test_initialization(self):
    """Test adapter initialization."""
    adapter = LatControlPIDAdapter()
    assert adapter._controller is not None

  def test_reset(self):
    """Test reset clears state."""
    adapter = LatControlPIDAdapter()
    state = LateralAlgorithmState(
      timestamp_ns=0,
      v_ego=20.0,
      a_ego=0.0,
      desired_curvature=0.01,
    )
    adapter.update(state)
    adapter.reset()
    # Should not raise

  def test_update_returns_output(self):
    """Test update returns valid output."""
    adapter = LatControlPIDAdapter()
    state = LateralAlgorithmState(
      timestamp_ns=0,
      v_ego=20.0,
      a_ego=0.0,
      desired_curvature=0.01,
      active=True,
    )

    output = adapter.update(state)

    assert isinstance(output, LateralAlgorithmOutput)
    assert isinstance(output.output, float)

  def test_inactive_returns_zero(self):
    """Test inactive state returns zero output."""
    adapter = LatControlPIDAdapter()
    state = LateralAlgorithmState(
      timestamp_ns=0,
      v_ego=20.0,
      a_ego=0.0,
      desired_curvature=0.01,
      active=False,
    )

    output = adapter.update(state)

    assert output.output == 0.0

  def test_custom_config(self):
    """Test adapter with custom configuration."""
    config = LateralControlConfig(
      kp_v=[1.0],
      ki_v=[0.2],
      kf=2.0,
    )
    adapter = LatControlPIDAdapter(config)

    assert adapter.config.kp_v == [1.0]


class TestLongControlAdapter:
  """Tests for LongControlAdapter."""

  def test_initialization(self):
    """Test adapter initialization."""
    adapter = LongControlAdapter()
    assert adapter._controller is not None

  def test_update_returns_output(self):
    """Test update returns valid output."""
    adapter = LongControlAdapter()
    state = LongitudinalAlgorithmState(
      timestamp_ns=0,
      v_ego=20.0,
      a_ego=0.0,
      a_target=1.0,
      active=True,
    )

    output = adapter.update(state)

    assert isinstance(output, LongitudinalAlgorithmOutput)
    assert isinstance(output.accel, float)

  def test_control_state_in_output(self):
    """Test control state is reported in output."""
    adapter = LongControlAdapter()
    state = LongitudinalAlgorithmState(
      timestamp_ns=0,
      v_ego=20.0,
      a_ego=0.0,
      a_target=0.0,
      active=True,
    )

    output = adapter.update(state)

    assert output.control_state in ['off', 'stopping', 'starting', 'pid']


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
  """Integration tests for the full harness."""

  def test_run_pid_adapter_with_scenario(self):
    """Test running PID adapter through full scenario."""
    runner = ScenarioRunner(safety_limits=(-1.0, 1.0))
    adapter = LatControlPIDAdapter()

    # Create lateral scenario
    states = []
    ground_truth = []
    for i in range(100):
      state = LateralAlgorithmState(
        timestamp_ns=i * 10_000_000,  # 10ms steps
        v_ego=20.0,
        a_ego=0.0,
        steering_angle_deg=0.0,
        desired_curvature=0.001 * np.sin(i * 0.1),
        active=True,
      )
      states.append(state)
      ground_truth.append(state.desired_curvature * 0.1)

    scenario = Scenario(
      name="lateral_sine",
      description="Sinusoidal curvature tracking",
      states=states,
      ground_truth=ground_truth,
    )

    result = runner.run(adapter, scenario, "LatControlPID")

    assert result.success is True
    assert result.metrics.total_steps == 100
    assert result.metrics.latency_mean_ms >= 0

  def test_compare_lateral_algorithms(self):
    """Test comparing two lateral algorithms."""
    runner = ScenarioRunner()

    algo1 = LateralSimpleAlgorithm(gain=0.1)
    algo2 = LateralSimpleAlgorithm(gain=0.2)

    states = [
      LateralAlgorithmState(
        timestamp_ns=i * 10_000_000,
        v_ego=20.0,
        a_ego=0.0,
        desired_curvature=0.01,
        active=True,
      )
      for i in range(50)
    ]
    scenario = Scenario(
      name="constant_curvature",
      states=states,
      ground_truth=[0.001] * 50,  # Target output
    )

    comparison = runner.compare(algo1, algo2, [scenario])

    assert comparison['aggregate']['baseline'] is not None
    assert comparison['aggregate']['candidate'] is not None
