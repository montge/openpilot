"""
Scenario runner for algorithm benchmarking.

This module provides the ScenarioRunner class that executes algorithms
against test scenarios and collects performance metrics.
"""

import random
from dataclasses import dataclass, field
from typing import Optional, Any
from collections.abc import Iterator
import numpy as np

from openpilot.selfdrive.controls.lib.tests.algorithm_harness.interface import (
  AlgorithmInterface,
  AlgorithmState,
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.metrics import (
  MetricsCollector,
  AlgorithmMetrics,
  compare_metrics,
)


@dataclass
class Scenario:
  """
  A test scenario containing timestamped states and optional ground truth.

  Scenarios can be loaded from Parquet files or generated synthetically.
  """

  name: str
  description: str = ""
  states: list[AlgorithmState] = field(default_factory=list)
  ground_truth: Optional[list[float]] = None  # Expected outputs for each state
  metadata: dict[str, Any] = field(default_factory=dict)

  def __len__(self) -> int:
    return len(self.states)

  def __iter__(self) -> Iterator[AlgorithmState]:
    return iter(self.states)

  def with_ground_truth(self) -> Iterator[tuple[AlgorithmState, Optional[float]]]:
    """Iterate over states with their ground truth values."""
    if self.ground_truth is None:
      for state in self.states:
        yield state, None
    else:
      for state, gt in zip(self.states, self.ground_truth):
        yield state, gt


@dataclass
class ScenarioResult:
  """Result of running an algorithm against a scenario."""

  scenario_name: str
  algorithm_name: str
  metrics: AlgorithmMetrics
  outputs: list[float] = field(default_factory=list)
  success: bool = True
  error_message: str = ""


class ScenarioRunner:
  """
  Executes algorithms against test scenarios and collects metrics.

  Usage:
    runner = ScenarioRunner()
    scenario = load_scenario("highway_following.parquet")
    result = runner.run(my_algorithm, scenario)
    print(result.metrics)
  """

  def __init__(
    self,
    deterministic: bool = True,
    random_seed: int = 42,
    safety_limits: Optional[tuple[float, float]] = None,
  ):
    """
    Initialize scenario runner.

    Args:
      deterministic: If True, use fixed random seeds for reproducibility
      random_seed: Seed for random number generators
      safety_limits: Optional (min, max) for safety margin calculation
    """
    self.deterministic = deterministic
    self.random_seed = random_seed
    self.safety_limits = safety_limits

    if deterministic:
      self._set_deterministic_mode()

  def _set_deterministic_mode(self) -> None:
    """Configure deterministic execution."""
    random.seed(self.random_seed)
    np.random.seed(self.random_seed)

  def run(
    self,
    algorithm: AlgorithmInterface,
    scenario: Scenario,
    algorithm_name: str = "algorithm",
  ) -> ScenarioResult:
    """
    Run an algorithm against a scenario.

    Args:
      algorithm: Algorithm implementing AlgorithmInterface
      scenario: Test scenario with states and optional ground truth
      algorithm_name: Name for result labeling

    Returns:
      ScenarioResult with metrics and outputs
    """
    if self.deterministic:
      self._set_deterministic_mode()

    collector = MetricsCollector(safety_limits=self.safety_limits)
    outputs: list[float] = []

    try:
      algorithm.reset()
      collector.start_run()

      for state, ground_truth in scenario.with_ground_truth():
        collector.start_step()
        output = algorithm.update(state)
        collector.end_step(
          output=output.output,
          target=ground_truth,
          saturated=output.saturated,
        )
        outputs.append(output.output)

      metrics = collector.compute_metrics()

      return ScenarioResult(
        scenario_name=scenario.name,
        algorithm_name=algorithm_name,
        metrics=metrics,
        outputs=outputs,
        success=True,
      )

    except Exception as e:
      return ScenarioResult(
        scenario_name=scenario.name,
        algorithm_name=algorithm_name,
        metrics=collector.compute_metrics(),
        outputs=outputs,
        success=False,
        error_message=str(e),
      )

  def run_batch(
    self,
    algorithm: AlgorithmInterface,
    scenarios: list[Scenario],
    algorithm_name: str = "algorithm",
  ) -> list[ScenarioResult]:
    """
    Run an algorithm against multiple scenarios.

    Args:
      algorithm: Algorithm implementing AlgorithmInterface
      scenarios: List of test scenarios
      algorithm_name: Name for result labeling

    Returns:
      List of ScenarioResults
    """
    return [self.run(algorithm, scenario, algorithm_name) for scenario in scenarios]

  def compare(
    self,
    baseline: AlgorithmInterface,
    candidate: AlgorithmInterface,
    scenarios: list[Scenario],
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
  ) -> dict[str, Any]:
    """
    Compare two algorithms across scenarios.

    Args:
      baseline: Baseline algorithm
      candidate: Candidate algorithm to compare
      scenarios: List of test scenarios
      baseline_name: Name for baseline
      candidate_name: Name for candidate

    Returns:
      Comparison results with per-scenario and aggregate metrics
    """
    baseline_results = self.run_batch(baseline, scenarios, baseline_name)
    candidate_results = self.run_batch(candidate, scenarios, candidate_name)

    comparisons = []
    for base_result, cand_result in zip(baseline_results, candidate_results):
      comparison = compare_metrics(base_result.metrics, cand_result.metrics)
      comparisons.append(
        {
          'scenario': base_result.scenario_name,
          'baseline_metrics': base_result.metrics,
          'candidate_metrics': cand_result.metrics,
          'comparison': comparison,
        }
      )

    # Aggregate metrics
    aggregate_baseline = self._aggregate_metrics([r.metrics for r in baseline_results])
    aggregate_candidate = self._aggregate_metrics([r.metrics for r in candidate_results])
    aggregate_comparison = compare_metrics(aggregate_baseline, aggregate_candidate)

    return {
      'per_scenario': comparisons,
      'aggregate': {
        'baseline': aggregate_baseline,
        'candidate': aggregate_candidate,
        'comparison': aggregate_comparison,
      },
      'baseline_results': baseline_results,
      'candidate_results': candidate_results,
    }

  def _aggregate_metrics(self, metrics_list: list[AlgorithmMetrics]) -> AlgorithmMetrics:
    """Aggregate metrics across multiple runs."""
    if not metrics_list:
      return AlgorithmMetrics()

    return AlgorithmMetrics(
      tracking_error_rmse=float(np.mean([m.tracking_error_rmse for m in metrics_list])),
      tracking_error_max=float(np.max([m.tracking_error_max for m in metrics_list])),
      tracking_error_mean=float(np.mean([m.tracking_error_mean for m in metrics_list])),
      output_smoothness=float(np.mean([m.output_smoothness for m in metrics_list])),
      output_std=float(np.mean([m.output_std for m in metrics_list])),
      latency_mean_ms=float(np.mean([m.latency_mean_ms for m in metrics_list])),
      latency_p50_ms=float(np.mean([m.latency_p50_ms for m in metrics_list])),
      latency_p99_ms=float(np.max([m.latency_p99_ms for m in metrics_list])),
      latency_max_ms=float(np.max([m.latency_max_ms for m in metrics_list])),
      saturation_ratio=float(np.mean([m.saturation_ratio for m in metrics_list])),
      safety_margin_min=float(np.min([m.safety_margin_min for m in metrics_list])),
      total_steps=sum(m.total_steps for m in metrics_list),
      total_time_s=sum(m.total_time_s for m in metrics_list),
      steps_per_second=float(np.mean([m.steps_per_second for m in metrics_list])),
    )


def generate_synthetic_scenario(
  name: str,
  duration_s: float = 10.0,
  dt: float = 0.01,
  scenario_type: str = "constant",
  **kwargs,
) -> Scenario:
  """
  Generate a synthetic test scenario.

  Args:
    name: Scenario name
    duration_s: Duration in seconds
    dt: Time step in seconds
    scenario_type: Type of scenario ("constant", "ramp", "sine", "step")
    **kwargs: Additional parameters for scenario generation

  Returns:
    Generated Scenario
  """
  num_steps = int(duration_s / dt)
  timestamps = [int(i * dt * 1e9) for i in range(num_steps)]

  v_ego_base = kwargs.get('v_ego', 20.0)  # m/s
  a_ego_base = kwargs.get('a_ego', 0.0)  # m/s^2

  states: list[AlgorithmState] = []
  ground_truth: list[float] = []

  for i, ts in enumerate(timestamps):
    t = i * dt

    if scenario_type == "constant":
      target = kwargs.get('target', 0.0)
    elif scenario_type == "ramp":
      rate = kwargs.get('rate', 0.1)
      target = rate * t
    elif scenario_type == "sine":
      amplitude = kwargs.get('amplitude', 1.0)
      frequency = kwargs.get('frequency', 0.5)
      target = amplitude * np.sin(2 * np.pi * frequency * t)
    elif scenario_type == "step":
      step_time = kwargs.get('step_time', duration_s / 2)
      step_value = kwargs.get('step_value', 1.0)
      target = step_value if t >= step_time else 0.0
    else:
      target = 0.0

    state = AlgorithmState(
      timestamp_ns=ts,
      v_ego=v_ego_base,
      a_ego=a_ego_base,
      active=True,
    )
    states.append(state)
    ground_truth.append(target)

  return Scenario(
    name=name,
    description=f"Synthetic {scenario_type} scenario",
    states=states,
    ground_truth=ground_truth,
    metadata={
      'duration_s': duration_s,
      'dt': dt,
      'scenario_type': scenario_type,
      **kwargs,
    },
  )
