"""
Metrics collection and analysis for algorithm benchmarking.

This module provides standardized metrics for evaluating control algorithm
performance, including tracking accuracy, smoothness, latency, and safety margins.
"""

import time
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class AlgorithmMetrics:
  """Collected metrics for an algorithm run."""

  # Tracking accuracy
  tracking_error_rmse: float = 0.0  # Root mean square error vs target
  tracking_error_max: float = 0.0  # Maximum absolute error
  tracking_error_mean: float = 0.0  # Mean absolute error

  # Output smoothness
  output_smoothness: float = 0.0  # RMS of output rate-of-change (jerk)
  output_std: float = 0.0  # Standard deviation of output

  # Latency
  latency_mean_ms: float = 0.0  # Mean update latency
  latency_p50_ms: float = 0.0  # Median latency
  latency_p99_ms: float = 0.0  # 99th percentile latency
  latency_max_ms: float = 0.0  # Maximum latency

  # Safety metrics
  saturation_ratio: float = 0.0  # Fraction of time output was saturated
  safety_margin_min: float = float('inf')  # Minimum margin to safety limits

  # Run metadata
  total_steps: int = 0
  total_time_s: float = 0.0
  steps_per_second: float = 0.0


class MetricsCollector:
  """
  Collects and computes metrics during algorithm execution.

  Usage:
    collector = MetricsCollector()
    for state in scenario:
      collector.start_step()
      output = algorithm.update(state)
      collector.end_step(state, output, ground_truth)
    metrics = collector.compute_metrics()
  """

  def __init__(self, safety_limits: Optional[tuple[float, float]] = None):
    """
    Initialize metrics collector.

    Args:
      safety_limits: Optional (min, max) tuple for safety margin calculation
    """
    self.safety_limits = safety_limits
    self.reset()

  def reset(self) -> None:
    """Reset all collected data."""
    self._outputs: list[float] = []
    self._targets: list[float] = []
    self._errors: list[float] = []
    self._latencies: list[float] = []
    self._saturated: list[bool] = []
    self._safety_margins: list[float] = []
    self._step_start_time: Optional[float] = None
    self._run_start_time: Optional[float] = None

  def start_run(self) -> None:
    """Mark the start of a benchmark run."""
    self._run_start_time = time.monotonic()

  def start_step(self) -> None:
    """Mark the start of a single algorithm update step."""
    self._step_start_time = time.monotonic()

  def end_step(
    self,
    output: float,
    target: Optional[float] = None,
    saturated: bool = False,
  ) -> None:
    """
    Record data for a completed step.

    Args:
      output: Algorithm output value
      target: Ground truth / target value (optional)
      saturated: Whether output was saturated/clipped
    """
    # Record latency
    if self._step_start_time is not None:
      latency = (time.monotonic() - self._step_start_time) * 1000  # ms
      self._latencies.append(latency)
      self._step_start_time = None

    # Record output
    self._outputs.append(output)
    self._saturated.append(saturated)

    # Record error if target provided
    if target is not None:
      self._targets.append(target)
      self._errors.append(abs(output - target))

    # Record safety margin if limits defined
    if self.safety_limits is not None:
      min_limit, max_limit = self.safety_limits
      margin = min(output - min_limit, max_limit - output)
      self._safety_margins.append(margin)

  def compute_metrics(self) -> AlgorithmMetrics:
    """
    Compute final metrics from collected data.

    Returns:
      AlgorithmMetrics with all computed values
    """
    metrics = AlgorithmMetrics()
    metrics.total_steps = len(self._outputs)

    if metrics.total_steps == 0:
      return metrics

    outputs = np.array(self._outputs)

    # Tracking accuracy
    if len(self._errors) > 0:
      errors = np.array(self._errors)
      metrics.tracking_error_rmse = float(np.sqrt(np.mean(errors**2)))
      metrics.tracking_error_max = float(np.max(errors))
      metrics.tracking_error_mean = float(np.mean(errors))

    # Output smoothness (RMS of derivative)
    if len(outputs) > 1:
      output_diff = np.diff(outputs)
      metrics.output_smoothness = float(np.sqrt(np.mean(output_diff**2)))
      metrics.output_std = float(np.std(outputs))

    # Latency metrics
    if len(self._latencies) > 0:
      latencies = np.array(self._latencies)
      metrics.latency_mean_ms = float(np.mean(latencies))
      metrics.latency_p50_ms = float(np.percentile(latencies, 50))
      metrics.latency_p99_ms = float(np.percentile(latencies, 99))
      metrics.latency_max_ms = float(np.max(latencies))

    # Safety metrics
    if len(self._saturated) > 0:
      metrics.saturation_ratio = float(np.mean(self._saturated))

    if len(self._safety_margins) > 0:
      metrics.safety_margin_min = float(np.min(self._safety_margins))

    # Run timing
    if self._run_start_time is not None:
      metrics.total_time_s = time.monotonic() - self._run_start_time
      if metrics.total_time_s > 0:
        metrics.steps_per_second = metrics.total_steps / metrics.total_time_s

    return metrics

  def get_raw_data(self) -> dict[str, list]:
    """
    Get raw collected data for detailed analysis.

    Returns:
      Dictionary with outputs, targets, errors, latencies, etc.
    """
    return {
      'outputs': self._outputs.copy(),
      'targets': self._targets.copy(),
      'errors': self._errors.copy(),
      'latencies': self._latencies.copy(),
      'saturated': self._saturated.copy(),
      'safety_margins': self._safety_margins.copy(),
    }


def compare_metrics(baseline: AlgorithmMetrics, candidate: AlgorithmMetrics) -> dict[str, dict]:
  """
  Compare metrics between two algorithm runs.

  Args:
    baseline: Metrics from baseline algorithm
    candidate: Metrics from candidate algorithm

  Returns:
    Dictionary with comparison results for each metric
  """
  comparisons = {}

  metric_fields = [
    ('tracking_error_rmse', 'lower_better'),
    ('tracking_error_max', 'lower_better'),
    ('tracking_error_mean', 'lower_better'),
    ('output_smoothness', 'lower_better'),
    ('latency_mean_ms', 'lower_better'),
    ('latency_p50_ms', 'lower_better'),
    ('latency_p99_ms', 'lower_better'),
    ('saturation_ratio', 'lower_better'),
    ('safety_margin_min', 'higher_better'),
    ('steps_per_second', 'higher_better'),
  ]

  for field_name, direction in metric_fields:
    base_val = getattr(baseline, field_name)
    cand_val = getattr(candidate, field_name)

    if base_val == 0:
      pct_change = float('inf') if cand_val != 0 else 0.0
    else:
      pct_change = ((cand_val - base_val) / abs(base_val)) * 100

    if direction == 'lower_better':
      improved = cand_val < base_val
    else:
      improved = cand_val > base_val

    comparisons[field_name] = {
      'baseline': base_val,
      'candidate': cand_val,
      'delta': cand_val - base_val,
      'pct_change': pct_change,
      'improved': improved,
      'direction': direction,
    }

  return comparisons


def format_metrics_table(metrics: AlgorithmMetrics, name: str = "Algorithm") -> str:
  """
  Format metrics as a readable table string.

  Args:
    metrics: Metrics to format
    name: Algorithm name for header

  Returns:
    Formatted table string
  """
  lines = [
    f"=== {name} Metrics ===",
    "",
    "Tracking Accuracy:",
    f"  RMSE:     {metrics.tracking_error_rmse:.6f}",
    f"  Max:      {metrics.tracking_error_max:.6f}",
    f"  Mean:     {metrics.tracking_error_mean:.6f}",
    "",
    "Output Quality:",
    f"  Smoothness (jerk RMS): {metrics.output_smoothness:.6f}",
    f"  Std Dev:              {metrics.output_std:.6f}",
    "",
    "Latency (ms):",
    f"  Mean:  {metrics.latency_mean_ms:.3f}",
    f"  P50:   {metrics.latency_p50_ms:.3f}",
    f"  P99:   {metrics.latency_p99_ms:.3f}",
    f"  Max:   {metrics.latency_max_ms:.3f}",
    "",
    "Safety:",
    f"  Saturation Ratio: {metrics.saturation_ratio:.2%}",
    f"  Min Safety Margin: {metrics.safety_margin_min:.6f}",
    "",
    "Performance:",
    f"  Total Steps:      {metrics.total_steps}",
    f"  Total Time:       {metrics.total_time_s:.3f}s",
    f"  Steps/Second:     {metrics.steps_per_second:.1f}",
  ]
  return "\n".join(lines)
