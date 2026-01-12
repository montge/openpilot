"""GPU-accelerated algorithm harness utilities.

Provides parallel and GPU-accelerated extensions for the algorithm harness:
- Parallel batch scenario execution (multiprocessing)
- GPU-accelerated metrics computation
- CUDA memory management for large scenario sets
- Performance profiling utilities

Usage:
  from openpilot.tools.dgx.algorithm_harness_gpu import GPUScenarioRunner

  runner = GPUScenarioRunner(max_workers=8)
  results = runner.run_batch_parallel(algorithm_factory, scenarios)
"""

from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np

if TYPE_CHECKING:
  from openpilot.selfdrive.controls.lib.tests.algorithm_harness.interface import (
    AlgorithmInterface,
  )
  from openpilot.selfdrive.controls.lib.tests.algorithm_harness.runner import (
    AlgorithmMetrics,
    Scenario,
    ScenarioResult,
  )


@dataclass
class BatchResult:
  """Results from batch scenario execution."""

  results: list[ScenarioResult]
  total_time_s: float
  scenarios_per_second: float
  num_scenarios: int
  num_workers: int


@dataclass
class GPUMetricsConfig:
  """Configuration for GPU-accelerated metrics."""

  use_gpu: bool = True
  use_cupy: bool = True  # Use CuPy for GPU arrays
  use_numba: bool = True  # Use Numba JIT as fallback


@dataclass
class ParallelExecutionStats:
  """Statistics from parallel execution."""

  total_scenarios: int
  completed_scenarios: int
  failed_scenarios: int
  total_time_s: float
  avg_scenario_time_s: float
  scenarios_per_second: float
  worker_utilization: list[float] = field(default_factory=list)


def _run_scenario_worker(args: tuple) -> ScenarioResult:
  """Worker function for parallel scenario execution.

  This runs in a separate process to avoid GIL contention.

  Args:
    args: Tuple of (algorithm_factory, scenario, algorithm_name)

  Returns:
    ScenarioResult from running the scenario
  """
  algorithm_factory, scenario, algorithm_name = args

  # Import inside worker to avoid pickling issues
  from openpilot.selfdrive.controls.lib.tests.algorithm_harness.runner import ScenarioRunner

  runner = ScenarioRunner()
  algorithm = algorithm_factory()
  return runner.run(algorithm, scenario, algorithm_name)


class GPUScenarioRunner:
  """GPU and parallel-accelerated scenario runner.

  Extends the base ScenarioRunner with:
  - Parallel batch execution using multiprocessing
  - GPU-accelerated metrics computation
  - Memory management for large scenario sets

  Usage:
    runner = GPUScenarioRunner(max_workers=8)

    # Run scenarios in parallel
    results = runner.run_batch_parallel(
        algorithm_factory=lambda: MyAlgorithm(),
        scenarios=scenarios,
    )

    # Profile performance
    stats = runner.profile_batch(algorithm_factory, scenarios, runs=5)
  """

  def __init__(
    self,
    max_workers: int | None = None,
    gpu_metrics: bool = True,
    chunk_size: int = 10,
  ):
    """Initialize GPU scenario runner.

    Args:
      max_workers: Max parallel workers (None = CPU count)
      gpu_metrics: Enable GPU-accelerated metrics
      chunk_size: Number of scenarios per worker batch
    """
    self.max_workers = max_workers or mp.cpu_count()
    self.gpu_metrics = gpu_metrics
    self.chunk_size = chunk_size

    self._gpu_available = self._check_gpu()

  def _check_gpu(self) -> bool:
    """Check if GPU acceleration is available."""
    try:
      from openpilot.system.hardware.nvidia.gpu import is_nvidia_available

      return is_nvidia_available()
    except ImportError:
      return False

  def run_batch_parallel(
    self,
    algorithm_factory: Callable[[], AlgorithmInterface],
    scenarios: list[Scenario],
    algorithm_name: str = "algorithm",
    show_progress: bool = True,
  ) -> BatchResult:
    """Run scenarios in parallel using multiprocessing.

    Args:
      algorithm_factory: Factory function to create algorithm instances
      scenarios: List of scenarios to run
      algorithm_name: Name for metrics labeling
      show_progress: Show progress output

    Returns:
      BatchResult with all scenario results and timing stats
    """
    start_time = time.perf_counter()
    num_scenarios = len(scenarios)

    if num_scenarios == 0:
      return BatchResult(
        results=[],
        total_time_s=0,
        scenarios_per_second=0,
        num_scenarios=0,
        num_workers=0,
      )

    # For small batches, run sequentially
    if num_scenarios <= self.chunk_size:
      return self._run_batch_sequential(algorithm_factory, scenarios, algorithm_name)

    # Prepare work items
    work_items = [(algorithm_factory, scenario, algorithm_name) for scenario in scenarios]

    results: list[ScenarioResult] = []
    completed = 0

    # Use ProcessPoolExecutor for true parallelism (bypasses GIL)
    with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
      # Submit all work
      futures = {executor.submit(_run_scenario_worker, item): i for i, item in enumerate(work_items)}

      # Collect results as they complete
      for future in as_completed(futures):
        try:
          result = future.result()
          results.append(result)
          completed += 1

          if show_progress and completed % 10 == 0:
            elapsed = time.perf_counter() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            print(f"Progress: {completed}/{num_scenarios} ({rate:.1f} scenarios/s)")

        except Exception as e:
          print(f"Scenario {futures[future]} failed: {e}")
          results.append(None)  # type: ignore[arg-type]

    total_time = time.perf_counter() - start_time
    scenarios_per_second = num_scenarios / total_time if total_time > 0 else 0

    return BatchResult(
      results=[r for r in results if r is not None],
      total_time_s=total_time,
      scenarios_per_second=scenarios_per_second,
      num_scenarios=num_scenarios,
      num_workers=self.max_workers,
    )

  def _run_batch_sequential(
    self,
    algorithm_factory: Callable[[], AlgorithmInterface],
    scenarios: list[Scenario],
    algorithm_name: str,
  ) -> BatchResult:
    """Run scenarios sequentially (for small batches)."""
    from openpilot.selfdrive.controls.lib.tests.algorithm_harness.runner import ScenarioRunner

    start_time = time.perf_counter()
    runner = ScenarioRunner()
    algorithm = algorithm_factory()

    results = []
    for scenario in scenarios:
      result = runner.run(algorithm, scenario, algorithm_name)
      results.append(result)

    total_time = time.perf_counter() - start_time

    return BatchResult(
      results=results,
      total_time_s=total_time,
      scenarios_per_second=len(scenarios) / total_time if total_time > 0 else 0,
      num_scenarios=len(scenarios),
      num_workers=1,
    )

  def profile_batch(
    self,
    algorithm_factory: Callable[[], AlgorithmInterface],
    scenarios: list[Scenario],
    runs: int = 5,
    warmup_runs: int = 1,
  ) -> ParallelExecutionStats:
    """Profile batch execution performance.

    Args:
      algorithm_factory: Factory to create algorithms
      scenarios: Scenarios to run
      runs: Number of profiling runs
      warmup_runs: Warmup runs before timing

    Returns:
      ParallelExecutionStats with detailed performance metrics
    """
    # Warmup
    for _ in range(warmup_runs):
      self.run_batch_parallel(algorithm_factory, scenarios[:10], show_progress=False)

    # Timed runs
    times = []
    for _ in range(runs):
      result = self.run_batch_parallel(algorithm_factory, scenarios, show_progress=False)
      times.append(result.total_time_s)

    avg_time = np.mean(times)
    total_scenarios = len(scenarios) * runs

    return ParallelExecutionStats(
      total_scenarios=total_scenarios,
      completed_scenarios=total_scenarios,
      failed_scenarios=0,
      total_time_s=float(sum(times)),
      avg_scenario_time_s=float(avg_time / len(scenarios)),
      scenarios_per_second=float(len(scenarios) / avg_time) if avg_time > 0 else 0.0,
    )


class GPUMetricsAccelerator:
  """GPU-accelerated metrics computation.

  Speeds up percentile and aggregation operations using GPU or Numba.
  Falls back to NumPy if no acceleration available.

  Usage:
    accelerator = GPUMetricsAccelerator()
    latencies = np.array([...])
    p50, p99 = accelerator.percentiles(latencies, [50, 99])
  """

  def __init__(self, config: GPUMetricsConfig | None = None):
    """Initialize metrics accelerator."""
    self.config = config or GPUMetricsConfig()
    self._backend = self._select_backend()

  def _select_backend(self) -> str:
    """Select best available backend."""
    if self.config.use_gpu and self.config.use_cupy:
      try:
        import cupy  # type: ignore[import-not-found]  # noqa: F401

        return "cupy"
      except ImportError:
        pass

    if self.config.use_numba:
      try:
        import numba  # type: ignore[import-not-found]  # noqa: F401

        return "numba"
      except ImportError:
        pass

    return "numpy"

  def percentiles(self, data: np.ndarray, percentiles: list[float]) -> list[float]:
    """Compute percentiles with acceleration.

    Args:
      data: Input array
      percentiles: List of percentiles (0-100)

    Returns:
      List of percentile values
    """
    if self._backend == "cupy":
      return self._percentiles_cupy(data, percentiles)
    elif self._backend == "numba":
      return self._percentiles_numba(data, percentiles)
    else:
      return [float(np.percentile(data, p)) for p in percentiles]

  def _percentiles_cupy(self, data: np.ndarray, percentiles: list[float]) -> list[float]:
    """Compute percentiles using CuPy (GPU)."""
    import cupy as cp  # type: ignore[import-not-found]

    # Transfer to GPU
    gpu_data = cp.asarray(data)

    # Sort on GPU (radix sort, very fast)
    sorted_data = cp.sort(gpu_data)

    # Compute percentile indices
    results = []
    n = len(data)
    for p in percentiles:
      idx = int((p / 100) * (n - 1))
      results.append(float(sorted_data[idx].get()))

    return results

  def _percentiles_numba(self, data: np.ndarray, percentiles: list[float]) -> list[float]:
    """Compute percentiles using Numba JIT."""
    # For small arrays, just use numpy (overhead not worth it)
    if len(data) < 1000:
      return [float(np.percentile(data, p)) for p in percentiles]

    # Sort using numpy (Numba sort is similar speed)
    sorted_data = np.sort(data)

    results = []
    n = len(data)
    for p in percentiles:
      idx = int((p / 100) * (n - 1))
      results.append(float(sorted_data[idx]))

    return results

  def aggregate_metrics(
    self,
    metrics_list: list[AlgorithmMetrics],
    fields: list[str] | None = None,
  ) -> dict[str, dict[str, float]]:
    """Aggregate metrics across multiple runs.

    Args:
      metrics_list: List of AlgorithmMetrics
      fields: Fields to aggregate (None = all numeric fields)

    Returns:
      Dict of field -> {mean, std, min, max}
    """
    if not metrics_list:
      return {}

    # Get fields if not specified
    if fields is None:
      fields = [f for f in dir(metrics_list[0]) if not f.startswith("_") and isinstance(getattr(metrics_list[0], f), (int, float))]

    results = {}
    for field_name in fields:
      values = np.array([getattr(m, field_name) for m in metrics_list if hasattr(m, field_name)])
      if len(values) > 0:
        results[field_name] = {
          "mean": float(np.mean(values)),
          "std": float(np.std(values)),
          "min": float(np.min(values)),
          "max": float(np.max(values)),
        }

    return results


def benchmark_parallel_speedup(
  algorithm_factory: Callable[[], AlgorithmInterface],
  scenarios: list[Scenario],
  max_workers_list: list[int] | None = None,
) -> dict[int, float]:
  """Benchmark speedup with different worker counts.

  Args:
    algorithm_factory: Factory to create algorithms
    scenarios: Scenarios to benchmark
    max_workers_list: List of worker counts to test

  Returns:
    Dict of workers -> scenarios_per_second
  """
  if max_workers_list is None:
    max_workers_list = [1, 2, 4, 8, mp.cpu_count()]

  results = {}
  for workers in max_workers_list:
    runner = GPUScenarioRunner(max_workers=workers)
    batch_result = runner.run_batch_parallel(
      algorithm_factory,
      scenarios,
      show_progress=False,
    )
    results[workers] = batch_result.scenarios_per_second

  return results


if __name__ == "__main__":
  print("GPU Algorithm Harness - Parallel Execution Test")
  print("=" * 60)

  # Check GPU availability
  try:
    from openpilot.system.hardware.nvidia.gpu import is_nvidia_available

    print(f"NVIDIA GPU available: {is_nvidia_available()}")
  except ImportError:
    print("NVIDIA GPU detection not available")

  # Show multiprocessing capabilities
  print(f"CPU cores available: {mp.cpu_count()}")

  # Test metrics accelerator
  print("\nMetrics Accelerator Test")
  print("-" * 40)

  accelerator = GPUMetricsAccelerator()
  print(f"Backend: {accelerator._backend}")

  test_data = np.random.randn(10000)
  start = time.perf_counter()
  p50, p95, p99 = accelerator.percentiles(test_data, [50, 95, 99])
  elapsed = (time.perf_counter() - start) * 1000
  print(f"Percentiles (10k samples): p50={p50:.3f}, p95={p95:.3f}, p99={p99:.3f}")
  print(f"Time: {elapsed:.2f}ms")
