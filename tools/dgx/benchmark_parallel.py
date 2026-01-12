#!/usr/bin/env python3
"""Benchmark parallel scenario execution vs CPU baseline.

Measures speedup from using GPUScenarioRunner with multiprocessing
compared to sequential execution.

Usage:
  python tools/dgx/benchmark_parallel.py
  python tools/dgx/benchmark_parallel.py --workers 8 --scenarios 50
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np


def create_synthetic_scenarios(count: int = 20, steps_per_scenario: int = 1000):
  """Create synthetic scenarios for benchmarking.

  Returns list of (scenario_id, data) tuples.
  """
  scenarios = []
  for i in range(count):
    # Simulate scenario data (state arrays)
    data = {
      "v_ego": np.random.uniform(0, 30, steps_per_scenario),
      "a_ego": np.random.uniform(-3, 3, steps_per_scenario),
      "steering_angle": np.random.uniform(-30, 30, steps_per_scenario),
      "yaw_rate": np.random.uniform(-0.5, 0.5, steps_per_scenario),
    }
    scenarios.append((i, data))
  return scenarios


def simulate_algorithm_step(state: dict) -> float:
  """Simulate algorithm computation (PID-like calculation)."""
  # Simple control calculation to simulate algorithm work
  error = state.get("error", 0)
  kp, ki, kd = 0.1, 0.01, 0.05
  prev_error = state.get("prev_error", 0)
  integral = state.get("integral", 0)

  p_term = kp * error
  integral += error
  i_term = ki * integral
  d_term = kd * (error - prev_error)

  return p_term + i_term + d_term


def run_scenario_sequential(scenario_data: tuple) -> dict:
  """Run a single scenario sequentially."""
  scenario_id, data = scenario_data
  steps = len(data["v_ego"])

  outputs = []
  state = {"error": 0, "prev_error": 0, "integral": 0}

  for i in range(steps):
    # Simulate state update
    state["error"] = data["steering_angle"][i] * 0.1
    output = simulate_algorithm_step(state)
    outputs.append(output)
    state["prev_error"] = state["error"]

  return {
    "scenario_id": scenario_id,
    "outputs": outputs,
    "mean_output": float(np.mean(outputs)),
  }


def _worker_run_scenario(args: tuple) -> dict:
  """Worker function for parallel execution."""
  return run_scenario_sequential(args)


def benchmark_sequential(scenarios: list, runs: int = 3) -> dict:
  """Benchmark sequential execution."""
  times = []

  for _ in range(runs):
    start = time.perf_counter()
    _ = [run_scenario_sequential(s) for s in scenarios]
    elapsed = time.perf_counter() - start
    times.append(elapsed)

  return {
    "method": "sequential",
    "workers": 1,
    "scenarios": len(scenarios),
    "runs": runs,
    "mean_time_s": float(np.mean(times)),
    "std_time_s": float(np.std(times)),
    "scenarios_per_second": len(scenarios) / float(np.mean(times)),
  }


def benchmark_parallel(scenarios: list, workers: int, runs: int = 3) -> dict:
  """Benchmark parallel execution with multiprocessing."""
  times = []

  for _ in range(runs):
    start = time.perf_counter()
    with mp.Pool(workers) as pool:
      results = pool.map(_worker_run_scenario, scenarios)  # noqa: F841
    elapsed = time.perf_counter() - start
    times.append(elapsed)

  return {
    "method": "parallel",
    "workers": workers,
    "scenarios": len(scenarios),
    "runs": runs,
    "mean_time_s": float(np.mean(times)),
    "std_time_s": float(np.std(times)),
    "scenarios_per_second": len(scenarios) / float(np.mean(times)),
  }


def benchmark_all(scenarios: list, max_workers: int, runs: int = 3) -> list[dict]:
  """Run benchmarks with different worker counts."""
  results = []

  # Sequential baseline
  print("Running sequential baseline...")
  results.append(benchmark_sequential(scenarios, runs))

  # Parallel with different worker counts
  worker_counts = [2, 4, 8, max_workers] if max_workers > 8 else [2, 4, max_workers]
  worker_counts = sorted({w for w in worker_counts if w <= max_workers})

  for workers in worker_counts:
    print(f"Running parallel with {workers} workers...")
    results.append(benchmark_parallel(scenarios, workers, runs))

  return results


def print_results(results: list[dict]) -> None:
  """Print benchmark results."""
  print("\n" + "=" * 70)
  print("Parallel Scenario Execution Benchmark Results")
  print("=" * 70)

  baseline = results[0]
  baseline_rate = baseline["scenarios_per_second"]

  print(f"\n{'Method':<15} {'Workers':<10} {'Time (s)':<12} {'Rate (/s)':<12} {'Speedup':<10}")
  print("-" * 70)

  for r in results:
    speedup = r["scenarios_per_second"] / baseline_rate
    line = f"{r['method']:<15} {r['workers']:<10} {r['mean_time_s']:<12.3f} "
    line += f"{r['scenarios_per_second']:<12.1f} {speedup:<10.2f}x"
    print(line)

  # Summary
  best = max(results, key=lambda x: x["scenarios_per_second"])
  print("\n" + "-" * 70)
  print(f"Best configuration: {best['workers']} workers")
  print(f"Maximum speedup: {best['scenarios_per_second'] / baseline_rate:.2f}x")
  print(f"Throughput: {best['scenarios_per_second']:.1f} scenarios/second")


def main():
  parser = argparse.ArgumentParser(description="Benchmark parallel scenario execution")
  parser.add_argument("--scenarios", type=int, default=50, help="Number of scenarios")
  parser.add_argument("--steps", type=int, default=1000, help="Steps per scenario")
  parser.add_argument("--workers", type=int, default=mp.cpu_count(), help="Max workers")
  parser.add_argument("--runs", type=int, default=3, help="Benchmark runs")
  args = parser.parse_args()

  print(f"Creating {args.scenarios} synthetic scenarios ({args.steps} steps each)...")
  scenarios = create_synthetic_scenarios(args.scenarios, args.steps)

  print(f"CPU cores available: {mp.cpu_count()}")
  print(f"Max workers to test: {args.workers}")
  print(f"Benchmark runs: {args.runs}")

  results = benchmark_all(scenarios, args.workers, args.runs)
  print_results(results)


if __name__ == "__main__":
  main()
