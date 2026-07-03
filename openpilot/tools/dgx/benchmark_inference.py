#!/usr/bin/env python3
"""Benchmark openpilot model inference on DGX Spark / NVIDIA GPU."""

import argparse
import os
import time

import numpy as np


def benchmark_model(runner, inputs: dict, warmup: int = 5, runs: int = 20) -> dict:
  """Run benchmark and return timing stats."""
  # Warmup
  for _ in range(warmup):
    out = runner(inputs)
    for v in out.values():
      v.realize()

  # Benchmark
  times = []
  for _ in range(runs):
    start = time.perf_counter()
    out = runner(inputs)
    for v in out.values():
      v.realize()
    times.append(time.perf_counter() - start)

  return {
    "mean_ms": np.mean(times) * 1000,
    "std_ms": np.std(times) * 1000,
    "min_ms": min(times) * 1000,
    "max_ms": max(times) * 1000,
    "fps": 1 / np.mean(times),
  }


def main():
  parser = argparse.ArgumentParser(description="Benchmark openpilot models")
  parser.add_argument("--beam", type=int, default=0, help="BEAM optimization level")
  parser.add_argument("--runs", type=int, default=20, help="Number of benchmark runs")
  parser.add_argument("--warmup", type=int, default=5, help="Number of warmup runs")
  args = parser.parse_args()

  if args.beam > 0:
    os.environ["BEAM"] = str(args.beam)

  from tinygrad import Device, Tensor
  from tinygrad.frontend.onnx import OnnxRunner  # type: ignore[import-not-found]

  Device.DEFAULT = "CUDA"
  print(f"Device: {Device.DEFAULT}")
  print(f"BEAM: {os.environ.get('BEAM', 'disabled')}")
  print("=" * 60)

  models_dir = "openpilot/selfdrive/modeld/models"

  def random_inputs(runner) -> dict:
    inputs = {}
    for k, v in runner.get_empty_input_data().items():
      if "float" in str(v.dtype):
        inputs[k] = Tensor(np.random.randn(*v.shape).astype(np.float32))
      else:
        inputs[k] = Tensor(np.random.randint(0, 255, v.shape, dtype=np.uint8))
    return inputs

  # driving_supercombo: merged vision+policy graph, one pass does everything
  print("\n[driving_supercombo.onnx]")
  supercombo = OnnxRunner(f"{models_dir}/driving_supercombo.onnx")
  stats = benchmark_model(supercombo, random_inputs(supercombo), args.warmup, args.runs)
  print(f"  {stats['mean_ms']:.2f}ms +/- {stats['std_ms']:.2f}ms ({stats['fps']:.1f} FPS)")

  # dmonitoring_model
  print("\n[dmonitoring_model.onnx]")
  dmon = OnnxRunner(f"{models_dir}/dmonitoring_model.onnx")
  stats = benchmark_model(dmon, random_inputs(dmon), args.warmup, args.runs)
  print(f"  {stats['mean_ms']:.2f}ms +/- {stats['std_ms']:.2f}ms ({stats['fps']:.1f} FPS)")

  print("\n" + "=" * 60)
  print("Benchmark complete!")


if __name__ == "__main__":
  main()
