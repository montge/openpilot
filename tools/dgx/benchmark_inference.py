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

  models_dir = "selfdrive/modeld/models"

  # driving_policy
  print("\n[driving_policy.onnx]")
  policy = OnnxRunner(f"{models_dir}/driving_policy.onnx")
  policy_inputs = {
    "desire_pulse": Tensor(np.random.randn(1, 25, 8).astype(np.float16)),
    "traffic_convention": Tensor(np.random.randn(1, 2).astype(np.float16)),
    "features_buffer": Tensor(np.random.randn(1, 25, 512).astype(np.float16)),
  }
  stats = benchmark_model(policy, policy_inputs, args.warmup, args.runs)
  print(f"  {stats['mean_ms']:.2f}ms +/- {stats['std_ms']:.2f}ms ({stats['fps']:.1f} FPS)")

  # driving_vision
  print("\n[driving_vision.onnx]")
  vision = OnnxRunner(f"{models_dir}/driving_vision.onnx")
  vision_inputs = {
    "img": Tensor(np.random.randint(0, 255, (1, 12, 128, 256), dtype=np.uint8)),
    "big_img": Tensor(np.random.randint(0, 255, (1, 12, 128, 256), dtype=np.uint8)),
  }
  stats = benchmark_model(vision, vision_inputs, args.warmup, args.runs)
  print(f"  {stats['mean_ms']:.2f}ms +/- {stats['std_ms']:.2f}ms ({stats['fps']:.1f} FPS)")

  # dmonitoring_model
  print("\n[dmonitoring_model.onnx]")
  dmon = OnnxRunner(f"{models_dir}/dmonitoring_model.onnx")
  dmon_empty = dmon.get_empty_input_data()
  dmon_inputs = {}
  for k, v in dmon_empty.items():
    if "float" in str(v.dtype):
      dmon_inputs[k] = Tensor(np.random.randn(*v.shape).astype(np.float32))
    else:
      dmon_inputs[k] = Tensor(np.random.randint(0, 255, v.shape, dtype=np.uint8))
  stats = benchmark_model(dmon, dmon_inputs, args.warmup, args.runs)
  print(f"  {stats['mean_ms']:.2f}ms +/- {stats['std_ms']:.2f}ms ({stats['fps']:.1f} FPS)")

  # Combined pipeline
  print("\n[Combined Vision + Policy Pipeline]")
  times = []
  for _ in range(args.warmup):
    vout = vision(vision_inputs)
    for v in vout.values():
      v.realize()
    pout = policy(policy_inputs)
    for v in pout.values():
      v.realize()

  for _ in range(args.runs):
    start = time.perf_counter()
    vout = vision(vision_inputs)
    for v in vout.values():
      v.realize()
    pout = policy(policy_inputs)
    for v in pout.values():
      v.realize()
    times.append(time.perf_counter() - start)

  mean_ms = np.mean(times) * 1000
  std_ms = np.std(times) * 1000
  fps = 1 / np.mean(times)
  print(f"  {mean_ms:.2f}ms +/- {std_ms:.2f}ms ({fps:.1f} FPS)")

  print("\n" + "=" * 60)
  print("Benchmark complete!")


if __name__ == "__main__":
  main()
