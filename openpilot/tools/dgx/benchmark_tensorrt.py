#!/usr/bin/env python3
"""Benchmark openpilot model inference using TensorRT on DGX Spark / NVIDIA GPU.

TensorRT provides optimized inference that is significantly faster than tinygrad CUDA.
On DGX Spark GB10 (Blackwell), TensorRT achieves 800+ FPS vs tinygrad's ~2 FPS.

Usage:
  python tools/dgx/benchmark_tensorrt.py
  python tools/dgx/benchmark_tensorrt.py --fp32  # Use FP32 instead of FP16
  python tools/dgx/benchmark_tensorrt.py --runs 50 --warmup 10
"""

import argparse
import os
import time

import numpy as np


def build_engine(onnx_path: str, fp16: bool = True, verbose: bool = False):
  """Build TensorRT engine from ONNX model."""
  import tensorrt as trt  # type: ignore[import-not-found]

  logger = trt.Logger(trt.Logger.WARNING if verbose else trt.Logger.ERROR)
  builder = trt.Builder(logger)
  network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  parser = trt.OnnxParser(network, logger)

  with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
      for i in range(parser.num_errors):
        print(f"  Parse error: {parser.get_error(i)}")
      return None

  config = builder.create_builder_config()
  config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
  if fp16:
    config.set_flag(trt.BuilderFlag.FP16)

  serialized = builder.build_serialized_network(network, config)
  runtime = trt.Runtime(logger)
  return runtime.deserialize_cuda_engine(serialized)


def benchmark_engine(engine, warmup: int = 10, runs: int = 100) -> dict:
  """Run inference benchmark on TensorRT engine."""
  import tensorrt as trt  # type: ignore[import-not-found]

  context = engine.create_execution_context()

  # Allocate buffers using numpy (works with unified memory on DGX Spark)
  buffers = {}
  for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    dtype = trt.nptype(engine.get_tensor_dtype(name))

    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
      if dtype == np.float16:
        buf = np.random.randn(*shape).astype(np.float16)
      elif dtype == np.uint8:
        buf = np.random.randint(0, 255, shape).astype(np.uint8)
      else:
        buf = np.random.randn(*shape).astype(dtype)
    else:
      buf = np.zeros(shape, dtype=dtype)

    buf = np.ascontiguousarray(buf)
    buffers[name] = buf
    context.set_tensor_address(name, buf.ctypes.data)

  # Warmup
  for _ in range(warmup):
    context.execute_async_v3(0)

  # Benchmark
  times = []
  for _ in range(runs):
    start = time.perf_counter()
    context.execute_async_v3(0)
    times.append(time.perf_counter() - start)

  return {
    "mean_ms": np.mean(times) * 1000,
    "std_ms": np.std(times) * 1000,
    "min_ms": min(times) * 1000,
    "max_ms": max(times) * 1000,
    "fps": 1000 / (np.mean(times) * 1000),
  }


def main():
  parser = argparse.ArgumentParser(description="Benchmark openpilot models with TensorRT")
  parser.add_argument("--fp32", action="store_true", help="Use FP32 instead of FP16")
  parser.add_argument("--runs", type=int, default=100, help="Number of benchmark runs")
  parser.add_argument("--warmup", type=int, default=20, help="Number of warmup runs")
  parser.add_argument("--verbose", action="store_true", help="Verbose TensorRT output")
  args = parser.parse_args()

  try:
    import tensorrt as trt  # type: ignore[import-not-found]
  except ImportError:
    print("ERROR: TensorRT not installed. Install with: pip install tensorrt")
    return 1

  print(f"TensorRT version: {trt.__version__}")
  print(f"Precision: {'FP32' if args.fp32 else 'FP16'}")
  print("=" * 60)

  models_dir = "selfdrive/modeld/models"
  models = [
    ("driving_policy.onnx", f"{models_dir}/driving_policy.onnx"),
    ("driving_vision.onnx", f"{models_dir}/driving_vision.onnx"),
    ("dmonitoring_model.onnx", f"{models_dir}/dmonitoring_model.onnx"),
  ]

  results = {}
  for name, path in models:
    if not os.path.exists(path):
      print(f"\n[{name}] NOT FOUND - skipping")
      continue

    print(f"\n[{name}]")
    print("  Building TensorRT engine...")

    start = time.perf_counter()
    engine = build_engine(path, fp16=not args.fp32, verbose=args.verbose)
    build_time = time.perf_counter() - start

    if engine:
      print(f"  Built in {build_time:.1f}s ({engine.num_io_tensors} tensors)")
      stats = benchmark_engine(engine, warmup=args.warmup, runs=args.runs)
      print(f"  Inference: {stats['mean_ms']:.3f}ms +/- {stats['std_ms']:.3f}ms ({stats['fps']:.1f} FPS)")
      results[name] = stats["mean_ms"]
    else:
      print("  FAILED to build engine")

  if not results:
    print("\nNo models benchmarked successfully")
    return 1

  print("\n" + "=" * 60)
  total = sum(results.values())
  print(f"[Combined Pipeline]: {total:.3f}ms ({1000 / total:.1f} FPS)")

  # Comparison with tinygrad
  print("\n[Comparison with tinygrad CUDA]")
  tinygrad_times = {
    "driving_policy.onnx": 58,
    "driving_vision.onnx": 366,
    "dmonitoring_model.onnx": 328,
  }
  for name, ms in results.items():
    tinygrad_ms = tinygrad_times.get(name, 0)
    if tinygrad_ms > 0:
      speedup = tinygrad_ms / ms
      print(f"  {name}: {speedup:.0f}x faster")

  tinygrad_total = sum(tinygrad_times.values())
  print(f"\n  Combined: {tinygrad_total / total:.0f}x faster than tinygrad CUDA")

  # Comparison with comma 3X
  comma3x_ms = 50  # ~20 FPS target
  ratio = comma3x_ms / total
  print(f"  vs comma 3X (~50ms): {ratio:.1f}x faster")

  print("\n" + "=" * 60)
  return 0


if __name__ == "__main__":
  exit(main())
