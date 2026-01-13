"""Benchmarking utilities for model profiling.

Provides latency, memory, and throughput profiling tools.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Check PyTorch availability
try:
  import torch

  TORCH_AVAILABLE = True
except ImportError:
  TORCH_AVAILABLE = False


@dataclass
class LatencyResult:
  """Result from latency profiling.

  Attributes:
    mean_ms: Mean latency in milliseconds
    std_ms: Standard deviation in milliseconds
    min_ms: Minimum latency
    max_ms: Maximum latency
    percentiles: Percentile values (p50, p90, p95, p99)
    num_runs: Number of profiling runs
  """

  mean_ms: float
  std_ms: float
  min_ms: float
  max_ms: float
  percentiles: dict[str, float] = field(default_factory=dict)
  num_runs: int = 0

  def __str__(self) -> str:
    """Format result as string."""
    return (
      f"Latency: {self.mean_ms:.2f} ± {self.std_ms:.2f} ms "
      + f"(min={self.min_ms:.2f}, max={self.max_ms:.2f}, "
      + f"p50={self.percentiles.get('p50', 0):.2f}, "
      + f"p95={self.percentiles.get('p95', 0):.2f})"
    )


@dataclass
class MemoryResult:
  """Result from memory profiling.

  Attributes:
    peak_mb: Peak memory usage in MB
    allocated_mb: Currently allocated memory in MB
    reserved_mb: Reserved memory in MB (GPU only)
    model_mb: Model parameters memory in MB
  """

  peak_mb: float
  allocated_mb: float
  reserved_mb: float = 0.0
  model_mb: float = 0.0

  def __str__(self) -> str:
    """Format result as string."""
    return f"Memory: peak={self.peak_mb:.1f}MB, " + f"allocated={self.allocated_mb:.1f}MB, " + f"model={self.model_mb:.1f}MB"


@dataclass
class ThroughputResult:
  """Result from throughput benchmarking.

  Attributes:
    samples_per_second: Throughput in samples/sec
    batches_per_second: Throughput in batches/sec
    batch_size: Batch size used
    total_samples: Total samples processed
    total_time_s: Total time in seconds
  """

  samples_per_second: float
  batches_per_second: float
  batch_size: int
  total_samples: int
  total_time_s: float

  def __str__(self) -> str:
    """Format result as string."""
    return f"Throughput: {self.samples_per_second:.1f} samples/s ({self.batches_per_second:.1f} batches/s)"


@dataclass
class ModelProfile:
  """Complete model profile.

  Attributes:
    name: Model name
    latency: Latency profiling result
    memory: Memory profiling result
    throughput: Throughput result
    input_shape: Input tensor shape
    device: Device used for profiling
  """

  name: str
  latency: LatencyResult | None = None
  memory: MemoryResult | None = None
  throughput: ThroughputResult | None = None
  input_shape: tuple[int, ...] | None = None
  device: str = "cpu"

  def __str__(self) -> str:
    """Format complete profile."""
    lines = [f"Model Profile: {self.name}"]
    lines.append(f"  Device: {self.device}")
    if self.input_shape:
      lines.append(f"  Input shape: {self.input_shape}")
    if self.latency:
      lines.append(f"  {self.latency}")
    if self.memory:
      lines.append(f"  {self.memory}")
    if self.throughput:
      lines.append(f"  {self.throughput}")
    return "\n".join(lines)


def profile_latency(
  model: Any,
  input_data: np.ndarray | Any,
  num_warmup: int = 10,
  num_runs: int = 100,
  sync_cuda: bool = True,
) -> LatencyResult:
  """Profile model inference latency.

  Args:
    model: Model with __call__ or forward method
    input_data: Input tensor or numpy array
    num_warmup: Number of warmup runs
    num_runs: Number of profiling runs
    sync_cuda: Synchronize CUDA before timing

  Returns:
    LatencyResult with timing statistics
  """
  # Convert numpy to torch if needed
  if TORCH_AVAILABLE and isinstance(input_data, np.ndarray):
    input_tensor = torch.from_numpy(input_data)
  else:
    input_tensor = input_data

  # Get inference function
  if hasattr(model, "forward"):
    infer_fn = model.forward
  elif callable(model):
    infer_fn = model
  else:
    raise ValueError("Model must be callable or have forward method")

  # Warmup
  for _ in range(num_warmup):
    _ = infer_fn(input_tensor)
    if TORCH_AVAILABLE and sync_cuda and torch.cuda.is_available():
      torch.cuda.synchronize()

  # Profile
  latencies = []
  for _ in range(num_runs):
    if TORCH_AVAILABLE and sync_cuda and torch.cuda.is_available():
      torch.cuda.synchronize()

    start = time.perf_counter()
    _ = infer_fn(input_tensor)

    if TORCH_AVAILABLE and sync_cuda and torch.cuda.is_available():
      torch.cuda.synchronize()

    end = time.perf_counter()
    latencies.append((end - start) * 1000)  # Convert to ms

  latencies = np.array(latencies)

  return LatencyResult(
    mean_ms=float(np.mean(latencies)),
    std_ms=float(np.std(latencies)),
    min_ms=float(np.min(latencies)),
    max_ms=float(np.max(latencies)),
    percentiles={
      "p50": float(np.percentile(latencies, 50)),
      "p90": float(np.percentile(latencies, 90)),
      "p95": float(np.percentile(latencies, 95)),
      "p99": float(np.percentile(latencies, 99)),
    },
    num_runs=num_runs,
  )


def profile_memory(
  model: Any,
  input_data: np.ndarray | Any,
  device: str = "cuda",
) -> MemoryResult:
  """Profile model memory usage.

  Args:
    model: Model to profile
    input_data: Input tensor
    device: Device to profile on

  Returns:
    MemoryResult with memory statistics
  """
  if not TORCH_AVAILABLE:
    raise ImportError("PyTorch required for memory profiling")

  # Convert to torch tensor
  if isinstance(input_data, np.ndarray):
    input_tensor = torch.from_numpy(input_data)
  else:
    input_tensor = input_data

  # Calculate model parameter memory
  model_mb = 0.0
  if hasattr(model, "parameters"):
    for param in model.parameters():
      model_mb += param.numel() * param.element_size()
    model_mb /= 1024 * 1024

  if device == "cuda" and torch.cuda.is_available():
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    # Move to GPU
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Run forward pass
    with torch.no_grad():
      _ = model(input_tensor)

    torch.cuda.synchronize()

    return MemoryResult(
      peak_mb=torch.cuda.max_memory_allocated() / 1024 / 1024,
      allocated_mb=torch.cuda.memory_allocated() / 1024 / 1024,
      reserved_mb=torch.cuda.memory_reserved() / 1024 / 1024,
      model_mb=model_mb,
    )
  else:
    # CPU memory profiling is more limited
    import tracemalloc

    tracemalloc.start()

    with torch.no_grad():
      _ = model(input_tensor)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return MemoryResult(
      peak_mb=peak / 1024 / 1024,
      allocated_mb=current / 1024 / 1024,
      model_mb=model_mb,
    )


def benchmark_throughput(
  model: Any,
  input_shape: tuple[int, ...],
  batch_sizes: list[int] | None = None,
  duration_s: float = 5.0,
  device: str = "cuda",
) -> dict[int, ThroughputResult]:
  """Benchmark model throughput at different batch sizes.

  Args:
    model: Model to benchmark
    input_shape: Input shape (without batch dimension)
    batch_sizes: List of batch sizes to test
    duration_s: Duration for each batch size test
    device: Device to run on

  Returns:
    Dictionary mapping batch size to ThroughputResult
  """
  if not TORCH_AVAILABLE:
    raise ImportError("PyTorch required for throughput benchmarking")

  if batch_sizes is None:
    batch_sizes = [1, 2, 4, 8, 16, 32]

  results = {}

  for batch_size in batch_sizes:
    # Create input
    full_shape = (batch_size,) + input_shape
    input_tensor = torch.randn(full_shape)

    if device == "cuda" and torch.cuda.is_available():
      model = model.to(device)
      input_tensor = input_tensor.to(device)

    # Warmup
    with torch.no_grad():
      for _ in range(5):
        _ = model(input_tensor)
        if torch.cuda.is_available():
          torch.cuda.synchronize()

    # Benchmark
    total_samples = 0
    total_batches = 0
    start_time = time.perf_counter()

    while True:
      with torch.no_grad():
        _ = model(input_tensor)
        if torch.cuda.is_available():
          torch.cuda.synchronize()

      total_samples += batch_size
      total_batches += 1

      elapsed = time.perf_counter() - start_time
      if elapsed >= duration_s:
        break

    results[batch_size] = ThroughputResult(
      samples_per_second=total_samples / elapsed,
      batches_per_second=total_batches / elapsed,
      batch_size=batch_size,
      total_samples=total_samples,
      total_time_s=elapsed,
    )

  return results


def profile_model(
  model: Any,
  input_shape: tuple[int, ...],
  batch_size: int = 1,
  device: str = "auto",
  num_latency_runs: int = 100,
  name: str | None = None,
) -> ModelProfile:
  """Complete model profiling.

  Args:
    model: Model to profile
    input_shape: Input shape (without batch dimension)
    batch_size: Batch size for profiling
    device: Device to run on ('auto', 'cuda', 'cpu')
    num_latency_runs: Number of latency profiling runs
    name: Model name for profile

  Returns:
    Complete ModelProfile
  """
  if not TORCH_AVAILABLE:
    raise ImportError("PyTorch required for profiling")

  # Resolve device
  if device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"

  # Create input
  full_shape = (batch_size,) + input_shape
  input_tensor = torch.randn(full_shape)

  if device == "cuda" and torch.cuda.is_available():
    model = model.to(device)
    input_tensor = input_tensor.to(device)

  # Get model name
  if name is None:
    name = model.__class__.__name__

  # Profile latency
  model.eval()
  with torch.no_grad():
    latency = profile_latency(model, input_tensor, num_runs=num_latency_runs)

  # Profile memory
  memory = profile_memory(model, input_tensor, device=device)

  # Single throughput measurement
  throughput_results = benchmark_throughput(
    model,
    input_shape,
    batch_sizes=[batch_size],
    duration_s=2.0,
    device=device,
  )
  throughput = throughput_results.get(batch_size)

  return ModelProfile(
    name=name,
    latency=latency,
    memory=memory,
    throughput=throughput,
    input_shape=full_shape,
    device=device,
  )


def compare_models(
  models: dict[str, Any],
  input_shape: tuple[int, ...],
  batch_size: int = 1,
  device: str = "auto",
) -> dict[str, ModelProfile]:
  """Compare multiple models.

  Args:
    models: Dictionary mapping name to model
    input_shape: Input shape for comparison
    batch_size: Batch size
    device: Device to run on

  Returns:
    Dictionary mapping name to ModelProfile
  """
  profiles = {}
  for name, model in models.items():
    profiles[name] = profile_model(
      model,
      input_shape,
      batch_size=batch_size,
      device=device,
      name=name,
    )
  return profiles


def format_comparison_table(profiles: dict[str, ModelProfile]) -> str:
  """Format model comparison as table.

  Args:
    profiles: Dictionary of model profiles

  Returns:
    Formatted table string
  """
  lines = []
  lines.append(f"{'Model':<20} {'Latency (ms)':<15} {'Memory (MB)':<15} {'Throughput':<15}")
  lines.append("-" * 65)

  for name, profile in profiles.items():
    lat_str = f"{profile.latency.mean_ms:.2f}±{profile.latency.std_ms:.2f}" if profile.latency else "N/A"
    mem_str = f"{profile.memory.peak_mb:.1f}" if profile.memory else "N/A"
    thr_str = f"{profile.throughput.samples_per_second:.1f}/s" if profile.throughput else "N/A"
    lines.append(f"{name:<20} {lat_str:<15} {mem_str:<15} {thr_str:<15}")

  return "\n".join(lines)
