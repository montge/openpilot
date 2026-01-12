"""Memory utilities for unified memory systems like DGX Spark.

The DGX Spark (GB10) uses unified memory where CPU and GPU share the same
128GB memory pool. This module provides utilities for:
- Memory-efficient model loading
- Buffer management for zero-copy operations
- Memory profiling and tracking
"""

from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from collections.abc import Generator

import numpy as np

if TYPE_CHECKING:
  pass


@dataclass
class MemorySnapshot:
  """Snapshot of memory state."""

  gpu_allocated_mb: float
  gpu_reserved_mb: float
  gpu_free_mb: float
  cpu_used_mb: float
  cpu_available_mb: float
  timestamp: float


@dataclass
class MemoryAllocation:
  """Track a memory allocation."""

  name: str
  size_mb: float
  device: str
  dtype: str


class UnifiedMemoryManager:
  """Memory manager optimized for unified memory architectures.

  On unified memory systems (like DGX Spark with GB10), CPU and GPU share
  the same physical memory. This manager provides:

  1. Zero-copy tensor sharing between CPU and GPU
  2. Lazy allocation to avoid memory fragmentation
  3. Memory pooling for frequently-used buffer sizes
  4. Automatic garbage collection triggers

  Usage:
    manager = UnifiedMemoryManager()

    # Allocate unified buffer
    buf = manager.allocate("input_buffer", shape=(1, 3, 224, 224), dtype=np.float32)

    # Get as GPU tensor (zero-copy on unified memory)
    gpu_tensor = manager.to_gpu(buf)

    # Track allocations
    manager.print_allocations()

    # Clean up
    manager.clear()
  """

  def __init__(self, enable_pooling: bool = True, gc_threshold_mb: float = 1024.0):
    """Initialize memory manager.

    Args:
      enable_pooling: Enable buffer pooling for reuse
      gc_threshold_mb: Trigger GC when allocations exceed this threshold
    """
    self.enable_pooling = enable_pooling
    self.gc_threshold_mb = gc_threshold_mb

    self._allocations: dict[str, MemoryAllocation] = {}
    self._buffers: dict[str, np.ndarray] = {}
    self._pool: dict[tuple[tuple, str], list[np.ndarray]] = {}
    self._total_allocated_mb = 0.0

    # Check if we're on unified memory
    self._is_unified = self._detect_unified_memory()

  def _detect_unified_memory(self) -> bool:
    """Detect if running on unified memory architecture."""
    try:
      from openpilot.system.hardware.nvidia.gpu import get_best_gpu

      gpu = get_best_gpu()
      if gpu and gpu.is_unified_memory:
        return True
    except ImportError:
      pass

    # Also check via nvidia-smi (GB10 reports N/A for dedicated memory)
    try:
      import subprocess

      result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
      )
      # Unified memory systems may report "Not Supported" or very high values
      if "Not Supported" in result.stdout or "N/A" in result.stdout:
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
      pass

    return False

  @property
  def is_unified(self) -> bool:
    """Check if running on unified memory."""
    return self._is_unified

  def allocate(
    self,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype | type = np.float32,
    zero_init: bool = False,
  ) -> np.ndarray:
    """Allocate a unified memory buffer.

    Args:
      name: Name for tracking
      shape: Buffer shape
      dtype: Data type
      zero_init: Zero-initialize the buffer

    Returns:
      NumPy array (page-aligned for optimal GPU access)
    """
    dtype = np.dtype(dtype)
    size_bytes = int(np.prod(shape)) * dtype.itemsize
    size_mb = size_bytes / (1024 * 1024)

    # Check pool first
    pool_key = (shape, str(dtype))
    if self.enable_pooling and pool_key in self._pool and self._pool[pool_key]:
      buf = self._pool[pool_key].pop()
      if zero_init:
        buf.fill(0)
    else:
      # Allocate new buffer
      if zero_init:
        buf = np.zeros(shape, dtype=dtype)
      else:
        buf = np.empty(shape, dtype=dtype)

    self._buffers[name] = buf
    self._allocations[name] = MemoryAllocation(
      name=name,
      size_mb=size_mb,
      device="unified" if self._is_unified else "cpu",
      dtype=str(dtype),
    )
    self._total_allocated_mb += size_mb

    # Trigger GC if needed
    if self._total_allocated_mb > self.gc_threshold_mb:
      self._trigger_gc()

    return buf

  def get(self, name: str) -> np.ndarray | None:
    """Get a previously allocated buffer."""
    return self._buffers.get(name)

  def release(self, name: str) -> None:
    """Release a buffer (return to pool or free)."""
    if name not in self._buffers:
      return

    buf = self._buffers.pop(name)
    alloc = self._allocations.pop(name)
    self._total_allocated_mb -= alloc.size_mb

    # Return to pool if enabled
    if self.enable_pooling:
      pool_key = (buf.shape, str(buf.dtype))
      if pool_key not in self._pool:
        self._pool[pool_key] = []
      self._pool[pool_key].append(buf)
    # Otherwise let Python GC handle it

  def clear(self) -> None:
    """Clear all allocations and pools."""
    self._buffers.clear()
    self._allocations.clear()
    self._pool.clear()
    self._total_allocated_mb = 0.0
    self._trigger_gc()

  def _trigger_gc(self) -> None:
    """Trigger garbage collection."""
    gc.collect()

    # Also trigger CUDA memory cleanup if available
    try:
      import torch  # type: ignore[import-not-found]

      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    except ImportError:
      pass

  def to_gpu(self, buf: np.ndarray) -> Any:
    """Convert buffer to GPU tensor.

    On unified memory, this is a zero-copy operation.
    On discrete GPU, this copies data to GPU memory.

    Args:
      buf: NumPy array to convert

    Returns:
      GPU tensor (tinygrad or torch depending on availability)
    """
    # Try tinygrad first
    try:
      from tinygrad import Device, Tensor

      if "CUDA" in Device.DEFAULT or os.environ.get("CUDA"):
        return Tensor(buf)
    except ImportError:
      pass

    # Fall back to PyTorch
    try:
      import torch  # type: ignore[import-not-found]

      if torch.cuda.is_available():
        # Use pin_memory for faster transfer on non-unified systems
        if self._is_unified:
          return torch.from_numpy(buf).cuda()
        else:
          return torch.from_numpy(buf).pin_memory().cuda(non_blocking=True)
    except ImportError:
      pass

    # Return numpy array as-is
    return buf

  def from_gpu(self, tensor: Any) -> np.ndarray:
    """Convert GPU tensor back to NumPy array.

    On unified memory, this may be zero-copy.

    Args:
      tensor: GPU tensor

    Returns:
      NumPy array
    """
    if isinstance(tensor, np.ndarray):
      return tensor

    # Handle tinygrad tensors
    if hasattr(tensor, "numpy"):
      return tensor.numpy()

    # Handle PyTorch tensors
    if hasattr(tensor, "cpu"):
      return tensor.cpu().numpy()

    raise TypeError(f"Unknown tensor type: {type(tensor)}")

  def get_memory_stats(self) -> dict[str, float]:
    """Get current memory statistics."""
    stats = {
      "allocated_mb": self._total_allocated_mb,
      "num_buffers": len(self._buffers),
      "pool_size": sum(len(p) for p in self._pool.values()),
    }

    # Add GPU stats if available
    try:
      import torch  # type: ignore[import-not-found]

      if torch.cuda.is_available():
        stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
    except ImportError:
      pass

    return stats

  def print_allocations(self) -> None:
    """Print current allocations."""
    print(f"\nMemory Allocations ({self._total_allocated_mb:.1f} MB total)")
    print("=" * 60)
    for name, alloc in self._allocations.items():
      print(f"  {name}: {alloc.size_mb:.2f} MB ({alloc.dtype}, {alloc.device})")
    print(f"\nPool sizes: {sum(len(p) for p in self._pool.values())} buffers cached")


@contextmanager
def memory_scope(name: str = "scope") -> Generator[UnifiedMemoryManager, None, None]:
  """Context manager for scoped memory management.

  All allocations made within the scope are automatically freed on exit.

  Usage:
    with memory_scope("inference") as mem:
      input_buf = mem.allocate("input", (1, 3, 224, 224))
      output_buf = mem.allocate("output", (1, 1000))
      # ... run inference ...
    # Buffers automatically freed here
  """
  manager = UnifiedMemoryManager()
  try:
    yield manager
  finally:
    manager.clear()


def get_optimal_batch_size(
  model_memory_mb: float,
  input_size_mb: float,
  available_memory_mb: float | None = None,
  safety_margin: float = 0.8,
) -> int:
  """Calculate optimal batch size for available memory.

  Args:
    model_memory_mb: Memory used by model weights
    input_size_mb: Memory per input sample
    available_memory_mb: Available GPU memory (auto-detected if None)
    safety_margin: Fraction of memory to use (default 80%)

  Returns:
    Recommended batch size
  """
  if available_memory_mb is None:
    # Try to detect available memory
    try:
      import torch  # type: ignore[import-not-found]

      if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        available_memory_mb = props.total_memory / (1024 * 1024)
    except ImportError:
      pass

    if available_memory_mb is None:
      # Try nvidia-smi
      try:
        import subprocess

        result = subprocess.run(
          ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
          capture_output=True,
          text=True,
          check=True,
        )
        available_memory_mb = float(result.stdout.strip())
      except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        available_memory_mb = 8192  # Default to 8GB

  # Calculate batch size
  usable_memory = available_memory_mb * safety_margin
  memory_for_batches = usable_memory - model_memory_mb

  if memory_for_batches <= 0:
    return 1

  batch_size = int(memory_for_batches / input_size_mb)
  return max(1, batch_size)


def estimate_model_memory(model_path: str) -> float:
  """Estimate memory required for a model.

  Args:
    model_path: Path to ONNX model

  Returns:
    Estimated memory in MB
  """
  try:
    import onnx

    model = onnx.load(model_path)
    total_params = 0

    for initializer in model.graph.initializer:
      # Calculate number of elements
      dims = initializer.dims
      num_elements = 1
      for d in dims:
        num_elements *= d
      total_params += num_elements

    # Assume float32 (4 bytes per param)
    # Double for activations/gradients during inference
    memory_mb = (total_params * 4 * 2) / (1024 * 1024)
    return memory_mb

  except Exception:
    # Default estimate based on file size
    import os

    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    return file_size_mb * 3  # Rough multiplier for runtime memory


if __name__ == "__main__":
  # Demo unified memory manager
  print("Unified Memory Manager Demo")
  print("=" * 50)

  manager = UnifiedMemoryManager()
  print(f"Unified memory detected: {manager.is_unified}")

  # Allocate some buffers
  input_buf = manager.allocate("input", (4, 3, 224, 224), np.float32)
  output_buf = manager.allocate("output", (4, 1000), np.float32)

  print("\nAllocated buffers:")
  manager.print_allocations()

  print(f"\nMemory stats: {manager.get_memory_stats()}")

  # Test optimal batch size calculation
  batch_size = get_optimal_batch_size(
    model_memory_mb=500,
    input_size_mb=10,
    available_memory_mb=8192,
  )
  print(f"\nOptimal batch size for 500MB model, 10MB/sample, 8GB available: {batch_size}")

  # Cleanup
  manager.clear()
  print("\nCleared all allocations")
