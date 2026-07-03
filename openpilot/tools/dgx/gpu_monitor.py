#!/usr/bin/env python3
"""GPU monitoring utilities for DGX Spark development.

Provides memory profiling and GPU utilization monitoring for:
- Training memory optimization
- Inference performance analysis
- Resource usage tracking
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass


@dataclass
class GPUMemoryInfo:
  """GPU memory usage information."""

  total_mb: float
  used_mb: float
  free_mb: float

  @property
  def used_percent(self) -> float:
    return (self.used_mb / self.total_mb) * 100 if self.total_mb > 0 else 0

  @property
  def free_percent(self) -> float:
    return (self.free_mb / self.total_mb) * 100 if self.total_mb > 0 else 0


@dataclass
class GPUUtilization:
  """GPU utilization metrics."""

  gpu_util_percent: float
  memory_util_percent: float
  temperature_c: float
  power_w: float
  power_limit_w: float


@dataclass
class GPUStatus:
  """Complete GPU status."""

  index: int
  name: str
  memory: GPUMemoryInfo
  utilization: GPUUtilization


def get_gpu_memory(gpu_index: int = 0) -> GPUMemoryInfo | None:
  """Get GPU memory usage.

  Args:
    gpu_index: GPU index to query

  Returns:
    GPUMemoryInfo or None if query fails
  """
  try:
    result = subprocess.run(
      [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
      ],
      capture_output=True,
      text=True,
      check=True,
    )

    values = result.stdout.strip().split(",")
    if len(values) >= 3:
      return GPUMemoryInfo(
        total_mb=float(values[0].strip()),
        used_mb=float(values[1].strip()),
        free_mb=float(values[2].strip()),
      )
  except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
    pass

  return None


def get_gpu_utilization(gpu_index: int = 0) -> GPUUtilization | None:
  """Get GPU utilization metrics.

  Args:
    gpu_index: GPU index to query

  Returns:
    GPUUtilization or None if query fails
  """
  try:
    result = subprocess.run(
      [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
      ],
      capture_output=True,
      text=True,
      check=True,
    )

    values = result.stdout.strip().split(",")
    if len(values) >= 5:
      return GPUUtilization(
        gpu_util_percent=float(values[0].strip()) if values[0].strip() != "[N/A]" else 0,
        memory_util_percent=float(values[1].strip()) if values[1].strip() != "[N/A]" else 0,
        temperature_c=float(values[2].strip()) if values[2].strip() != "[N/A]" else 0,
        power_w=float(values[3].strip()) if values[3].strip() != "[N/A]" else 0,
        power_limit_w=float(values[4].strip()) if values[4].strip() != "[N/A]" else 0,
      )
  except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
    pass

  return None


def get_gpu_status(gpu_index: int = 0) -> GPUStatus | None:
  """Get complete GPU status.

  Args:
    gpu_index: GPU index to query

  Returns:
    GPUStatus or None if query fails
  """
  try:
    result = subprocess.run(
      ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=name", "--format=csv,noheader"],
      capture_output=True,
      text=True,
      check=True,
    )
    name = result.stdout.strip()

    memory = get_gpu_memory(gpu_index)
    utilization = get_gpu_utilization(gpu_index)

    if memory and utilization:
      return GPUStatus(
        index=gpu_index,
        name=name,
        memory=memory,
        utilization=utilization,
      )
  except (subprocess.CalledProcessError, FileNotFoundError):
    pass

  return None


class GPUMonitor:
  """Continuous GPU monitoring."""

  def __init__(self, gpu_index: int = 0, interval_sec: float = 1.0):
    """Initialize GPU monitor.

    Args:
      gpu_index: GPU index to monitor
      interval_sec: Sampling interval in seconds
    """
    self.gpu_index = gpu_index
    self.interval_sec = interval_sec
    self._running = False
    self._samples: list[GPUStatus] = []

  def sample(self) -> GPUStatus | None:
    """Take a single sample."""
    status = get_gpu_status(self.gpu_index)
    if status:
      self._samples.append(status)
    return status

  def clear(self):
    """Clear collected samples."""
    self._samples.clear()

  @property
  def samples(self) -> list[GPUStatus]:
    """Get collected samples."""
    return self._samples.copy()

  def get_peak_memory(self) -> float:
    """Get peak memory usage in MB."""
    if not self._samples:
      return 0
    return max(s.memory.used_mb for s in self._samples)

  def get_avg_utilization(self) -> float:
    """Get average GPU utilization."""
    if not self._samples:
      return 0
    return sum(s.utilization.gpu_util_percent for s in self._samples) / len(self._samples)

  def get_avg_power(self) -> float:
    """Get average power draw in watts."""
    if not self._samples:
      return 0
    return sum(s.utilization.power_w for s in self._samples) / len(self._samples)

  def print_summary(self):
    """Print monitoring summary."""
    if not self._samples:
      print("No samples collected")
      return

    print(f"\nGPU Monitoring Summary ({len(self._samples)} samples)")
    print("=" * 50)
    print(f"GPU: {self._samples[0].name}")
    print(f"Peak Memory: {self.get_peak_memory():.0f} MB")
    print(f"Avg Utilization: {self.get_avg_utilization():.1f}%")
    print(f"Avg Power: {self.get_avg_power():.1f} W")

    # Memory range
    min_mem = min(s.memory.used_mb for s in self._samples)
    max_mem = max(s.memory.used_mb for s in self._samples)
    print(f"Memory Range: {min_mem:.0f} - {max_mem:.0f} MB")

    # Temperature range
    min_temp = min(s.utilization.temperature_c for s in self._samples)
    max_temp = max(s.utilization.temperature_c for s in self._samples)
    print(f"Temperature Range: {min_temp:.0f} - {max_temp:.0f} C")


class MemoryTracker:
  """Context manager for tracking memory usage during operations."""

  def __init__(self, label: str = "", gpu_index: int = 0):
    """Initialize memory tracker.

    Args:
      label: Label for the tracked operation
      gpu_index: GPU index to track
    """
    self.label = label
    self.gpu_index = gpu_index
    self.start_memory: float = 0
    self.end_memory: float = 0
    self.peak_memory: float = 0

  def __enter__(self):
    mem = get_gpu_memory(self.gpu_index)
    self.start_memory = mem.used_mb if mem else 0
    self.peak_memory = self.start_memory
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    mem = get_gpu_memory(self.gpu_index)
    self.end_memory = mem.used_mb if mem else 0
    self.peak_memory = max(self.peak_memory, self.end_memory)

    delta = self.end_memory - self.start_memory
    label = f"[{self.label}] " if self.label else ""
    print(f"{label}Memory: {self.start_memory:.0f} -> {self.end_memory:.0f} MB (delta: {delta:+.0f} MB)")

    return False

  def update_peak(self):
    """Update peak memory (call during operation)."""
    mem = get_gpu_memory(self.gpu_index)
    if mem:
      self.peak_memory = max(self.peak_memory, mem.used_mb)


def print_gpu_status(gpu_index: int = 0):
  """Print current GPU status."""
  status = get_gpu_status(gpu_index)
  if status:
    print(f"\nGPU {status.index}: {status.name}")
    print("=" * 50)
    print(f"Memory: {status.memory.used_mb:.0f} / {status.memory.total_mb:.0f} MB ({status.memory.used_percent:.1f}%)")
    print(f"GPU Utilization: {status.utilization.gpu_util_percent:.0f}%")
    print(f"Memory Utilization: {status.utilization.memory_util_percent:.0f}%")
    print(f"Temperature: {status.utilization.temperature_c:.0f} C")
    print(f"Power: {status.utilization.power_w:.0f} / {status.utilization.power_limit_w:.0f} W")
  else:
    print("Could not get GPU status. Is nvidia-smi available?")


def monitor_continuous(gpu_index: int = 0, interval_sec: float = 1.0, duration_sec: float | None = None):
  """Run continuous monitoring.

  Args:
    gpu_index: GPU index to monitor
    interval_sec: Sampling interval in seconds
    duration_sec: Total duration (None = run until interrupted)
  """
  monitor = GPUMonitor(gpu_index, interval_sec)

  print(f"Monitoring GPU {gpu_index} (Ctrl+C to stop)")
  print("-" * 70)
  print(f"{'Time':>8} {'Memory MB':>12} {'GPU%':>8} {'Mem%':>8} {'Temp C':>8} {'Power W':>10}")
  print("-" * 70)

  start_time = time.monotonic()

  try:
    while True:
      status = monitor.sample()
      if status:
        elapsed = time.monotonic() - start_time
        line = (
          f"{elapsed:>7.1f}s "
          + f"{status.memory.used_mb:>11.0f} "
          + f"{status.utilization.gpu_util_percent:>7.0f} "
          + f"{status.utilization.memory_util_percent:>7.0f} "
          + f"{status.utilization.temperature_c:>7.0f} "
          + f"{status.utilization.power_w:>9.0f}"
        )
        print(line)

      if duration_sec and (time.monotonic() - start_time) >= duration_sec:
        break

      time.sleep(interval_sec)
  except KeyboardInterrupt:
    pass

  print("-" * 70)
  monitor.print_summary()


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="GPU monitoring utilities")
  parser.add_argument("--gpu", type=int, default=0, help="GPU index")
  parser.add_argument("--status", action="store_true", help="Print current GPU status")
  parser.add_argument("--monitor", action="store_true", help="Run continuous monitoring")
  parser.add_argument("--interval", type=float, default=1.0, help="Monitoring interval (seconds)")
  parser.add_argument("--duration", type=float, default=None, help="Monitoring duration (seconds)")
  args = parser.parse_args()

  if args.status:
    print_gpu_status(args.gpu)
  elif args.monitor:
    monitor_continuous(args.gpu, args.interval, args.duration)
  else:
    print_gpu_status(args.gpu)
