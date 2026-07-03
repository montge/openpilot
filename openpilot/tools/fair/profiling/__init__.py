"""Model profiling utilities for FAIR models.

Provides tools for measuring model performance including:
- Latency profiling
- Memory usage
- FLOPs estimation
- Throughput benchmarking
"""

from openpilot.tools.fair.profiling.benchmark import (
  profile_latency,
  profile_memory,
  benchmark_throughput,
  ModelProfile,
)
from openpilot.tools.fair.profiling.flops import (
  estimate_flops,
  count_parameters,
  ModelStats,
)

__all__ = [
  "profile_latency",
  "profile_memory",
  "benchmark_throughput",
  "ModelProfile",
  "estimate_flops",
  "count_parameters",
  "ModelStats",
]
