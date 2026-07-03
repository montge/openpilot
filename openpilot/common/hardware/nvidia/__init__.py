"""
NVIDIA GPU hardware abstraction for openpilot.

Provides support for:
- DGX Spark (GB10 Grace Blackwell)
- Desktop NVIDIA GPUs (RTX 3090, 4090, etc.)
- Other CUDA-capable devices

This module enables GPU-accelerated inference and training
for development and experimentation.
"""

from openpilot.common.hardware.nvidia.hardware import NvidiaPC
from openpilot.common.hardware.nvidia.gpu import (
  get_nvidia_gpus,
  is_nvidia_available,
  is_dgx_spark,
  get_cuda_version,
  GPUInfo,
)

__all__ = [
  'NvidiaPC',
  'get_nvidia_gpus',
  'is_nvidia_available',
  'is_dgx_spark',
  'get_cuda_version',
  'GPUInfo',
]
