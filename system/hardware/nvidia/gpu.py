"""
NVIDIA GPU detection and capability queries.

Provides utilities to detect NVIDIA GPUs, query their capabilities,
and determine optimal runtime configuration.
"""

import os
import subprocess
import shutil
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass
class GPUInfo:
  """Information about an NVIDIA GPU."""
  index: int
  name: str
  uuid: str
  memory_total_mb: int
  memory_free_mb: int
  compute_capability: tuple[int, int]
  driver_version: str
  cuda_version: str
  is_dgx_spark: bool = False
  is_unified_memory: bool = False

  @property
  def memory_total_gb(self) -> float:
    """Total memory in GB."""
    return self.memory_total_mb / 1024

  @property
  def memory_free_gb(self) -> float:
    """Free memory in GB."""
    return self.memory_free_mb / 1024

  @property
  def compute_capability_str(self) -> str:
    """Compute capability as string (e.g., '8.9')."""
    return f"{self.compute_capability[0]}.{self.compute_capability[1]}"

  def supports_fp16(self) -> bool:
    """Check if GPU supports FP16 tensor cores."""
    # Volta (7.0+) and later support FP16
    return self.compute_capability >= (7, 0)

  def supports_bf16(self) -> bool:
    """Check if GPU supports BF16."""
    # Ampere (8.0+) and later support BF16
    return self.compute_capability >= (8, 0)

  def supports_fp8(self) -> bool:
    """Check if GPU supports FP8."""
    # Hopper (9.0+) and later support FP8
    return self.compute_capability >= (9, 0)

  def supports_nvfp4(self) -> bool:
    """Check if GPU supports NVFP4 (Blackwell)."""
    # Blackwell (10.0+) supports NVFP4
    return self.compute_capability >= (10, 0)


def _run_nvidia_smi(args: list[str]) -> Optional[str]:
  """Run nvidia-smi with given arguments."""
  nvidia_smi = shutil.which('nvidia-smi')
  if nvidia_smi is None:
    return None

  try:
    result = subprocess.run(
      [nvidia_smi] + args,
      capture_output=True,
      text=True,
      timeout=5,
    )
    if result.returncode == 0:
      return result.stdout.strip()
  except (subprocess.TimeoutExpired, subprocess.SubprocessError):
    pass
  return None


@lru_cache(maxsize=1)
def is_nvidia_available() -> bool:
  """Check if NVIDIA GPU is available."""
  # Check for nvidia-smi
  if shutil.which('nvidia-smi') is None:
    return False

  # Try to query GPU
  output = _run_nvidia_smi(['--query-gpu=name', '--format=csv,noheader'])
  return output is not None and len(output) > 0


@lru_cache(maxsize=1)
def get_cuda_version() -> Optional[str]:
  """Get CUDA version from nvidia-smi."""
  output = _run_nvidia_smi(['--query-gpu=driver_version', '--format=csv,noheader'])
  if output is None:
    return None

  # Also try to get CUDA version from nvidia-smi
  full_output = _run_nvidia_smi([])
  if full_output and 'CUDA Version:' in full_output:
    for line in full_output.split('\n'):
      if 'CUDA Version:' in line:
        parts = line.split('CUDA Version:')
        if len(parts) > 1:
          return parts[1].strip().split()[0]
  return None


@lru_cache(maxsize=1)
def get_nvidia_gpus() -> list[GPUInfo]:
  """Get information about all NVIDIA GPUs."""
  if not is_nvidia_available():
    return []

  # Query GPU information
  query = 'index,name,uuid,memory.total,memory.free,driver_version'
  output = _run_nvidia_smi([
    f'--query-gpu={query}',
    '--format=csv,noheader,nounits'
  ])

  if output is None:
    return []

  gpus = []
  cuda_version = get_cuda_version() or "unknown"

  for line in output.strip().split('\n'):
    if not line.strip():
      continue

    parts = [p.strip() for p in line.split(',')]
    if len(parts) < 6:
      continue

    index = int(parts[0])
    name = parts[1]
    uuid = parts[2]
    memory_total = int(parts[3])
    memory_free = int(parts[4])
    driver_version = parts[5]

    # Determine compute capability from GPU name
    compute_cap = _get_compute_capability(name)

    # Check for DGX Spark (GB10 Grace Blackwell)
    is_dgx = 'GB10' in name.upper() or 'DGX' in name.upper() or 'BLACKWELL' in name.upper()

    # Check for unified memory (Grace Hopper, DGX Spark)
    is_unified = 'GRACE' in name.upper() or is_dgx

    gpus.append(GPUInfo(
      index=index,
      name=name,
      uuid=uuid,
      memory_total_mb=memory_total,
      memory_free_mb=memory_free,
      compute_capability=compute_cap,
      driver_version=driver_version,
      cuda_version=cuda_version,
      is_dgx_spark=is_dgx,
      is_unified_memory=is_unified,
    ))

  return gpus


def _get_compute_capability(gpu_name: str) -> tuple[int, int]:
  """
  Determine compute capability from GPU name.

  This is a heuristic based on known GPU architectures.
  For precise detection, use CUDA runtime queries.
  """
  name_upper = gpu_name.upper()

  # Blackwell (10.x) - 2024+
  if any(x in name_upper for x in ['B100', 'B200', 'GB10', 'GB100', 'GB200', 'BLACKWELL', 'RTX 50']):
    return (10, 0)

  # Hopper (9.0) - 2022+
  if any(x in name_upper for x in ['H100', 'H200', 'GH100', 'GH200', 'HOPPER']):
    return (9, 0)

  # Ada Lovelace (8.9) - RTX 40 series
  if any(x in name_upper for x in ['RTX 40', 'L40', 'ADA']):
    return (8, 9)

  # Ampere (8.6) - RTX 30 series
  if any(x in name_upper for x in ['RTX 30', 'A100', 'A40', 'A30', 'A10', 'A16', 'A6000', 'A5000', 'A4000']):
    return (8, 6)

  # Ampere (8.0) - A100
  if 'A100' in name_upper:
    return (8, 0)

  # Turing (7.5) - RTX 20 series
  if any(x in name_upper for x in ['RTX 20', 'T4', 'TURING', 'TITAN RTX', 'QUADRO RTX']):
    return (7, 5)

  # Volta (7.0)
  if any(x in name_upper for x in ['V100', 'VOLTA', 'TITAN V']):
    return (7, 0)

  # Pascal (6.1) - GTX 10 series
  if any(x in name_upper for x in ['GTX 10', 'P100', 'P40', 'P4', 'PASCAL', 'TITAN X']):
    return (6, 1)

  # Default to Maxwell (5.2)
  return (5, 2)


@lru_cache(maxsize=1)
def is_dgx_spark() -> bool:
  """Check if running on DGX Spark."""
  gpus = get_nvidia_gpus()
  return any(gpu.is_dgx_spark for gpu in gpus)


def get_best_gpu() -> Optional[GPUInfo]:
  """Get the best available GPU for inference."""
  gpus = get_nvidia_gpus()
  if not gpus:
    return None

  # Prefer DGX Spark, then highest compute capability, then most memory
  return max(gpus, key=lambda g: (
    g.is_dgx_spark,
    g.compute_capability,
    g.memory_total_mb,
  ))


def get_recommended_precision(gpu: Optional[GPUInfo] = None) -> str:
  """Get recommended precision for a GPU."""
  if gpu is None:
    gpu = get_best_gpu()

  if gpu is None:
    return 'fp32'

  if gpu.supports_nvfp4():
    return 'fp4'  # Blackwell
  elif gpu.supports_fp8():
    return 'fp8'  # Hopper
  elif gpu.supports_bf16():
    return 'bf16'  # Ampere+
  elif gpu.supports_fp16():
    return 'fp16'  # Volta+
  else:
    return 'fp32'


def get_tinygrad_device() -> str:
  """Get the appropriate tinygrad device string for NVIDIA GPU."""
  if not is_nvidia_available():
    return 'CPU'

  # Check environment override
  if 'DEV' in os.environ:
    return os.environ['DEV']

  return 'CUDA'
