#!/usr/bin/env python3
"""
DGX Spark environment setup script.

Sets up the development environment for openpilot on NVIDIA DGX Spark.
Creates an isolated .venv, installs dependencies, and verifies GPU access.

Usage:
  python tools/dgx/setup.py           # Full setup
  python tools/dgx/setup.py --check   # Check environment only
  python tools/dgx/setup.py --verify  # Verify GPU and tinygrad
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
  """Run a command and optionally check for errors."""
  print(f"  Running: {' '.join(cmd)}")
  return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def check_nvidia_gpu() -> dict | None:
  """Check for NVIDIA GPU and return info."""
  nvidia_smi = shutil.which('nvidia-smi')
  if nvidia_smi is None:
    return None

  try:
    result = subprocess.run([nvidia_smi, '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
      parts = [p.strip() for p in result.stdout.strip().split(',')]
      return {
        'name': parts[0] if len(parts) > 0 else 'Unknown',
        'driver': parts[1] if len(parts) > 1 else 'Unknown',
        'memory': parts[2] if len(parts) > 2 else 'Unknown',
      }
  except Exception:
    pass
  return None


def check_cuda_version() -> str | None:
  """Get CUDA version from nvidia-smi."""
  nvidia_smi = shutil.which('nvidia-smi')
  if nvidia_smi is None:
    return None

  try:
    result = subprocess.run([nvidia_smi], capture_output=True, text=True, timeout=5)
    if result.returncode == 0 and 'CUDA Version:' in result.stdout:
      for line in result.stdout.split('\n'):
        if 'CUDA Version:' in line:
          parts = line.split('CUDA Version:')
          if len(parts) > 1:
            return parts[1].strip().split()[0]
  except Exception:
    pass
  return None


def is_dgx_spark(gpu_info: dict | None) -> bool:
  """Check if running on DGX Spark (GB10 GPU)."""
  if gpu_info is None:
    return False
  name = gpu_info.get('name', '').upper()
  return 'GB10' in name or 'BLACKWELL' in name


def get_compute_capability(gpu_name: str) -> tuple[int, int]:
  """Determine compute capability from GPU name."""
  name_upper = gpu_name.upper()

  if any(x in name_upper for x in ['B100', 'B200', 'GB10', 'GB100', 'GB200', 'BLACKWELL', 'RTX 50']):
    return (10, 0)  # Blackwell
  if any(x in name_upper for x in ['H100', 'H200', 'GH100', 'GH200', 'HOPPER']):
    return (9, 0)  # Hopper
  if any(x in name_upper for x in ['RTX 40', 'L40', 'ADA']):
    return (8, 9)  # Ada Lovelace
  if any(x in name_upper for x in ['RTX 30', 'A100', 'A40', 'A30', 'A10']):
    return (8, 6)  # Ampere
  return (7, 0)  # Default to Volta


def check_environment() -> bool:
  """Check the current environment."""
  print("\n=== DGX Spark Environment Check ===\n")

  # Check GPU
  gpu_info = check_nvidia_gpu()
  if gpu_info is None:
    print("[ ] NVIDIA GPU: Not detected")
    return False

  print(f"[x] NVIDIA GPU: {gpu_info['name']}")
  print(f"    Driver: {gpu_info['driver']}")
  print(f"    Memory: {gpu_info['memory']}")

  # Check CUDA
  cuda_version = check_cuda_version()
  if cuda_version:
    print(f"[x] CUDA Version: {cuda_version}")
  else:
    print("[ ] CUDA Version: Not detected")

  # Check if DGX Spark
  if is_dgx_spark(gpu_info):
    print("[x] DGX Spark (GB10): Detected")
    compute_cap = get_compute_capability(gpu_info['name'])
    print(f"    Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    print("    Supports NVFP4: Yes")
  else:
    print("[ ] DGX Spark (GB10): Not detected (other NVIDIA GPU)")

  # Check Python
  print(f"\n[x] Python: {sys.version.split()[0]}")

  # Check if in venv
  if sys.prefix != sys.base_prefix:
    print(f"[x] Virtual Environment: {sys.prefix}")
  else:
    print("[ ] Virtual Environment: Not activated")

  return True


def verify_tinygrad() -> bool:
  """Verify tinygrad CUDA backend."""
  print("\n=== Verifying tinygrad CUDA Backend ===\n")

  try:
    from tinygrad import Tensor, Device

    # Set CUDA as default
    Device.DEFAULT = "CUDA"
    print(f"[x] tinygrad loaded, device: {Device.DEFAULT}")

    # Test tensor operation
    a = Tensor([1.0, 2.0, 3.0, 4.0])
    b = Tensor([5.0, 6.0, 7.0, 8.0])
    c = (a + b).numpy()
    expected = [6.0, 8.0, 10.0, 12.0]

    if all(abs(c[i] - expected[i]) < 0.01 for i in range(4)):
      print("[x] CUDA tensor operations: Working")
      return True
    else:
      print(f"[ ] CUDA tensor operations: Incorrect result {c}")
      return False

  except ImportError as e:
    print(f"[ ] tinygrad not installed: {e}")
    return False
  except Exception as e:
    print(f"[ ] tinygrad CUDA error: {e}")
    return False


def setup_venv(project_root: Path) -> bool:
  """Create and set up virtual environment."""
  print("\n=== Setting Up Virtual Environment ===\n")

  venv_path = project_root / '.venv'

  # Check if .venv exists and is a symlink (from git)
  if venv_path.is_symlink():
    print(f"  Removing symlink: {venv_path}")
    venv_path.unlink()

  if not venv_path.exists():
    print(f"  Creating venv: {venv_path}")
    run_cmd([sys.executable, '-m', 'venv', str(venv_path)])

  # Get venv python
  venv_python = venv_path / 'bin' / 'python'
  if not venv_python.exists():
    print("[ ] Failed to create venv")
    return False

  print(f"[x] Virtual environment: {venv_path}")

  # Upgrade pip
  print("  Upgrading pip...")
  run_cmd([str(venv_python), '-m', 'pip', 'install', '--upgrade', 'pip', 'wheel', 'setuptools'], capture=True)

  # Install core dependencies
  print("  Installing core dependencies...")
  deps = ['numpy', 'tinygrad', 'onnx', 'pytest']
  run_cmd([str(venv_python), '-m', 'pip', 'install'] + deps, capture=True)

  print(f"[x] Dependencies installed: {', '.join(deps)}")
  return True


def main():
  parser = argparse.ArgumentParser(description='DGX Spark environment setup')
  parser.add_argument('--check', action='store_true', help='Check environment only')
  parser.add_argument('--verify', action='store_true', help='Verify GPU and tinygrad')
  args = parser.parse_args()

  # Find project root
  script_path = Path(__file__).resolve()
  project_root = script_path.parent.parent.parent
  os.chdir(project_root)

  print(f"Project root: {project_root}")

  if args.check:
    success = check_environment()
    sys.exit(0 if success else 1)

  if args.verify:
    check_environment()
    success = verify_tinygrad()
    sys.exit(0 if success else 1)

  # Full setup
  if not check_environment():
    print("\nEnvironment check failed. Please ensure NVIDIA drivers are installed.")
    sys.exit(1)

  if not setup_venv(project_root):
    print("\nFailed to set up virtual environment.")
    sys.exit(1)

  print("\n=== Setup Complete ===")
  print("\nTo activate the environment:")
  print(f"  source {project_root / '.venv' / 'bin' / 'activate'}")
  print("\nTo verify tinygrad CUDA:")
  print("  python tools/dgx/setup.py --verify")


if __name__ == '__main__':
  main()
