# DGX Spark Development Tools

Development utilities for running openpilot on NVIDIA DGX Spark hardware.

## Hardware Requirements

- NVIDIA DGX Spark with GB10 (Blackwell) GPU
- Or any NVIDIA GPU with CUDA support (for development/testing)

## Quick Start

```bash
# Run the quick start script (creates venv, installs deps, verifies GPU)
./tools/dgx/quickstart.sh

# Or use Python setup script
python tools/dgx/setup.py
```

## Environment Setup

The setup creates an isolated `.venv` inside the openpilot directory:

```bash
# Manual setup
cd /path/to/openpilot
python3 -m venv .venv
source .venv/bin/activate
pip install numpy tinygrad onnx pytest
```

## Checking Your Environment

```bash
# Check GPU and environment
python tools/dgx/setup.py --check

# Verify tinygrad CUDA backend
python tools/dgx/setup.py --verify
```

## GPU Detection

The hardware abstraction layer automatically detects NVIDIA GPUs:

```python
from openpilot.system.hardware.nvidia.gpu import (
    is_nvidia_available,
    is_dgx_spark,
    get_nvidia_gpus,
    get_best_gpu,
    get_recommended_precision,
)

# Check if NVIDIA GPU is available
if is_nvidia_available():
    gpu = get_best_gpu()
    print(f"GPU: {gpu.name}")
    print(f"Compute Capability: {gpu.compute_capability_str}")
    print(f"Recommended Precision: {get_recommended_precision(gpu)}")

# Check specifically for DGX Spark
if is_dgx_spark():
    print("Running on DGX Spark!")
```

## Precision Modes

The GB10 GPU supports multiple precision modes:

| Mode | Description | Compute Capability |
|------|-------------|-------------------|
| FP32 | Full precision | All GPUs |
| FP16 | Half precision | 7.0+ (Volta) |
| BF16 | Brain float 16 | 8.0+ (Ampere) |
| FP8 | 8-bit float | 9.0+ (Hopper) |
| NVFP4 | 4-bit float | 10.0+ (Blackwell) |

Set precision via environment variable:
```bash
export OPENPILOT_DGX_PRECISION=fp16  # or bf16, fp8, fp4
```

## Using tinygrad with CUDA

```python
from tinygrad import Tensor, Device

# Set CUDA as default device
Device.DEFAULT = "CUDA"

# Create tensors (automatically on GPU)
a = Tensor([1.0, 2.0, 3.0])
b = Tensor([4.0, 5.0, 6.0])
c = (a + b).numpy()
```

## Troubleshooting

### nvidia-smi not found
Ensure NVIDIA drivers are installed:
```bash
sudo apt install nvidia-driver-550  # or latest version
```

### CUDA backend not available in tinygrad
Ensure CUDA toolkit is installed:
```bash
# Check CUDA version
nvidia-smi | grep "CUDA Version"

# Install CUDA toolkit if needed
sudo apt install nvidia-cuda-toolkit
```

### Memory shows "Not Supported"
This is normal for unified memory architectures like DGX Spark (GB10).
The CPU and GPU share the same 128GB memory pool.

## Files

```
tools/dgx/
├── __init__.py      # Package init
├── setup.py         # Environment setup script
├── quickstart.sh    # Quick start bash script
└── README.md        # This file
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENPILOT_DGX_ENABLED` | Enable DGX features | `1` (auto-detect) |
| `OPENPILOT_DGX_PRECISION` | Precision mode | Auto (based on GPU) |
| `CUDA` | Force CUDA backend in tinygrad | `1` |
| `DEBUG` | Enable debug output | `0` |
