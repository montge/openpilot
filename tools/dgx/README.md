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

### TensorRT installation fails
```bash
# Install TensorRT from pip (requires CUDA)
pip install tensorrt

# If pip install fails, try NVIDIA's package
# Download from: https://developer.nvidia.com/tensorrt
```

### Model loading errors (ONNX)
```bash
# Ensure models are downloaded (git-lfs required)
git lfs pull

# Or copy models manually
scp user@source:/path/to/models/*.onnx selfdrive/modeld/models/
```

### Out of memory during training
- Reduce batch size: `--batch-size 16` or `--batch-size 8`
- Use gradient checkpointing (if implemented)
- Monitor memory: `python tools/dgx/gpu_monitor.py --monitor`

### PyTorch not found
Training requires PyTorch:
```bash
pip install torch
# Or with CUDA support:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### SSH connection to DGX
```bash
# Find your SSH key (NVIDIA Sync stores keys here on Windows)
# /mnt/c/Users/<username>/AppData/Local/NVIDIA Corporation/Sync/config/nvsync.key

# Copy key and connect
cp /path/to/nvsync.key ~/.ssh/dgx_key
chmod 600 ~/.ssh/dgx_key
ssh -i ~/.ssh/dgx_key user@dgx-spark-ip
```

### Slow tinygrad performance
tinygrad CUDA is not optimized for Blackwell architecture. Use TensorRT:
```bash
# TensorRT: ~800 FPS
python tools/dgx/benchmark_tensorrt.py

# tinygrad: ~2 FPS (not optimized)
python tools/dgx/benchmark_inference.py
```

### Pre-commit hook failures
```bash
# Ensure mypy is in PATH
export PATH="/home/$USER/.venv/bin:$PATH"

# Common fixes:
# - Add type: ignore comments for torch imports
# - Remove unused imports
# - Make scripts executable: chmod +x script.py
```

### Process killed during training
Check for OOM (Out of Memory):
```bash
dmesg | tail -20  # Check for OOM killer messages
python tools/dgx/gpu_monitor.py --status  # Check GPU memory
```

## Benchmarking

### TensorRT (Recommended)

For maximum performance, use TensorRT:
```bash
pip install tensorrt
python tools/dgx/benchmark_tensorrt.py
```

Options:
- `--runs N`: Number of benchmark runs (default: 100)
- `--warmup N`: Number of warmup runs (default: 20)
- `--fp32`: Use FP32 instead of FP16

### tinygrad CUDA

For tinygrad-based inference:
```bash
python tools/dgx/benchmark_inference.py --runs 20 --warmup 5
```

Options:
- `--runs N`: Number of benchmark runs (default: 20)
- `--warmup N`: Number of warmup runs (default: 5)
- `--beam N`: BEAM optimization level (default: 0, disabled)

### Benchmark Results (2026-01)

Tested on DGX Spark (GB10, Blackwell, compute 12.1):

#### TensorRT FP16 (Recommended)

| Model | Inference | FPS | vs tinygrad |
|-------|-----------|-----|-------------|
| driving_policy | 0.09ms | 11,355 | 659x faster |
| driving_vision | 0.85ms | 1,181 | 432x faster |
| dmonitoring | 0.28ms | 3,584 | 1,175x faster |
| **Combined** | **1.21ms** | **824** | **620x faster** |

**TensorRT is 41x faster than comma 3X** (1.2ms vs ~50ms).

#### tinygrad CUDA (Not Optimized for Blackwell)

| Model | Inference | FPS |
|-------|-----------|-----|
| driving_policy | 58ms | 17.2 |
| driving_vision | 366ms | 2.7 |
| dmonitoring | 328ms | 3.0 |
| **Combined** | **514ms** | **1.9** |

tinygrad's CUDA backend is not yet optimized for Blackwell architecture.
Use TensorRT for production-level performance.

## DoRA Fine-Tuning

The DGX Spark is ideal for fine-tuning openpilot models using DoRA (Weight-Decomposed Low-Rank Adaptation). DoRA achieves efficient fine-tuning by decomposing pretrained weights into magnitude and direction components, then applying low-rank updates only to the direction.

### Why DoRA?

- **Parameter Efficient**: Only ~2-5% of parameters are trained
- **Preserves Base Model**: Original weights are frozen, preventing catastrophic forgetting
- **Fast Training**: With TensorRT teacher, pseudo-label generation runs at 800+ FPS
- **Easy Deployment**: DoRA weights can be merged back into the base model for inference

### Quick Start Training

```bash
# Install PyTorch and training dependencies
pip install torch onnx2pytorch tensorrt

# Run training with dummy data (tests the pipeline)
python tools/dgx/training/train.py --dry-run --epochs 2

# Train with CI test segments
python tools/dgx/training/train.py --data ci --epochs 5

# Train with commaCarSegments dataset
python tools/dgx/training/train.py --data comma_car_segments --epochs 10
```

### Training Configuration

```bash
python tools/dgx/training/train.py \
  --data /path/to/segments \    # Training data path
  --model selfdrive/modeld/models/driving_policy.onnx \
  --epochs 10 \                 # Number of training epochs
  --batch-size 32 \             # Batch size
  --lr 1e-4 \                   # Learning rate
  --dora-rank 16 \              # DoRA rank (higher = more capacity)
  --dora-alpha 1.0 \            # DoRA scaling factor
  --output checkpoints/         # Checkpoint directory
```

### Training Pipeline Architecture

```
Route Logs (rlog.zst)
        │
        ▼
┌─────────────────┐
│  Data Loader    │  Extracts frames, desire, traffic convention
│  (dataloader.py)│  from route segments
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Student Model  │     │  Teacher Model  │
│  (PyTorch+DoRA) │     │  (TensorRT)     │
│  ~2-5% trainable│     │  800+ FPS       │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│         Knowledge Distillation          │
│  - Laplacian NLL (winner-takes-all)     │
│  - Feature matching                     │
│  - Path probability alignment           │
└─────────────────────────────────────────┘
         │
         ▼
   DoRA Checkpoints
   (adapter weights only)
```

### DoRA Components

**DoRALinear**: Linear layer with weight decomposition
```python
from openpilot.tools.dgx.training import DoRALinear, apply_dora_to_model

# Apply DoRA to an existing model
model = apply_dora_to_model(
    model,
    target_modules=["fc", "proj"],  # Layer name patterns to adapt
    rank=16,                         # Low-rank dimension
    alpha=1.0,                       # Scaling factor
)
```

**Loss Functions**:
- `LaplacianNLLLoss`: Robust loss for path predictions (heavier tails than Gaussian)
- `PathDistillationLoss`: Combined loss for path, lane lines, and road edges
- `FeatureDistillationLoss`: Matches intermediate representations

### Available Training Data

| Dataset | Size | Description |
|---------|------|-------------|
| CI Test Segments | ~20 segments | Quick validation from openpilot CI |
| commaCarSegments | 188K segments (3,148 hours) | HuggingFace community dataset |
| comma2k19 | 33 hours (~100GB) | Academic Torrents |

### Merging Weights for Inference

After training, merge DoRA weights back into the base model:

```python
from openpilot.tools.dgx.training import DoRALinear

# Load trained DoRA model
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Merge weights for each DoRA layer
for name, module in model.named_modules():
    if isinstance(module, DoRALinear):
        merged = module.merge_weights()
        # Replace DoRA layer with merged Linear
        parent = get_parent_module(model, name)
        setattr(parent, name.split(".")[-1], merged)

# Export merged model to ONNX
torch.onnx.export(model, dummy_input, "fine_tuned_model.onnx")
```

### Testing DoRA

```bash
# Run DoRA unit tests (requires PyTorch)
pytest tools/dgx/training/test_dora.py -v

# Test dataloader with CI segments
python tools/dgx/training/test_dataloader.py --download --read
```

## Files

```
tools/dgx/
├── __init__.py                 # Package init
├── setup.py                    # Environment setup script
├── quickstart.sh               # Quick start bash script
├── gpu_monitor.py              # GPU memory and utilization monitoring
├── benchmark_inference.py      # tinygrad CUDA benchmark
├── benchmark_tensorrt.py       # TensorRT benchmark (recommended)
├── README.md                   # This file
├── HARDWARE_TEST_CHECKLIST.md  # Manual hardware validation checklist
└── training/
    ├── __init__.py        # Training module init
    ├── dora.py            # DoRA layer implementations
    ├── losses.py          # Training loss functions
    ├── teacher.py         # TensorRT teacher model
    ├── dataloader.py      # Route log dataset loader
    ├── train.py           # Training entry point
    ├── test_dora.py       # DoRA unit tests
    └── test_dataloader.py # Dataloader test script
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENPILOT_DGX_ENABLED` | Enable DGX features | `1` (auto-detect) |
| `OPENPILOT_DGX_PRECISION` | Precision mode | Auto (based on GPU) |
| `CUDA` | Force CUDA backend in tinygrad | `1` |
| `DEBUG` | Enable debug output | `0` |
