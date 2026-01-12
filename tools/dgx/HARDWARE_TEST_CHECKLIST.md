# DGX Spark Hardware Validation Checklist

Manual test checklist for validating openpilot on NVIDIA DGX Spark hardware.
Run these tests after setting up a new DGX environment or after significant code changes.

## Prerequisites

- [ ] SSH access to DGX Spark established
- [ ] openpilot repository cloned
- [ ] `./tools/dgx/quickstart.sh` completed successfully
- [ ] `nvidia-smi` shows GPU available

## 1. Environment Setup

### 1.1 GPU Detection
```bash
python -c "from openpilot.system.hardware.nvidia.gpu import get_nvidia_gpus; print(get_nvidia_gpus())"
```
- [ ] GPU list is not empty
- [ ] GPU name matches expected hardware (e.g., "NVIDIA GB10")
- [ ] Compute capability is detected (e.g., 12.1 for Blackwell)

### 1.2 DGX Spark Detection
```bash
python -c "from openpilot.system.hardware.nvidia.gpu import is_dgx_spark; print(f'DGX Spark: {is_dgx_spark()}')"
```
- [ ] Returns `True` on DGX Spark hardware
- [ ] Returns `False` on other NVIDIA GPUs

### 1.3 Precision Detection
```bash
python -c "
from openpilot.system.hardware.nvidia.gpu import get_best_gpu, get_recommended_precision
gpu = get_best_gpu()
print(f'Recommended precision: {get_recommended_precision(gpu)}')
print(f'Supports FP16: {gpu.supports_fp16}')
print(f'Supports BF16: {gpu.supports_bf16}')
print(f'Supports FP8: {gpu.supports_fp8}')
print(f'Supports NVFP4: {gpu.supports_nvfp4}')
"
```
- [ ] FP16 supported (compute 7.0+)
- [ ] BF16 supported (compute 8.0+)
- [ ] FP8 supported (compute 9.0+)
- [ ] NVFP4 supported (compute 10.0+, Blackwell only)

### 1.4 Hardware Abstraction
```bash
python -c "
from openpilot.system.hardware import HARDWARE
print(f'Hardware type: {type(HARDWARE).__name__}')
print(f'EON: {HARDWARE.get_device_type()}')
"
```
- [ ] Hardware type is `NvidiaPC` when on DGX
- [ ] Falls back gracefully on non-NVIDIA systems

## 2. Model Inference

### 2.1 TensorRT Benchmark (Recommended)
```bash
python tools/dgx/benchmark_tensorrt.py --runs 50 --warmup 10
```
Expected results (DGX Spark GB10):
- [ ] driving_policy: < 0.5ms (>2000 FPS)
- [ ] driving_vision: < 2ms (>500 FPS)
- [ ] dmonitoring: < 1ms (>1000 FPS)
- [ ] Combined: < 3ms (>300 FPS)
- [ ] No CUDA errors or warnings

### 2.2 tinygrad CUDA Benchmark
```bash
python tools/dgx/benchmark_inference.py --runs 10 --warmup 3
```
- [ ] Models load without errors
- [ ] Inference completes (performance will be slow, ~2 FPS expected)
- [ ] No memory errors

### 2.3 Model Loading
```bash
python -c "
from tinygrad import Device
Device.DEFAULT = 'CUDA'
# Attempt to load each model
import onnx
models = [
    'selfdrive/modeld/models/driving_policy.onnx',
    'selfdrive/modeld/models/driving_vision.onnx',
    'selfdrive/modeld/models/dmonitoring_model.onnx',
]
for m in models:
    try:
        onnx.load(m)
        print(f'{m}: OK')
    except Exception as e:
        print(f'{m}: FAILED - {e}')
"
```
- [ ] All models load successfully
- [ ] No ONNX parsing errors

## 3. GPU Monitoring

### 3.1 Status Check
```bash
python tools/dgx/gpu_monitor.py --status
```
- [ ] GPU name displayed correctly
- [ ] Memory usage shown (may show 0 for unified memory)
- [ ] Temperature displayed
- [ ] Power draw displayed

### 3.2 Continuous Monitoring
```bash
python tools/dgx/gpu_monitor.py --monitor --duration 10
```
- [ ] Samples collected every second
- [ ] Summary displayed at end
- [ ] No errors during monitoring

### 3.3 Memory Tracking
```bash
python -c "
from openpilot.tools.dgx.gpu_monitor import MemoryTracker
with MemoryTracker('test'):
    import torch
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x.T
"
```
- [ ] Memory delta shown
- [ ] No CUDA OOM errors

## 4. Training Pipeline

### 4.1 DoRA Layer Tests
```bash
pytest tools/dgx/training/test_dora.py -v
```
- [ ] All tests pass
- [ ] No CUDA errors

### 4.2 Dataloader Test
```bash
python tools/dgx/training/test_dataloader.py --download --read
```
- [ ] CI segment downloads successfully
- [ ] Route log parsed correctly
- [ ] modelV2 messages extracted

### 4.3 Training Dry Run
```bash
python tools/dgx/training/train.py --dry-run --epochs 1
```
- [ ] Model loads and converts to PyTorch
- [ ] DoRA layers applied correctly
- [ ] Forward pass completes
- [ ] Backward pass completes
- [ ] No OOM errors

### 4.4 Training with Data (Optional)
```bash
python tools/dgx/training/train.py --data ci --epochs 2 --batch-size 8
```
- [ ] Data loads from CI segments
- [ ] Loss decreases over epochs
- [ ] Checkpoints saved
- [ ] No training crashes

## 5. Memory Management

### 5.1 Large Batch Test
```bash
python -c "
import torch
# Test unified memory with large tensors
sizes = [1000, 5000, 10000, 20000]
for s in sizes:
    try:
        x = torch.randn(s, s, device='cuda')
        print(f'{s}x{s} tensor ({x.numel() * 4 / 1e9:.2f} GB): OK')
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'{s}x{s} tensor: FAILED - {e}')
        break
"
```
- [ ] Can allocate tensors up to available memory
- [ ] No unexpected OOM with unified memory

### 5.2 Memory Cleanup
```bash
python -c "
import torch
import gc
x = torch.randn(10000, 10000, device='cuda')
print(f'Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
del x
gc.collect()
torch.cuda.empty_cache()
print(f'After cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
"
```
- [ ] Memory properly freed after deletion
- [ ] `empty_cache()` reclaims memory

## 6. Integration Tests

### 6.1 Hardware Selection
```bash
python -c "
from openpilot.system.hardware import HARDWARE
from openpilot.system.hardware.nvidia import NvidiaPC
assert isinstance(HARDWARE, NvidiaPC), 'Expected NvidiaPC'
print('Hardware selection: OK')
"
```
- [ ] Correct hardware class selected

### 6.2 Model Inference Integration
```bash
# Run a short inference session
python tools/dgx/benchmark_tensorrt.py --runs 5 --warmup 2
```
- [ ] All models run without errors
- [ ] Results are numerically reasonable (not NaN/Inf)

## 7. Stress Tests (Optional)

### 7.1 Extended Inference
```bash
python tools/dgx/benchmark_tensorrt.py --runs 1000 --warmup 50
```
- [ ] No degradation over time
- [ ] Memory stable
- [ ] No thermal throttling (check with `nvidia-smi`)

### 7.2 Training Stability
```bash
python tools/dgx/training/train.py --data ci --epochs 20 --batch-size 16
```
- [ ] Training completes without crashes
- [ ] Loss converges
- [ ] GPU temperature stable

## Sign-off

| Test Section | Pass/Fail | Notes |
|--------------|-----------|-------|
| 1. Environment Setup | | |
| 2. Model Inference | | |
| 3. GPU Monitoring | | |
| 4. Training Pipeline | | |
| 5. Memory Management | | |
| 6. Integration Tests | | |
| 7. Stress Tests | | |

**Tester**: _______________
**Date**: _______________
**DGX Model**: _______________
**Driver Version**: _______________
**CUDA Version**: _______________
