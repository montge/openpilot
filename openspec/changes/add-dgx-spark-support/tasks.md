## 1. Hardware Abstraction

- [x] 1.1 Create `system/hardware/nvidia/` directory structure
- [x] 1.2 Implement `NvidiaPC` class extending `HardwareBase`
- [x] 1.3 Implement hardware detection (`is_nvidia_available()`, `is_dgx_spark()`)
- [x] 1.4 Implement capability queries (`GPUInfo` with memory, compute capability, unified memory)
- [x] 1.5 Add NVIDIA GPU to `HARDWARE` singleton selection logic
- [x] 1.6 Add unit tests for hardware detection (with mocking)

## 2. tinygrad CUDA Backend Integration

- [x] 2.1 Configure tinygrad CUDA backend selection for NVIDIA GPUs
- [x] 2.2 Add `NVIDIAGPU` / `CUDAGPU` environment variable support to modeld
- [x] 2.3 Test supercombo model loading on CUDA backend (driving_policy, driving_vision, dmonitoring)
- [x] 2.4 Implement precision detection (`supports_fp16`, `supports_bf16`, `supports_fp8`, `supports_nvfp4`)
- [x] 2.5 Benchmark inference performance vs comma device baseline (GB10: 1.9 FPS vs comma 3X: 20 FPS)

## 3. Model Loading and Inference

- [ ] 3.1 Extend model runner to support precision selection
- [ ] 3.2 Implement memory-efficient model loading for unified memory
- [ ] 3.3 Add model warm-up and performance profiling
- [x] 3.4 Create inference benchmark script (`tools/dgx/benchmark_inference.py`)
- [x] 3.5 Document performance characteristics (see tools/dgx/README.md)

## 4. DoRA Fine-Tuning Support

- [x] 4.1 Create `tools/dgx/` directory structure
- [ ] 4.2 Implement `DoRAAdapter` class for weight decomposition
- [ ] 4.3 Implement `DoRALayer` for low-rank adaptation
- [ ] 4.4 Create dataset loader for route logs → training data
- [ ] 4.5 Implement fine-tuning entry point (`tools/dgx/finetune.py`)
- [ ] 4.6 Implement adapter checkpoint save/load
- [ ] 4.7 Add adapter merging utility (merge DoRA weights into base model)
- [ ] 4.8 Add unit tests for DoRA implementation

## 5. Algorithm Harness Integration

- [ ] 5.1 Add GPU acceleration option to `ScenarioRunner`
- [ ] 5.2 Implement batch scenario processing on GPU
- [ ] 5.3 Add CUDA memory management for large scenario sets
- [ ] 5.4 Benchmark simulation acceleration vs CPU baseline
- [ ] 5.5 Update algorithm harness documentation

## 6. Development Utilities

- [x] 6.1 Create `tools/dgx/setup.py` for environment setup
- [x] 6.2 Add CUDA toolkit version checking
- [ ] 6.3 Create memory profiling utility
- [ ] 6.4 Add GPU utilization monitoring
- [x] 6.5 Create quick-start script for DGX Spark development

## 7. Configuration and Feature Flags

- [x] 7.1 Add `NVIDIAGPU` / `CUDAGPU` environment variable support
- [x] 7.2 Add precision recommendation based on compute capability
- [x] 7.3 Add fallback logic when NVIDIA features unavailable
- [x] 7.4 Document all configuration options

## 8. Testing

- [x] 8.1 Add pytest fixtures for NVIDIA testing (with hardware mocking)
- [ ] 8.2 Add integration tests for model inference
- [ ] 8.3 Add tests for DoRA adapter correctness
- [ ] 8.4 Add CI job for NVIDIA compatibility (software-only checks)
- [ ] 8.5 Create manual test checklist for hardware validation

## 9. Documentation

- [x] 9.1 Add NVIDIA/DGX setup guide to tools/dgx/README.md
- [ ] 9.2 Document DoRA fine-tuning workflow
- [ ] 9.3 Add troubleshooting guide for common issues
- [ ] 9.4 Update CLAUDE.md with NVIDIA development notes
