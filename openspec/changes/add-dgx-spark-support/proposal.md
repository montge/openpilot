# Change: Add DGX Spark External GPU Support

## Why

The NVIDIA DGX Spark represents a significant opportunity for openpilot development and experimentation:

1. **Algorithm Research**: Developers can run larger models and experiment with architectures (e.g., MARL, DoRA fine-tuning) that exceed comma device capabilities
2. **Training Acceleration**: Local fine-tuning of driving models using DoRA or similar parameter-efficient techniques
3. **Simulation Performance**: GPU-accelerated simulation for the algorithm test harness
4. **Development Without Hardware**: Full-featured development environment without comma device

The DGX Spark's specifications make it well-suited:
- 128GB unified memory (handles 200B+ parameter models)
- GB10 Grace Blackwell chip (1 PFLOP FP4 performance)
- ARM-based architecture (similar to comma device)
- ConnectX-7 networking (can link two units for larger workloads)
- Compact form factor ($3,999 price point)

## What Changes

- **Add Hardware Abstraction** (`system/hardware/dgx_spark/`):
  - Device detection and capability discovery
  - Memory management for unified architecture
  - GPU compute interface via tinygrad CUDA backend

- **Extend tinygrad Integration** (`selfdrive/modeld/runners/`):
  - Add CUDA backend configuration for DGX Spark
  - Memory-efficient model loading for unified memory
  - Optional FP4/NVFP4 precision support

- **Add Development Mode** (`tools/dgx/`):
  - Model inference benchmarking
  - Training/fine-tuning utilities (DoRA integration)
  - Simulation acceleration hooks

- **Configuration System**:
  - Auto-detection of DGX Spark hardware
  - Fallback to standard PC mode when not present
  - Environment variable overrides for development

## Impact

- Affected specs: New `dgx-spark-integration` capability
- Affected code:
  - `system/hardware/` - New hardware abstraction
  - `selfdrive/modeld/` - tinygrad backend selection
  - `tools/` - New development utilities
- Dependencies: tinygrad (existing), CUDA toolkit (new for DGX)
- **No changes to safety-critical runtime code** (development-only feature)
- **BREAKING**: None (additive changes only)
