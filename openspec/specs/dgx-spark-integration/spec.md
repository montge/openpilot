# dgx-spark-integration Specification

## Purpose
TBD - created by archiving change add-dgx-spark-support. Update Purpose after archive.
## Requirements
### Requirement: DGX Spark Hardware Detection
The system SHALL automatically detect when running on NVIDIA DGX Spark hardware.

#### Scenario: DGX Spark detected on compatible hardware
- **GIVEN** the system is running on NVIDIA DGX Spark or GB10-based OEM system
- **WHEN** the hardware abstraction layer initializes
- **THEN** `HARDWARE.is_dgx_spark()` returns `True`
- **AND** GPU capabilities are queried and cached

#### Scenario: DGX Spark detection on incompatible hardware
- **GIVEN** the system is running on PC or comma device
- **WHEN** the hardware abstraction layer initializes
- **THEN** `HARDWARE.is_dgx_spark()` returns `False`
- **AND** standard hardware abstraction is used

### Requirement: CUDA Backend Selection
The model runner SHALL use CUDA backend when running on DGX Spark.

#### Scenario: CUDA backend activated on DGX Spark
- **GIVEN** DGX Spark hardware is detected
- **WHEN** the model runner initializes
- **THEN** tinygrad uses the CUDA backend
- **AND** GPU memory is allocated from unified memory pool

#### Scenario: Fallback to default backend on non-DGX hardware
- **GIVEN** DGX Spark hardware is not detected
- **WHEN** the model runner initializes
- **THEN** tinygrad uses the default backend (QCOM for comma, GPU/CPU for PC)

### Requirement: Unified Memory Optimization
The system SHALL leverage DGX Spark unified memory for efficient CPU-GPU data sharing.

#### Scenario: Zero-copy tensor access on unified memory
- **GIVEN** a tensor allocated in unified memory
- **WHEN** accessed by both CPU and GPU operations
- **THEN** no explicit memory transfer is performed
- **AND** memory bandwidth is optimized for the access pattern

#### Scenario: Fallback to explicit transfers when required
- **GIVEN** unified memory is not available or disabled
- **WHEN** tensors are moved between CPU and GPU
- **THEN** explicit memory copies are performed
- **AND** performance warning is logged

### Requirement: Precision Mode Selection
The model runner SHALL support multiple precision modes for inference.

#### Scenario: FP32 precision mode
- **GIVEN** `OPENPILOT_DGX_PRECISION=fp32` is set
- **WHEN** the model loads
- **THEN** all computations use 32-bit floating point
- **AND** maximum numerical accuracy is maintained

#### Scenario: FP16 precision mode
- **GIVEN** `OPENPILOT_DGX_PRECISION=fp16` is set
- **WHEN** the model loads
- **THEN** computations use 16-bit floating point where safe
- **AND** inference throughput increases with acceptable accuracy loss

#### Scenario: FP4 precision mode (experimental)
- **GIVEN** `OPENPILOT_DGX_PRECISION=fp4` is set
- **WHEN** the model loads
- **THEN** computations use NVFP4 format
- **AND** inference throughput is maximized
- **AND** a warning indicates experimental status

### Requirement: DoRA Fine-Tuning Support
The system SHALL support DoRA (Weight-Decomposed Low-Rank Adaptation) for model fine-tuning.

#### Scenario: Create DoRA adapter for base model
- **GIVEN** a pre-trained supercombo model
- **WHEN** `tools/dgx/finetune.py --create-adapter` is executed
- **THEN** DoRA adapter layers are initialized
- **AND** base model weights are frozen
- **AND** only adapter parameters are trainable

#### Scenario: Fine-tune with route data
- **GIVEN** a DoRA adapter and route log data
- **WHEN** fine-tuning is executed
- **THEN** adapter weights are updated via backpropagation
- **AND** training metrics (loss, learning rate) are logged
- **AND** checkpoints are saved at specified intervals

#### Scenario: Save and load adapters
- **GIVEN** a trained DoRA adapter
- **WHEN** the adapter is saved
- **THEN** only adapter weights are persisted (not base model)
- **AND** adapter file size is <1% of base model size
- **AND** adapter can be loaded and applied to matching base model

#### Scenario: Merge adapter into base model
- **GIVEN** a trained DoRA adapter
- **WHEN** merge operation is executed
- **THEN** adapter weights are folded into base model weights
- **AND** resulting model runs without adapter overhead
- **AND** merged model produces equivalent outputs

### Requirement: Simulation Acceleration
The algorithm test harness SHALL leverage DGX Spark GPU for accelerated simulation.

#### Scenario: GPU-accelerated scenario processing
- **GIVEN** DGX Spark hardware and the algorithm test harness
- **WHEN** a batch of scenarios is executed
- **THEN** scenarios are processed in parallel on GPU
- **AND** throughput exceeds CPU-only baseline by at least 5x

#### Scenario: Automatic device selection
- **GIVEN** the algorithm test harness initializing
- **WHEN** `use_gpu=None` (auto mode)
- **THEN** GPU is used if DGX Spark is detected
- **AND** CPU is used otherwise

### Requirement: Inference Benchmarking
The system SHALL provide benchmarking tools for model inference performance.

#### Scenario: Run inference benchmark
- **GIVEN** a model loaded on DGX Spark
- **WHEN** `tools/dgx/benchmark_inference.py` is executed
- **THEN** inference latency statistics are reported (mean, p50, p99)
- **AND** throughput (inferences/second) is reported
- **AND** GPU utilization and memory usage are reported

#### Scenario: Compare precision modes
- **GIVEN** benchmark results for multiple precision modes
- **WHEN** comparison report is generated
- **THEN** latency/throughput differences are shown
- **AND** accuracy metrics are compared (if ground truth available)

### Requirement: Configuration and Feature Flags
The system SHALL support configuration via environment variables.

#### Scenario: Enable DGX features via environment
- **GIVEN** `OPENPILOT_DGX_ENABLED=1` is set
- **WHEN** the system initializes
- **THEN** DGX Spark features are activated if hardware is compatible

#### Scenario: Disable DGX features via environment
- **GIVEN** `OPENPILOT_DGX_ENABLED=0` is set
- **WHEN** the system initializes on DGX Spark hardware
- **THEN** DGX-specific features are disabled
- **AND** standard PC mode is used

#### Scenario: Invalid configuration handled gracefully
- **GIVEN** invalid configuration values
- **WHEN** the system initializes
- **THEN** a warning is logged
- **AND** default values are used
