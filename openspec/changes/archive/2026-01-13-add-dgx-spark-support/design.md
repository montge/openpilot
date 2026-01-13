## Context

openpilot currently targets comma device hardware (comma 3X) with Qualcomm Snapdragon SoC and Adreno GPU. The codebase uses tinygrad for ML inference, which abstracts GPU backends. Adding DGX Spark support enables:

- Algorithm experimentation with larger models
- DoRA-based fine-tuning of driving models
- MARL research requiring significant compute
- GPU-accelerated simulation

**Stakeholders**: Algorithm researchers, model developers, contributors without comma hardware

## Goals / Non-Goals

**Goals**:
- Enable DGX Spark as a first-class development platform
- Support running existing openpilot models on DGX Spark
- Enable training/fine-tuning workflows
- Accelerate simulation and testing
- Maintain compatibility with existing PC development

**Non-Goals**:
- Production deployment on DGX Spark (it's not a vehicle-mountable device)
- Replacing comma device hardware abstraction
- Supporting other NVIDIA desktop GPUs (future work)
- Real-time safety guarantees on DGX Spark

## Decisions

### Decision 1: Hardware Abstraction Layer

Extend the existing `system/hardware/` abstraction:

```
system/hardware/
├── __init__.py           # HARDWARE singleton selection
├── base.py               # HardwareBase abstract class
├── pc/                   # Existing PC implementation
├── tici/                 # comma device implementation
└── dgx_spark/            # NEW: DGX Spark implementation
    ├── __init__.py
    ├── hardware.py       # DGXSparkHardware class
    └── capabilities.py   # GPU/memory capability queries
```

**Detection logic** (in `__init__.py`):
```python
def get_hardware():
    if is_dgx_spark():
        return DGXSparkHardware()
    elif is_tici():
        return TICIHardware()
    else:
        return PCHardware()

def is_dgx_spark():
    # Check for GB10 chip via /proc/device-tree or nvidia-smi
    return Path('/proc/device-tree/model').read_text().startswith('NVIDIA DGX Spark')
```

**Rationale**: Follows existing pattern. DGX Spark is ARM-based like comma device but with CUDA GPU.

### Decision 2: tinygrad Backend Selection

Leverage tinygrad's multi-backend support:

```python
# In modeld initialization
if HARDWARE.is_dgx_spark():
    os.environ['TINYGRAD_BACKEND'] = 'CUDA'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
elif HARDWARE.is_tici():
    os.environ['TINYGRAD_BACKEND'] = 'QCOM'
else:
    os.environ['TINYGRAD_BACKEND'] = 'GPU'  # OpenCL fallback
```

**Unified Memory Optimization**:
The DGX Spark's 128GB unified memory allows zero-copy tensor sharing between CPU and GPU. tinygrad can exploit this:

```python
# Detect unified memory and skip explicit transfers
if dgx_spark_unified_memory():
    tensor = Tensor.from_blob(cpu_ptr, shape, device='CUDA')
else:
    tensor = Tensor(data).to('CUDA')
```

**Rationale**: tinygrad already supports CUDA. Key optimization is unified memory awareness.

### Decision 3: Model Loading Strategy

Support multiple precision modes for the supercombo model:

| Mode | Precision | Memory | Performance | Use Case |
|------|-----------|--------|-------------|----------|
| Full | FP32 | ~4GB | Baseline | Debugging |
| Half | FP16 | ~2GB | 1.5-2x | Development |
| Quarter | FP4/NVFP4 | ~500MB | 4-8x | Inference testing |

**Rationale**: DGX Spark's GB10 excels at FP4 (1 PFLOP). Development typically uses FP16. FP32 for numerical debugging.

### Decision 4: DoRA Integration for Fine-Tuning

Add optional DoRA (Weight-Decomposed Low-Rank Adaptation) support:

```
tools/dgx/
├── finetune.py           # Fine-tuning entry point
├── dora_adapter.py       # DoRA implementation
├── dataset_loader.py     # Route log to training data
└── checkpoint.py         # Adapter checkpoint management
```

**DoRA Architecture**:
```python
class DoRALayer:
    def __init__(self, base_weight, rank=16):
        self.magnitude = nn.Parameter(base_weight.norm(dim=1))
        self.direction = base_weight / self.magnitude.unsqueeze(1)
        self.lora_A = nn.Parameter(torch.randn(rank, base_weight.shape[1]))
        self.lora_B = nn.Parameter(torch.zeros(base_weight.shape[0], rank))

    def forward(self, x):
        adapted_direction = self.direction + (self.lora_B @ self.lora_A)
        adapted_direction = adapted_direction / adapted_direction.norm(dim=1, keepdim=True)
        return x @ (self.magnitude.unsqueeze(1) * adapted_direction).T
```

**Rationale**: DoRA achieves near full-finetuning performance with <0.1% parameter overhead. Enables community model adaptations.

### Decision 5: Simulation Acceleration

Integrate with algorithm test harness for GPU-accelerated simulation:

```python
# In algorithm_harness/runner.py
class ScenarioRunner:
    def __init__(self, use_gpu=None):
        if use_gpu is None:
            use_gpu = HARDWARE.is_dgx_spark()
        self.device = 'cuda' if use_gpu else 'cpu'

    def run_batch(self, scenarios, algorithm):
        # Batch process multiple scenarios in parallel on GPU
        states = torch.stack([s.to_tensor() for s in scenarios]).to(self.device)
        outputs = algorithm.batch_update(states)
        return outputs
```

**Rationale**: DGX Spark's GPU enables batch processing of scenarios for faster A/B testing.

### Decision 6: ConnectX-7 Multi-Node Support (Future)

The DGX Spark can link two units via ConnectX-7 (200Gbps). Design for future expansion:

```python
# Placeholder for multi-node support
class DGXSparkCluster:
    def __init__(self, nodes: list[str]):
        self.nodes = nodes  # IP addresses of linked DGX Sparks

    def distributed_inference(self, model, inputs):
        # Future: Split model or data across nodes
        raise NotImplementedError("Multi-node support planned for future release")
```

**Rationale**: 200B+ parameter models may require two linked units. Design interface now, implement later.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| tinygrad CUDA backend maturity | Test against tinygrad CI; contribute fixes upstream |
| DGX Spark availability | Support GB10 OEM systems (Acer, Dell, etc.) |
| Unified memory complexity | Clear documentation; fallback to explicit transfers |
| Training workflow divergence | Align with comma.ai's training infrastructure when possible |

## Migration Plan

1. **Phase 1**: Hardware detection + basic inference
2. **Phase 2**: Unified memory optimization + precision modes
3. **Phase 3**: DoRA fine-tuning workflow
4. **Phase 4**: Simulation acceleration integration
5. **Rollback**: Feature-flagged; disable via `OPENPILOT_DISABLE_DGX=1`

## Open Questions

1. Should we support other GB10 OEM systems (Dell, Lenovo, etc.) from day one?
2. What's the right granularity for DoRA adapters (per-car, per-condition, per-user)?
3. Should fine-tuned adapters be shareable via comma connect?
