# Change: Integrate Meta/FAIR Research Algorithms

## Why

Meta/Facebook AI Research (FAIR) has released state-of-the-art perception models that could significantly improve openpilot's capabilities:

1. **DINOv2/v3** - Self-supervised vision features with excellent depth estimation and segmentation
2. **SAM 2/3** - Real-time video segmentation with object tracking through occlusions
3. **CoTracker** - Dense point tracking for lane markings and road features
4. **DETR** - End-to-end transformer detection without hand-crafted components
5. **UnSAMFlow** - Unsupervised optical flow for motion estimation

**Target Hardware**:
- **Development**: DGX Spark (full models, training, experimentation)
- **Deployment**: comma 4 (distilled/quantized models)

The strategy is to use FAIR models as teachers on DGX Spark to train smaller, efficient student models for comma 4.

## What Changes

- **Add FAIR Model Integration** (`tools/fair/`):
  - Model loading and inference wrappers
  - Feature extraction pipelines
  - Benchmark utilities

- **Add Knowledge Distillation** (`tools/fair/distillation/`):
  - Teacher-student training framework
  - Feature distillation losses
  - Quantization-aware training

- **Add Perception Enhancements** (`selfdrive/modeld/fair/`):
  - DINOv2 depth estimation head
  - SAM-guided segmentation
  - CoTracker lane tracking
  - Distilled models for comma 4

- **Add Training Pipelines** (`tools/fair/training/`):
  - DoRA fine-tuning with FAIR teachers
  - Multi-task learning (depth + segmentation + detection)
  - Dataset preparation from route logs

## Impact

- Affected specs: New `fair-integration` capability
- Affected code:
  - `tools/fair/` - New FAIR integration
  - `selfdrive/modeld/fair/` - Perception enhancements
- Dependencies: torch, torchvision, segment-anything, dinov2 (new)
- Hardware requirements:
  - DGX Spark for training/teacher inference
  - comma 4 for distilled model deployment
- **No changes to safety-critical code** until validated
