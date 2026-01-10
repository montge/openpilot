## Context

openpilot's perception uses a monolithic "supercombo" CNN trained end-to-end. While effective, it may benefit from:

1. **Richer features** - DINOv2's self-supervised features are more robust across conditions
2. **Explicit depth** - Current implicit depth; DINOv2 depth head provides explicit estimates
3. **Object permanence** - SAM 2/3's memory enables tracking through occlusions
4. **Lane tracking** - CoTracker can track lane markings even when partially visible
5. **Better detection** - DETR's transformer architecture handles multi-scale better

**Target Hardware**:
- **DGX Spark**: GB10 Grace Blackwell, 128GB unified memory, 1 PFLOP FP4
- **comma 4**: Next-generation comma device (assumed improved from comma 3X)

**Strategy**: Train on DGX Spark, deploy distilled models on comma 4.

## Goals / Non-Goals

**Goals**:
- Integrate FAIR models for development and experimentation on DGX Spark
- Create knowledge distillation pipeline for comma 4 deployment
- Improve perception in challenging conditions (weather, lighting, occlusions)
- Benchmark FAIR approaches vs current openpilot perception

**Non-Goals**:
- Run full FAIR models on comma 4 (too computationally expensive)
- Replace entire perception pipeline immediately (incremental integration)
- Real-time FAIR model inference on comma device

## Decisions

### Decision 1: Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DGX Spark (Development)                   │
├─────────────────────────────────────────────────────────────┤
│  Full FAIR Models  │  Training  │  Teacher Inference        │
│  (DINOv2, SAM 2/3) │  Pipeline  │  (Generate Soft Labels)   │
├─────────────────────────────────────────────────────────────┤
│                    Knowledge Distillation                    │
├─────────────────────────────────────────────────────────────┤
│                     comma 4 (Deployment)                     │
├─────────────────────────────────────────────────────────────┤
│  Distilled Student Models  │  Quantized  │  Real-time       │
│  (Small ViT, Efficient)    │  INT8/FP16  │  Inference       │
└─────────────────────────────────────────────────────────────┘
```

### Decision 2: DINOv2 Integration

**Full model on DGX Spark**:
```python
# tools/fair/dinov2/inference.py
from dinov2.models import build_model

class DINOv2Wrapper:
    def __init__(self, model_size='vit_base'):
        self.model = build_model(model_size)  # ViT-B/14
        self.depth_head = DepthHead()  # Linear probe for depth

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract DINOv2 features (batch, 768, H/14, W/14)."""
        return self.model.get_intermediate_layers(images)

    def estimate_depth(self, images: torch.Tensor) -> torch.Tensor:
        """Monocular depth estimation."""
        features = self.extract_features(images)
        return self.depth_head(features)
```

**Distilled model for comma 4**:
```python
# selfdrive/modeld/fair/dinov2_student.py
class DINOv2Student(nn.Module):
    """Distilled DINOv2 for comma 4 deployment."""

    def __init__(self):
        self.backbone = EfficientViT_B0()  # ~5M params vs 86M
        self.depth_head = nn.Conv2d(256, 1, 1)

    def forward(self, x):
        features = self.backbone(x)
        depth = self.depth_head(features)
        return {'features': features, 'depth': depth}
```

### Decision 3: SAM 2/3 Video Segmentation

**Teacher on DGX Spark**:
```python
# tools/fair/sam/inference.py
from sam2.build_sam import build_sam2_video_predictor

class SAM2VideoWrapper:
    def __init__(self):
        self.predictor = build_sam2_video_predictor('sam2_hiera_large')

    def track_objects(self, video_frames, initial_boxes):
        """Track objects through video with SAM2's memory."""
        with torch.inference_mode():
            state = self.predictor.init_state(video_frames[0])
            for box in initial_boxes:
                self.predictor.add_new_points_or_box(state, box=box)

            masks = []
            for frame in video_frames:
                out = self.predictor.propagate_in_video(state)
                masks.append(out['masks'])
        return masks
```

**Student for comma 4**:
- Lightweight segmentation head on distilled backbone
- Track objects via simple mask propagation + re-detection

### Decision 4: CoTracker Lane Tracking

**Full model on DGX Spark**:
```python
# tools/fair/cotracker/inference.py
from cotracker.predictor import CoTrackerPredictor

class LaneTracker:
    def __init__(self):
        self.cotracker = CoTrackerPredictor('cotracker2')

    def track_lane_points(self, video: torch.Tensor, lane_points: torch.Tensor):
        """Track lane marking points through video."""
        # lane_points: (N, 2) initial point coordinates
        pred_tracks, pred_visibility = self.cotracker(
            video,
            queries=lane_points,
            backward_tracking=False
        )
        return pred_tracks, pred_visibility  # (T, N, 2), (T, N)
```

**Use cases**:
- Track lane markings through shadows, glare, partial occlusion
- Improve lane-keeping in construction zones with faded/missing markings
- Generate pseudo-labels for lane detection training

### Decision 5: Knowledge Distillation Pipeline

```python
# tools/fair/distillation/trainer.py
class DistillationTrainer:
    def __init__(self, teacher, student, temperature=4.0):
        self.teacher = teacher.eval()
        self.student = student
        self.temperature = temperature

    def distillation_loss(self, student_out, teacher_out, targets):
        """Combined distillation + task loss."""
        # Soft label distillation
        soft_loss = F.kl_div(
            F.log_softmax(student_out / self.temperature, dim=-1),
            F.softmax(teacher_out / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Hard label task loss
        hard_loss = F.cross_entropy(student_out, targets)

        return 0.7 * soft_loss + 0.3 * hard_loss

    def feature_distillation_loss(self, student_features, teacher_features):
        """Match intermediate feature representations."""
        # Project student features to teacher dimension
        projected = self.projector(student_features)
        return F.mse_loss(projected, teacher_features)
```

### Decision 6: DoRA + FAIR Teachers

Combine DoRA fine-tuning with FAIR model distillation:

```python
# tools/fair/training/dora_distillation.py
class DoRADistillation:
    """Fine-tune openpilot model with DoRA using FAIR teachers."""

    def __init__(self, base_model, fair_teacher):
        self.student = DoRAWrapper(base_model)  # DoRA-enabled student
        self.teacher = fair_teacher  # DINOv2 or SAM

    def train_step(self, images, targets):
        with torch.no_grad():
            teacher_features = self.teacher.extract_features(images)
            teacher_depth = self.teacher.estimate_depth(images)

        student_out = self.student(images)

        loss = (
            self.task_loss(student_out, targets) +
            self.feature_loss(student_out['features'], teacher_features) +
            self.depth_loss(student_out['depth'], teacher_depth)
        )
        return loss
```

### Decision 7: Comma 4 Deployment Targets

| Model | DGX Spark Size | Comma 4 Target | Compression |
|-------|---------------|----------------|-------------|
| DINOv2-B | 86M params | 5M params | 17x |
| SAM 2 | 312M params | 10M params | 31x |
| CoTracker | 24M params | 3M params | 8x |
| DETR | 41M params | 8M params | 5x |

**Techniques**:
- Knowledge distillation (soft labels from teacher)
- Feature distillation (match intermediate representations)
- Quantization-aware training (INT8 for comma 4)
- Pruning (remove redundant weights)

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| FAIR model licenses (Apache 2.0, research-only) | Verify license compatibility; distilled models are new works |
| Distillation quality loss | Extensive benchmarking; multi-stage distillation |
| comma 4 unknown specs | Design for flexibility; prepare multiple model sizes |
| Training compute cost | Use DGX Spark efficiently; incremental training |

## Migration Plan

1. **Phase 1**: DINOv2 integration on DGX Spark + depth benchmarks
2. **Phase 2**: SAM 2 video segmentation + object tracking
3. **Phase 3**: CoTracker lane tracking experiments
4. **Phase 4**: Distillation pipeline + comma 4 student models
5. **Phase 5**: Integration with openpilot perception (behind feature flag)

## Open Questions

1. What are comma 4's compute constraints (Qualcomm 8 Gen 3? Custom silicon?)
2. Can distilled models match FAIR model accuracy within 5%?
3. Should we distill into existing supercombo or create parallel heads?
4. How to handle FAIR model updates (re-distill periodically?)
