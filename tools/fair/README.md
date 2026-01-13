# FAIR Model Integration Tools

This module provides wrappers for Meta FAIR research models and knowledge distillation utilities for training efficient student models suitable for real-time inference in openpilot.

## Overview

The FAIR tools module contains:

- **Model Wrappers**: Unified interfaces for FAIR vision models (DINOv2, SAM2, CoTracker, DETR)
- **Knowledge Distillation**: Framework for training efficient student models from large teacher models
- **Student Models**: Lightweight architectures optimized for real-time inference

## Installation

The FAIR tools have optional dependencies. Install PyTorch to use the full functionality:

```bash
pip install torch torchvision
```

For SAM2 support:
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

## Model Wrappers

### DINOv2

Self-supervised vision transformer for feature extraction and depth estimation.

```python
from openpilot.tools.fair.models import DINOv2Wrapper
from openpilot.tools.fair.models.dinov2 import DINOv2Config

# Create wrapper with custom config
config = DINOv2Config(model_name="dinov2_vitb14")
model = DINOv2Wrapper(config)

# Use context manager for automatic load/unload
with model:
    features = model.extract_features(images)  # [B, N, D]
    cls_token = model.get_cls_token(images)    # [B, D]
```

Available models: `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14`

### SAM2

Video object segmentation with memory-based tracking.

```python
from openpilot.tools.fair.models import SAM2Wrapper
from openpilot.tools.fair.models.sam2 import SAM2Config

config = SAM2Config(model_name="sam2_hiera_base_plus")
model = SAM2Wrapper(config)

with model:
    # Single image segmentation with point prompts
    result = model.segment_image(image, points=[[100, 200]], point_labels=[1])

    # Video tracking
    model.init_video_tracking(first_frame, initial_masks={1: mask})
    for frame in video_frames:
        state = model.track_video_frame(frame)
```

### CoTracker

Point tracking through video sequences.

```python
from openpilot.tools.fair.models import CoTrackerWrapper
from openpilot.tools.fair.models.cotracker import CoTrackerConfig

config = CoTrackerConfig(model_name="cotracker2", grid_size=10)
model = CoTrackerWrapper(config)

with model:
    # Track specific points [B, N, 3] where each point is [t, x, y]
    result = model.track_points(video, query_points)

    # Dense grid tracking
    result = model.track_grid(video, grid_size=10)

    # Lane point tracking
    result = model.track_lane_points(video, lane_points)
```

### DETR

End-to-end object detection with transformers.

```python
from openpilot.tools.fair.models import DETRWrapper
from openpilot.tools.fair.models.detr import DETRConfig

config = DETRConfig(model_name="detr_resnet50", confidence_threshold=0.7)
model = DETRWrapper(config)

with model:
    # Detect all objects
    result = model.detect(image)

    # Detect vehicles only
    vehicles = model.detect_vehicles(image)

    # Batch detection
    results = model.detect_batch(images)
```

## Knowledge Distillation

### Basic Distillation

```python
from openpilot.tools.fair.distillation import (
    DistillationTrainer,
    DistillationConfig,
    ResponseDistillationLoss,
)

# Setup teacher and student
teacher = DINOv2Wrapper(DINOv2Config(model_name="dinov2_vitl14"))
student = TinyViT(TinyViTConfig(embed_dim=192))

# Configure training
config = DistillationConfig(
    epochs=100,
    learning_rate=1e-4,
    batch_size=32,
)

# Train
trainer = DistillationTrainer(teacher, student, config)
trainer.train(train_loader, val_loader)
```

### Loss Functions

- **ResponseDistillationLoss**: KL divergence on softened predictions
- **FeatureDistillationLoss**: MSE/cosine loss on intermediate features
- **AttentionDistillationLoss**: Match attention patterns
- **CombinedDistillationLoss**: Weighted combination of multiple losses

```python
from openpilot.tools.fair.distillation import (
    ResponseDistillationLoss,
    FeatureDistillationLoss,
    AttentionDistillationLoss,
)
from openpilot.tools.fair.distillation.losses import (
    ResponseDistillationConfig,
    FeatureDistillationConfig,
    CombinedDistillationLoss,
)

# Response distillation with temperature
response_loss = ResponseDistillationLoss(
    ResponseDistillationConfig(temperature=4.0, alpha=0.7)
)

# Feature matching
feature_loss = FeatureDistillationLoss(
    FeatureDistillationConfig(normalize=True, loss_type="cosine")
)

# Combined loss
combined = CombinedDistillationLoss([
    (response_loss, 1.0),
    (feature_loss, 0.5),
])
```

## Student Models

### Vision Models

```python
from openpilot.tools.fair.students import TinyViT, MobileViT, EfficientStudent
from openpilot.tools.fair.students.vision import TinyViTConfig

# TinyViT - lightweight ViT
config = TinyViTConfig(embed_dim=192, depth=6, num_heads=3)
model = TinyViT(config)

# MobileViT - hybrid conv/transformer
model = MobileViT()

# EfficientStudent - simple CNN
model = EfficientStudent()
```

### Detection Models

```python
from openpilot.tools.fair.students import TinyDETR, MobileDetector
from openpilot.tools.fair.students.detection import TinyDETRConfig

# TinyDETR - lightweight DETR
config = TinyDETRConfig(
    hidden_dim=128,
    num_queries=50,
    backbone="mobilenet",
)
model = TinyDETR(config)

# MobileDetector - anchor-based detector
model = MobileDetector()
```

## Checking Availability

All model wrappers gracefully handle missing dependencies:

```python
from openpilot.tools.fair import (
    DINOV2_AVAILABLE,
    SAM2_AVAILABLE,
    COTRACKER_AVAILABLE,
    DETR_AVAILABLE,
)

if DINOV2_AVAILABLE:
    from openpilot.tools.fair.models import DINOv2Wrapper
    # Use DINOv2
```

## Use Cases for openpilot

### Lane Detection Enhancement

Use CoTracker to track lane markings through video:

```python
# Extract lane points from current frame
lane_points = extract_lane_points(frame)

# Track through video
with CoTrackerWrapper() as tracker:
    result = tracker.track_lane_points(video_buffer, lane_points)
    # Use tracked points for lane fitting
```

### Vehicle Segmentation

Use SAM2 for accurate vehicle segmentation:

```python
with SAM2Wrapper() as sam:
    # Get vehicle detections from DETR
    detections = detr.detect_vehicles(frame)

    # Segment each vehicle
    for det in detections.detections:
        cx, cy = (det.box[:2] + det.box[2:]) / 2
        mask = sam.segment_image(frame, points=[[cx, cy]], point_labels=[1])
```

### Feature Extraction for Planning

Use DINOv2 for rich visual features:

```python
with DINOv2Wrapper() as dino:
    features = dino.extract_features(frames)
    # Use features for scene understanding
```
