"""FAIR model integration and knowledge distillation tools.

This module provides wrappers for Meta FAIR research models and
knowledge distillation utilities for training efficient student models.

Components:
- models/: Model wrappers (DINOv2, SAM2, CoTracker, DETR, UnSAMFlow)
- distillation/: Knowledge distillation training framework
- students/: Lightweight student model architectures
- training/: DoRA adapters, multi-task learning, datasets
"""

from openpilot.tools.fair.models import (
  DINOV2_AVAILABLE,
  COTRACKER_AVAILABLE,
  DETR_AVAILABLE,
  SAM2_AVAILABLE,
  UNSAMFLOW_AVAILABLE,
)

__all__ = [
  "DINOV2_AVAILABLE",
  "COTRACKER_AVAILABLE",
  "DETR_AVAILABLE",
  "SAM2_AVAILABLE",
  "UNSAMFLOW_AVAILABLE",
]
