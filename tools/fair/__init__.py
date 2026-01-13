"""FAIR model integration and knowledge distillation tools.

This module provides wrappers for Meta FAIR research models and
knowledge distillation utilities for training efficient student models.

Components:
- models/: Model wrappers (DINOv2, SAM2, CoTracker, DETR)
- distillation/: Knowledge distillation training framework
"""

from openpilot.tools.fair.models import (
  DINOV2_AVAILABLE,
  COTRACKER_AVAILABLE,
  DETR_AVAILABLE,
  SAM2_AVAILABLE,
)

__all__ = [
  "DINOV2_AVAILABLE",
  "COTRACKER_AVAILABLE",
  "DETR_AVAILABLE",
  "SAM2_AVAILABLE",
]
