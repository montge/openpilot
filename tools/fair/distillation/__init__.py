"""Knowledge distillation framework.

Provides utilities for distilling knowledge from FAIR teacher models
to efficient student models suitable for real-time inference.
"""

from openpilot.tools.fair.distillation.losses import (
  DistillationLoss,
  FeatureDistillationLoss,
  AttentionDistillationLoss,
  ResponseDistillationLoss,
  TaskDistillationLoss,
)
from openpilot.tools.fair.distillation.trainer import (
  DistillationConfig,
  DistillationTrainer,
)

__all__ = [
  "DistillationLoss",
  "FeatureDistillationLoss",
  "AttentionDistillationLoss",
  "ResponseDistillationLoss",
  "TaskDistillationLoss",
  "DistillationConfig",
  "DistillationTrainer",
]
