"""Training pipelines for FAIR models.

Provides utilities for fine-tuning FAIR teacher models and
training efficient student models for deployment.
"""

from openpilot.tools.fair.training.dora import (
  DoRAConfig,
  DoRALayer,
  apply_dora,
)
from openpilot.tools.fair.training.multitask import (
  MultiTaskConfig,
  MultiTaskHead,
  MultiTaskTrainer,
)
from openpilot.tools.fair.training.dataset import (
  RouteDataset,
  FrameData,
)

__all__ = [
  "DoRAConfig",
  "DoRALayer",
  "apply_dora",
  "MultiTaskConfig",
  "MultiTaskHead",
  "MultiTaskTrainer",
  "RouteDataset",
  "FrameData",
]
