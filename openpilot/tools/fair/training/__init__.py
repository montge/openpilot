"""Training pipelines for FAIR models.

Provides utilities for fine-tuning FAIR teacher models and
training efficient student models for deployment.
"""

from openpilot.tools.fair.training.dora import (
  DoRAConfig,
  DoRALayer,
  apply_dora,
  save_dora_checkpoint,
  load_dora_checkpoint,
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
from openpilot.tools.fair.training.hard_mining import (
  HardMiningConfig,
  DifficultyTracker,
  HardExampleSampler,
)

__all__ = [
  "DoRAConfig",
  "DoRALayer",
  "apply_dora",
  "save_dora_checkpoint",
  "load_dora_checkpoint",
  "MultiTaskConfig",
  "MultiTaskHead",
  "MultiTaskTrainer",
  "RouteDataset",
  "FrameData",
  "HardMiningConfig",
  "DifficultyTracker",
  "HardExampleSampler",
]
