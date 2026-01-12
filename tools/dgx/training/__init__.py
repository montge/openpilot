"""DoRA fine-tuning training module for openpilot models."""

from openpilot.tools.dgx.training.dora import (
  DoRAConv2d,
  DoRALinear,
  apply_dora_to_model,
  count_parameters,
  get_dora_parameters,
)
from openpilot.tools.dgx.training.losses import (
  CombinedTrainingLoss,
  FeatureDistillationLoss,
  GaussianNLLLoss,
  LaplacianNLLLoss,
  PathDistillationLoss,
)

__all__ = [
  # DoRA
  "DoRALinear",
  "DoRAConv2d",
  "apply_dora_to_model",
  "get_dora_parameters",
  "count_parameters",
  # Losses
  "LaplacianNLLLoss",
  "GaussianNLLLoss",
  "PathDistillationLoss",
  "FeatureDistillationLoss",
  "CombinedTrainingLoss",
]
