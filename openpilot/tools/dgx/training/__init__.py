"""DoRA fine-tuning training module for openpilot models.

Requires PyTorch for training functionality. Imports are conditional
to allow the module to be imported even when torch is not installed.
"""

__all__: list[str] = []

# Conditionally import torch-dependent modules
try:
  from openpilot.tools.dgx.training.dora import (  # noqa: F401
    DoRAConv2d,
    DoRALinear,
    apply_dora_to_model,
    count_parameters,
    get_dora_parameters,
  )
  from openpilot.tools.dgx.training.losses import (  # noqa: F401
    CombinedTrainingLoss,
    FeatureDistillationLoss,
    GaussianNLLLoss,
    LaplacianNLLLoss,
    PathDistillationLoss,
  )

  __all__.extend(
    [
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
  )
except ImportError:
  # torch not installed - training features unavailable
  pass
