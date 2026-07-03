"""DoRA (Weight-Decomposed Low-Rank Adaptation) for FAIR models.

DoRA decomposes model weights into magnitude and direction components,
applying low-rank adaptation only to the direction. This improves
training stability and final performance over standard LoRA.

Reference: https://arxiv.org/abs/2402.09353
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

# Check PyTorch availability
try:
  import torch
  import torch.nn as nn

  TORCH_AVAILABLE = True
  _BaseModule = nn.Module
except ImportError:
  TORCH_AVAILABLE = False
  _BaseModule = object


@dataclass
class DoRAConfig:
  """DoRA configuration.

  Attributes:
    rank: Low-rank dimension for adaptation
    alpha: Scaling factor (default: rank)
    dropout: Dropout probability for adaptation
    target_modules: Module names to apply DoRA (e.g., ['q_proj', 'v_proj'])
    use_rslora: Use rank-stabilized LoRA scaling
  """

  rank: int = 8
  alpha: float | None = None
  dropout: float = 0.0
  target_modules: list[str] | None = None
  use_rslora: bool = True

  def __post_init__(self) -> None:
    """Set alpha to rank if not specified."""
    if self.alpha is None:
      self.alpha = float(self.rank)


class DoRALayer(_BaseModule):
  """DoRA layer that wraps a linear layer.

  Decomposes the weight update into magnitude and direction:
  W' = m * (W + BA) / ||W + BA||

  where m is the learned magnitude, B and A are low-rank matrices.
  """

  def __init__(
    self,
    base_layer: nn.Linear,
    config: DoRAConfig,
  ):
    """Initialize DoRA layer.

    Args:
      base_layer: Original linear layer to adapt
      config: DoRA configuration
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required for DoRA")

    super().__init__()

    self.base_layer = base_layer
    self.config = config

    in_features = base_layer.in_features
    out_features = base_layer.out_features

    # Freeze base layer
    base_layer.weight.requires_grad = False
    if base_layer.bias is not None:
      base_layer.bias.requires_grad = False

    # Low-rank adaptation matrices
    self.lora_A = nn.Linear(in_features, config.rank, bias=False)
    self.lora_B = nn.Linear(config.rank, out_features, bias=False)

    # Magnitude parameter (one per output neuron)
    with torch.no_grad():
      # Initialize magnitude to match original weight norms
      weight_norm = base_layer.weight.norm(dim=1, keepdim=True)
    self.magnitude = nn.Parameter(weight_norm.squeeze())

    # Dropout
    self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    # Scaling
    if config.use_rslora:
      self.scaling = config.alpha / math.sqrt(config.rank)
    else:
      self.scaling = config.alpha / config.rank

    # Initialize
    self._init_weights()

  def _init_weights(self) -> None:
    """Initialize adaptation weights."""
    # A: Kaiming uniform, B: zeros (start with no adaptation)
    nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
    nn.init.zeros_(self.lora_B.weight)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass with DoRA adaptation.

    Args:
      x: Input tensor

    Returns:
      Adapted output
    """
    # Original output
    base_out = self.base_layer(x)

    # Low-rank adaptation
    lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

    # Compute adapted weight direction
    adapted_weight = self.base_layer.weight + self.lora_B.weight @ self.lora_A.weight * self.scaling

    # Normalize direction and apply magnitude
    weight_norm = adapted_weight.norm(dim=1, keepdim=True)
    direction_scale = self.magnitude / (weight_norm.squeeze() + 1e-8)

    # Scale the outputs by direction normalization factor
    # This approximates: m * (Wx + BA x) / ||W + BA||
    combined = base_out + lora_out
    return combined * direction_scale.unsqueeze(0).unsqueeze(0)

  def merge(self) -> nn.Linear:
    """Merge adaptation into base layer.

    Returns:
      New linear layer with merged weights
    """
    with torch.no_grad():
      # Compute adapted weight
      delta = self.lora_B.weight @ self.lora_A.weight * self.scaling
      adapted_weight = self.base_layer.weight + delta

      # Normalize and apply magnitude
      weight_norm = adapted_weight.norm(dim=1, keepdim=True)
      merged_weight = self.magnitude.unsqueeze(1) * adapted_weight / (weight_norm + 1e-8)

      # Create merged layer
      merged = nn.Linear(
        self.base_layer.in_features,
        self.base_layer.out_features,
        bias=self.base_layer.bias is not None,
      )
      merged.weight.data = merged_weight
      if self.base_layer.bias is not None:
        merged.bias.data = self.base_layer.bias.data

    return merged


def apply_dora(
  model: nn.Module,
  config: DoRAConfig,
  target_modules: list[str] | None = None,
) -> nn.Module:
  """Apply DoRA to target modules in a model.

  Args:
    model: Model to adapt
    config: DoRA configuration
    target_modules: Module names to adapt (overrides config)

  Returns:
    Model with DoRA layers applied
  """
  if not TORCH_AVAILABLE:
    raise ImportError("PyTorch required for DoRA")

  targets = target_modules or config.target_modules or []

  for name, module in model.named_modules():
    if any(target in name for target in targets):
      if isinstance(module, nn.Linear):
        # Replace with DoRA layer
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]

        parent = model
        if parent_name:
          for part in parent_name.split("."):
            parent = getattr(parent, part)

        dora_layer = DoRALayer(module, config)
        setattr(parent, child_name, dora_layer)

  return model


def get_dora_parameters(model: nn.Module) -> list[nn.Parameter]:
  """Get only DoRA trainable parameters.

  Args:
    model: Model with DoRA layers

  Returns:
    List of trainable DoRA parameters
  """
  params = []
  for module in model.modules():
    if isinstance(module, DoRALayer):
      params.extend(
        [
          module.lora_A.weight,
          module.lora_B.weight,
          module.magnitude,
        ]
      )
  return params


def save_dora_checkpoint(
  model: nn.Module,
  path: str | Path,
  config: DoRAConfig | None = None,
  metadata: dict | None = None,
) -> Path:
  """Save only DoRA adapter weights (not the full model).

  This is much smaller than a full model checkpoint since only
  the low-rank adaptation parameters are saved.

  Args:
    model: Model with DoRA layers
    path: Path to save checkpoint
    config: DoRA config to save with checkpoint
    metadata: Optional metadata (epoch, loss, etc.)

  Returns:
    Path to saved checkpoint
  """
  if not TORCH_AVAILABLE:
    raise ImportError("PyTorch required for DoRA")

  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)

  adapter_state = {}
  for name, module in model.named_modules():
    if isinstance(module, DoRALayer):
      adapter_state[f"{name}.lora_A.weight"] = module.lora_A.weight.data.cpu()
      adapter_state[f"{name}.lora_B.weight"] = module.lora_B.weight.data.cpu()
      adapter_state[f"{name}.magnitude"] = module.magnitude.data.cpu()

  checkpoint = {
    "adapter_state": adapter_state,
    "config": config.__dict__ if config else None,
    "metadata": metadata or {},
  }

  torch.save(checkpoint, path)
  return path


def load_dora_checkpoint(
  model: nn.Module,
  path: str | Path,
) -> dict:
  """Load DoRA adapter weights into a model.

  The model must already have DoRA layers applied.

  Args:
    model: Model with DoRA layers
    path: Path to checkpoint

  Returns:
    Checkpoint metadata
  """
  if not TORCH_AVAILABLE:
    raise ImportError("PyTorch required for DoRA")

  checkpoint = torch.load(path, map_location="cpu", weights_only=True)
  adapter_state = checkpoint["adapter_state"]

  for name, module in model.named_modules():
    if isinstance(module, DoRALayer):
      a_key = f"{name}.lora_A.weight"
      b_key = f"{name}.lora_B.weight"
      m_key = f"{name}.magnitude"

      if a_key in adapter_state:
        module.lora_A.weight.data = adapter_state[a_key].to(module.lora_A.weight.device)
      if b_key in adapter_state:
        module.lora_B.weight.data = adapter_state[b_key].to(module.lora_B.weight.device)
      if m_key in adapter_state:
        module.magnitude.data = adapter_state[m_key].to(module.magnitude.device)

  return checkpoint.get("metadata", {})


def merge_dora(model: nn.Module) -> nn.Module:
  """Merge all DoRA layers back into linear layers.

  Args:
    model: Model with DoRA layers

  Returns:
    Model with merged weights
  """
  if not TORCH_AVAILABLE:
    raise ImportError("PyTorch required for DoRA")

  for name, module in list(model.named_modules()):
    if isinstance(module, DoRALayer):
      parent_name = ".".join(name.split(".")[:-1])
      child_name = name.split(".")[-1]

      parent = model
      if parent_name:
        for part in parent_name.split("."):
          parent = getattr(parent, part)

      merged = module.merge()
      setattr(parent, child_name, merged)

  return model
