"""DoRA: Weight-Decomposed Low-Rank Adaptation.

DoRA decomposes pretrained weights into magnitude and direction components,
then applies low-rank updates only to the direction. This achieves better
fine-tuning results than LoRA with similar parameter efficiency.

Reference: https://arxiv.org/abs/2402.09353
"""

from __future__ import annotations

import math

import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
import torch.nn.functional as F  # type: ignore[import-not-found]


class DoRALinear(nn.Module):
  """DoRA-adapted Linear layer.

  Decomposes W = m * (W_0 + BA) / ||W_0 + BA||
  where:
    - m: learnable magnitude vector (out_features,)
    - W_0: frozen pretrained weights
    - B, A: low-rank adaptation matrices
  """

  def __init__(
    self,
    in_features: int,
    out_features: int,
    rank: int = 16,
    alpha: float = 1.0,
    dropout: float = 0.0,
    bias: bool = True,
  ):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.rank = rank
    self.alpha = alpha
    self.scaling = alpha / rank

    # Frozen base weight (initialized later from pretrained)
    self.register_buffer("base_weight", torch.zeros(out_features, in_features))

    # Learnable magnitude (initialized from pretrained weight norms)
    self.magnitude = nn.Parameter(torch.ones(out_features))

    # Low-rank adaptation matrices
    self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
    self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    # Optional bias
    if bias:
      self.bias = nn.Parameter(torch.zeros(out_features))
    else:
      self.register_parameter("bias", None)

    # Dropout for regularization
    self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    self._init_lora_weights()

  def _init_lora_weights(self):
    """Initialize LoRA weights using Kaiming uniform for A, zeros for B."""
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    nn.init.zeros_(self.lora_B)

  @classmethod
  def from_linear(
    cls,
    linear: nn.Linear,
    rank: int = 16,
    alpha: float = 1.0,
    dropout: float = 0.0,
  ) -> DoRALinear:
    """Create DoRALinear from an existing nn.Linear layer."""
    dora = cls(
      in_features=linear.in_features,
      out_features=linear.out_features,
      rank=rank,
      alpha=alpha,
      dropout=dropout,
      bias=linear.bias is not None,
    )

    # Copy pretrained weights
    dora.base_weight.copy_(linear.weight.data)

    # Initialize magnitude from pretrained weight norms
    with torch.no_grad():
      dora.magnitude.copy_(linear.weight.data.norm(dim=1))

    # Copy bias if present
    if linear.bias is not None:
      dora.bias.data.copy_(linear.bias.data)

    return dora

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Compute low-rank update: delta_W = B @ A * scaling
    delta_W = (self.lora_B @ self.lora_A) * self.scaling

    # Updated weight = base + delta
    updated_weight = self.base_weight + delta_W

    # Normalize to get direction
    weight_norm = updated_weight.norm(dim=1, keepdim=True)
    direction = updated_weight / (weight_norm + 1e-8)

    # Apply magnitude scaling
    weight = self.magnitude.unsqueeze(1) * direction

    # Apply dropout and linear transformation
    x = self.dropout(x)
    return F.linear(x, weight, self.bias)

  def merge_weights(self) -> nn.Linear:
    """Merge DoRA weights into a standard Linear layer for inference."""
    delta_W = (self.lora_B @ self.lora_A) * self.scaling
    updated_weight = self.base_weight + delta_W
    weight_norm = updated_weight.norm(dim=1, keepdim=True)
    direction = updated_weight / (weight_norm + 1e-8)
    merged_weight = self.magnitude.unsqueeze(1) * direction

    linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
    linear.weight.data.copy_(merged_weight)
    if self.bias is not None:
      linear.bias.data.copy_(self.bias.data)

    return linear

  def extra_repr(self) -> str:
    return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, alpha={self.alpha}"


class DoRAConv2d(nn.Module):
  """DoRA-adapted Conv2d layer.

  Similar to DoRALinear but for convolutional layers.
  Treats (out_channels, in_channels*k*k) as the weight matrix.
  """

  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    rank: int = 16,
    alpha: float = 1.0,
    dropout: float = 0.0,
    bias: bool = True,
  ):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    self.stride = stride if isinstance(stride, tuple) else (stride, stride)
    self.padding = padding if isinstance(padding, tuple) else (padding, padding)
    self.rank = rank
    self.alpha = alpha
    self.scaling = alpha / rank

    # Flattened weight dimensions
    self.weight_dim = in_channels * self.kernel_size[0] * self.kernel_size[1]

    # Frozen base weight
    self.register_buffer("base_weight", torch.zeros(out_channels, in_channels, *self.kernel_size))

    # Learnable magnitude
    self.magnitude = nn.Parameter(torch.ones(out_channels))

    # Low-rank matrices (operate on flattened weights)
    self.lora_A = nn.Parameter(torch.zeros(rank, self.weight_dim))
    self.lora_B = nn.Parameter(torch.zeros(out_channels, rank))

    if bias:
      self.bias = nn.Parameter(torch.zeros(out_channels))
    else:
      self.register_parameter("bias", None)

    self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
    self._init_lora_weights()

  def _init_lora_weights(self):
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    nn.init.zeros_(self.lora_B)

  @classmethod
  def from_conv2d(
    cls,
    conv: nn.Conv2d,
    rank: int = 16,
    alpha: float = 1.0,
    dropout: float = 0.0,
  ) -> DoRAConv2d:
    """Create DoRAConv2d from an existing nn.Conv2d layer."""
    dora = cls(
      in_channels=conv.in_channels,
      out_channels=conv.out_channels,
      kernel_size=conv.kernel_size,
      stride=conv.stride,
      padding=conv.padding,
      rank=rank,
      alpha=alpha,
      dropout=dropout,
      bias=conv.bias is not None,
    )

    dora.base_weight.copy_(conv.weight.data)

    with torch.no_grad():
      flat_weight = conv.weight.data.view(conv.out_channels, -1)
      dora.magnitude.copy_(flat_weight.norm(dim=1))

    if conv.bias is not None:
      dora.bias.data.copy_(conv.bias.data)

    return dora

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Compute low-rank update
    delta_W = (self.lora_B @ self.lora_A) * self.scaling
    delta_W = delta_W.view(self.out_channels, self.in_channels, *self.kernel_size)

    # Updated weight
    updated_weight = self.base_weight + delta_W

    # Normalize and apply magnitude
    flat_weight = updated_weight.view(self.out_channels, -1)
    weight_norm = flat_weight.norm(dim=1, keepdim=True)
    direction = flat_weight / (weight_norm + 1e-8)
    merged_flat = self.magnitude.unsqueeze(1) * direction
    weight = merged_flat.view(self.out_channels, self.in_channels, *self.kernel_size)

    x = self.dropout(x)
    return F.conv2d(x, weight, self.bias, self.stride, self.padding)


def apply_dora_to_model(
  model: nn.Module,
  target_modules: list[str] | None = None,
  rank: int = 16,
  alpha: float = 1.0,
  dropout: float = 0.0,
) -> nn.Module:
  """Apply DoRA adaptation to specified modules in a model.

  Args:
    model: The model to adapt
    target_modules: List of module name patterns to adapt (e.g., ["fc", "proj"])
                   If None, adapts all Linear and Conv2d layers
    rank: LoRA rank
    alpha: LoRA alpha scaling
    dropout: Dropout probability

  Returns:
    Model with DoRA layers replacing specified modules
  """
  replaced = 0

  for name, module in list(model.named_modules()):
    # Check if this module should be adapted
    if target_modules is not None:
      if not any(target in name for target in target_modules):
        continue

    # Get parent module and attribute name
    parts = name.rsplit(".", 1)
    if len(parts) == 2:
      parent_name, attr_name = parts
      parent = model.get_submodule(parent_name)
    else:
      parent = model
      attr_name = name

    # Replace with DoRA version
    if isinstance(module, nn.Linear):
      dora_layer = DoRALinear.from_linear(module, rank=rank, alpha=alpha, dropout=dropout)
      setattr(parent, attr_name, dora_layer)
      replaced += 1
    elif isinstance(module, nn.Conv2d):
      dora_layer = DoRAConv2d.from_conv2d(module, rank=rank, alpha=alpha, dropout=dropout)
      setattr(parent, attr_name, dora_layer)
      replaced += 1

  print(f"Applied DoRA to {replaced} layers (rank={rank}, alpha={alpha})")
  return model


def get_dora_parameters(model: nn.Module) -> list[nn.Parameter]:
  """Get only the trainable DoRA parameters (magnitude + LoRA weights)."""
  params = []
  for module in model.modules():
    if isinstance(module, (DoRALinear, DoRAConv2d)):
      params.extend([module.magnitude, module.lora_A, module.lora_B])
      if module.bias is not None:
        params.append(module.bias)
  return params


def count_parameters(model: nn.Module) -> dict[str, int]:
  """Count total and trainable parameters."""
  total = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  dora_params = sum(p.numel() for p in get_dora_parameters(model))

  return {
    "total": total,
    "trainable": trainable,
    "dora": dora_params,
    "frozen": total - trainable,
    "dora_percent": 100 * dora_params / total if total > 0 else 0,
  }
