"""FLOPs and parameter counting utilities.

Provides tools for estimating model computational cost.
"""

from __future__ import annotations

from dataclasses import dataclass

# Check PyTorch availability
try:
  import torch
  import torch.nn as nn

  TORCH_AVAILABLE = True
except ImportError:
  TORCH_AVAILABLE = False


@dataclass
class ModelStats:
  """Model statistics.

  Attributes:
    total_params: Total number of parameters
    trainable_params: Number of trainable parameters
    total_flops: Estimated total FLOPs
    param_bytes: Total parameter memory in bytes
    input_shape: Input shape used for FLOPs calculation
  """

  total_params: int
  trainable_params: int
  total_flops: int
  param_bytes: int
  input_shape: tuple[int, ...] | None = None

  @property
  def total_params_m(self) -> float:
    """Total parameters in millions."""
    return self.total_params / 1e6

  @property
  def total_flops_g(self) -> float:
    """Total FLOPs in GFLOPs."""
    return self.total_flops / 1e9

  @property
  def param_mb(self) -> float:
    """Parameter memory in MB."""
    return self.param_bytes / 1024 / 1024

  def __str__(self) -> str:
    """Format stats as string."""
    return (
      f"Parameters: {self.total_params_m:.2f}M ({self.trainable_params / 1e6:.2f}M trainable)\n"
      + f"FLOPs: {self.total_flops_g:.2f}G\n"
      + f"Memory: {self.param_mb:.2f}MB"
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
  """Count model parameters.

  Args:
    model: PyTorch model

  Returns:
    Tuple of (total_params, trainable_params)
  """
  if not TORCH_AVAILABLE:
    raise ImportError("PyTorch required")

  total = 0
  trainable = 0

  for param in model.parameters():
    num = param.numel()
    total += num
    if param.requires_grad:
      trainable += num

  return total, trainable


def _conv_flops(module: nn.Module, input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> int:
  """Calculate FLOPs for convolution layer."""
  # FLOPs = 2 * Cout * Hout * Wout * Cin * K * K
  batch, c_out, h_out, w_out = output_shape
  c_in = input_shape[1]

  k_h = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
  k_w = module.kernel_size[1] if isinstance(module.kernel_size, tuple) else module.kernel_size

  flops = 2 * c_out * h_out * w_out * c_in * k_h * k_w // module.groups

  if module.bias is not None:
    flops += c_out * h_out * w_out

  return flops


def _linear_flops(module: nn.Module, input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> int:
  """Calculate FLOPs for linear layer."""
  # FLOPs = 2 * in_features * out_features
  batch_size = input_shape[0] if len(input_shape) > 1 else 1
  flops = 2 * module.in_features * module.out_features * batch_size

  if module.bias is not None:
    flops += module.out_features * batch_size

  return flops


def _attention_flops(
  embed_dim: int,
  num_heads: int,
  seq_len: int,
  batch_size: int = 1,
) -> int:
  """Estimate FLOPs for multi-head attention.

  Args:
    embed_dim: Embedding dimension
    num_heads: Number of attention heads
    seq_len: Sequence length
    batch_size: Batch size

  Returns:
    Estimated FLOPs
  """
  # Q, K, V projections
  qkv_flops = 3 * 2 * seq_len * embed_dim * embed_dim * batch_size

  # Attention scores: Q @ K^T
  attn_flops = 2 * seq_len * seq_len * embed_dim * batch_size

  # Attention @ V
  context_flops = 2 * seq_len * seq_len * embed_dim * batch_size

  # Output projection
  out_flops = 2 * seq_len * embed_dim * embed_dim * batch_size

  return qkv_flops + attn_flops + context_flops + out_flops


def estimate_flops(
  model: nn.Module,
  input_shape: tuple[int, ...],
  detailed: bool = False,
) -> int | dict[str, int]:
  """Estimate model FLOPs.

  This is a simplified estimation that handles common layers.
  For more accurate results, use specialized profilers.

  Args:
    model: PyTorch model
    input_shape: Input tensor shape (including batch)
    detailed: Return per-layer breakdown

  Returns:
    Total FLOPs or dictionary with per-layer FLOPs
  """
  if not TORCH_AVAILABLE:
    raise ImportError("PyTorch required")

  total_flops = 0
  layer_flops = {}

  # Hook to capture shapes
  handles = []
  input_shapes = {}
  output_shapes = {}

  def hook_fn(name):
    def hook(module, inp, out):
      if len(inp) > 0 and hasattr(inp[0], "shape"):
        input_shapes[name] = tuple(inp[0].shape)
      if hasattr(out, "shape"):
        output_shapes[name] = tuple(out.shape)

    return hook

  # Register hooks
  for name, module in model.named_modules():
    handles.append(module.register_forward_hook(hook_fn(name)))

  # Forward pass
  dummy_input = torch.randn(input_shape)
  model.eval()
  with torch.no_grad():
    model(dummy_input)

  # Remove hooks
  for handle in handles:
    handle.remove()

  # Calculate FLOPs
  for name, module in model.named_modules():
    if name not in input_shapes:
      continue

    inp_shape = input_shapes[name]
    out_shape = output_shapes.get(name, inp_shape)
    flops = 0

    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
      flops = _conv_flops(module, inp_shape, out_shape)
    elif isinstance(module, nn.Linear):
      flops = _linear_flops(module, inp_shape, out_shape)
    elif isinstance(module, nn.MultiheadAttention):
      seq_len = inp_shape[0] if len(inp_shape) >= 2 else 1
      batch_size = inp_shape[1] if len(inp_shape) >= 2 else 1
      flops = _attention_flops(module.embed_dim, module.num_heads, seq_len, batch_size)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
      # Normalization: ~4 ops per element
      num_elements = 1
      for dim in inp_shape:
        num_elements *= dim
      flops = 4 * num_elements

    if flops > 0:
      layer_flops[name] = flops
      total_flops += flops

  if detailed:
    return layer_flops

  return total_flops


def get_model_stats(
  model: nn.Module,
  input_shape: tuple[int, ...] | None = None,
) -> ModelStats:
  """Get complete model statistics.

  Args:
    model: PyTorch model
    input_shape: Input shape for FLOPs calculation

  Returns:
    ModelStats with parameters and FLOPs
  """
  if not TORCH_AVAILABLE:
    raise ImportError("PyTorch required")

  total_params, trainable_params = count_parameters(model)

  # Calculate parameter memory
  param_bytes = 0
  for param in model.parameters():
    param_bytes += param.numel() * param.element_size()

  # Calculate FLOPs if input shape provided
  total_flops = 0
  if input_shape is not None:
    total_flops = estimate_flops(model, input_shape)

  return ModelStats(
    total_params=total_params,
    trainable_params=trainable_params,
    total_flops=total_flops,
    param_bytes=param_bytes,
    input_shape=input_shape,
  )


def format_layer_flops(layer_flops: dict[str, int], top_k: int = 20) -> str:
  """Format layer FLOPs as table.

  Args:
    layer_flops: Dictionary mapping layer name to FLOPs
    top_k: Number of top layers to show

  Returns:
    Formatted table string
  """
  # Sort by FLOPs
  sorted_layers = sorted(layer_flops.items(), key=lambda x: x[1], reverse=True)

  total = sum(layer_flops.values())
  lines = []
  lines.append(f"{'Layer':<50} {'FLOPs':<15} {'%':<10}")
  lines.append("-" * 75)

  for name, flops in sorted_layers[:top_k]:
    pct = 100 * flops / total if total > 0 else 0
    flops_str = f"{flops / 1e6:.2f}M" if flops >= 1e6 else f"{flops / 1e3:.2f}K"
    lines.append(f"{name:<50} {flops_str:<15} {pct:.1f}%")

  if len(sorted_layers) > top_k:
    lines.append(f"... and {len(sorted_layers) - top_k} more layers")

  lines.append("-" * 75)
  lines.append(f"{'Total':<50} {total / 1e9:.2f}G FLOPs")

  return "\n".join(lines)
