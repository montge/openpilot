"""Quantization-aware training and INT8 quantization.

Supports:
- Quantization-aware training (QAT) with fake quantization
- Post-training static quantization (INT8)
- Quantization-friendly layer replacements
"""

from __future__ import annotations

from dataclasses import dataclass

try:
  import torch
  import torch.nn as nn

  TORCH_AVAILABLE = True
except ImportError:
  TORCH_AVAILABLE = False


@dataclass
class QuantizationConfig:
  """Quantization configuration.

  Attributes:
    backend: Quantization backend ('x86', 'qnnpack', 'onednn')
    calibration_batches: Number of batches for calibration
    per_channel: Use per-channel quantization (better accuracy)
    symmetric: Use symmetric quantization
  """

  backend: str = "qnnpack"
  calibration_batches: int = 100
  per_channel: bool = True
  symmetric: bool = False


def replace_batchnorm_with_layernorm(model: nn.Module) -> nn.Module:
  """Replace BatchNorm layers with LayerNorm for quantization friendliness.

  BatchNorm has issues with quantization (running stats + affine params).
  LayerNorm is more stable for INT8.

  Args:
    model: Model to modify

  Returns:
    Model with BatchNorm replaced by LayerNorm
  """
  if not TORCH_AVAILABLE:
    raise ImportError("PyTorch required")

  for name, module in list(model.named_children()):
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
      num_features = module.num_features
      if isinstance(module, nn.BatchNorm2d):
        # Replace with GroupNorm(1, C) which is equivalent to LayerNorm for conv
        replacement = nn.GroupNorm(1, num_features, affine=True)
      else:
        replacement = nn.LayerNorm(num_features)
      setattr(model, name, replacement)
    else:
      replace_batchnorm_with_layernorm(module)

  return model


def fuse_conv_bn(model: nn.Module) -> nn.Module:
  """Fuse Conv+BN layers for quantization efficiency.

  Args:
    model: Model with separate Conv and BN layers

  Returns:
    Model with fused Conv+BN
  """
  if not TORCH_AVAILABLE:
    raise ImportError("PyTorch required")

  torch.quantization.fuse_modules(model, _find_fusable_pairs(model), inplace=True)
  return model


def _find_fusable_pairs(model: nn.Module) -> list[list[str]]:
  """Find Conv+BN+ReLU sequences that can be fused."""
  pairs = []
  modules = dict(model.named_modules())
  names = list(modules.keys())

  for i, name in enumerate(names):
    module = modules[name]
    if isinstance(module, (nn.Conv2d, nn.Conv1d)):
      # Check for BN following
      if i + 1 < len(names):
        next_mod = modules[names[i + 1]]
        if isinstance(next_mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
          pair = [name, names[i + 1]]
          # Check for ReLU following BN
          if i + 2 < len(names):
            next_next = modules[names[i + 2]]
            if isinstance(next_next, nn.ReLU):
              pair.append(names[i + 2])
          pairs.append(pair)

  return pairs


def prepare_qat(
  model: nn.Module,
  config: QuantizationConfig | None = None,
) -> nn.Module:
  """Prepare model for quantization-aware training.

  Inserts fake quantization observers into the model.

  Args:
    model: Model to prepare for QAT
    config: Quantization configuration

  Returns:
    Model with fake quantization layers inserted
  """
  if not TORCH_AVAILABLE:
    raise ImportError("PyTorch required for QAT")

  config = config or QuantizationConfig()

  torch.backends.quantized.engine = config.backend

  # Set QAT configuration
  if config.per_channel:
    qconfig = torch.quantization.get_default_qat_qconfig(config.backend)
  else:
    qconfig = torch.quantization.QConfig(
      activation=torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
      ),
      weight=torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
      ),
    )

  model.qconfig = qconfig
  model.train()

  # Prepare model (inserts fake quant modules)
  torch.quantization.prepare_qat(model, inplace=True)

  return model


def calibrate(
  model: nn.Module,
  dataloader: torch.utils.data.DataLoader,
  config: QuantizationConfig | None = None,
) -> None:
  """Calibrate quantization observers with representative data.

  Run after prepare_qat to collect activation statistics.

  Args:
    model: QAT-prepared model
    dataloader: Calibration data
    config: Quantization config
  """
  if not TORCH_AVAILABLE:
    raise ImportError("PyTorch required")

  config = config or QuantizationConfig()

  model.eval()
  with torch.no_grad():
    for i, batch in enumerate(dataloader):
      if i >= config.calibration_batches:
        break
      if isinstance(batch, dict):
        image = batch["image"]
      else:
        image = batch[0]
      model(image)


def convert_to_quantized(model: nn.Module) -> nn.Module:
  """Convert QAT model to actual INT8 quantized model.

  Call after QAT training or calibration is complete.

  Args:
    model: QAT-trained or calibrated model

  Returns:
    INT8 quantized model
  """
  if not TORCH_AVAILABLE:
    raise ImportError("PyTorch required")

  model.eval()
  quantized = torch.quantization.convert(model, inplace=False)
  return quantized
