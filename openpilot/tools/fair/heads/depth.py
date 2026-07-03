"""Depth estimation heads for DINOv2 and other vision transformers.

Provides various depth prediction heads that can be attached to
vision transformer backbones like DINOv2.

Architectures:
- LinearDepthHead: Simple linear probe (fast, limited capacity)
- DPTDepthHead: Dense Prediction Transformer head (accurate, more compute)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

# Check PyTorch availability
try:
  import torch
  import torch.nn as nn
  import torch.nn.functional as F

  TORCH_AVAILABLE = True
  _BaseModule = nn.Module
except ImportError:
  TORCH_AVAILABLE = False
  _BaseModule = object


class DepthHead(Protocol):
  """Protocol for depth prediction heads."""

  def forward(self, features: np.ndarray) -> np.ndarray:
    """Predict depth from features."""
    ...


@dataclass
class LinearDepthConfig:
  """Configuration for linear depth head.

  Attributes:
    embed_dim: Input feature dimension
    hidden_dim: Hidden layer dimension (0 for no hidden layer)
    output_size: Output depth map size (H, W)
    min_depth: Minimum depth value
    max_depth: Maximum depth value
  """

  embed_dim: int = 768
  hidden_dim: int = 384
  output_size: tuple[int, int] = (128, 256)
  min_depth: float = 0.1
  max_depth: float = 100.0


class LinearDepthHead(_BaseModule):
  """Simple linear probe for depth estimation.

  Fastest option but limited capacity. Good for fine-tuning
  or when compute is constrained.
  """

  def __init__(self, config: LinearDepthConfig):
    """Initialize linear depth head.

    Args:
      config: Head configuration
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required for LinearDepthHead")

    super().__init__()
    self.config = config

    if config.hidden_dim > 0:
      self.head = nn.Sequential(
        nn.Linear(config.embed_dim, config.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(config.hidden_dim, 1),
      )
    else:
      self.head = nn.Linear(config.embed_dim, 1)

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    """Predict depth from patch features.

    Args:
      features: Patch features [B, N, D] from vision transformer

    Returns:
      Depth map [B, H, W]
    """
    b, n, _ = features.shape

    # Compute spatial dimensions from patch count
    h = w = int(math.sqrt(n))

    # Project to depth values
    depth = self.head(features)  # [B, N, 1]
    depth = depth.squeeze(-1)  # [B, N]

    # Reshape to spatial grid
    depth = depth.view(b, h, w)

    # Upsample to output size
    depth = F.interpolate(
      depth.unsqueeze(1),
      size=self.config.output_size,
      mode="bilinear",
      align_corners=False,
    ).squeeze(1)

    # Scale to depth range
    depth = torch.sigmoid(depth)
    depth = self.config.min_depth + depth * (self.config.max_depth - self.config.min_depth)

    return depth


@dataclass
class DPTDepthConfig:
  """Configuration for DPT depth head.

  Attributes:
    embed_dim: Input feature dimension
    features: Hidden feature dimensions for decoder
    output_size: Output depth map size (H, W)
    min_depth: Minimum depth value
    max_depth: Maximum depth value
    use_bn: Use batch normalization
  """

  embed_dim: int = 768
  features: int = 256
  output_size: tuple[int, int] = (128, 256)
  min_depth: float = 0.1
  max_depth: float = 100.0
  use_bn: bool = False


class DPTDepthHead(_BaseModule):
  """Dense Prediction Transformer (DPT) style depth head.

  Implements a simplified DPT decoder for monocular depth estimation.
  Better accuracy than linear probe but more computation.

  Reference: https://arxiv.org/abs/2103.13413
  """

  def __init__(self, config: DPTDepthConfig):
    """Initialize DPT depth head.

    Args:
      config: Head configuration
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required for DPTDepthHead")

    super().__init__()
    self.config = config

    features = config.features

    # Project from transformer dimension
    self.project = nn.Conv2d(config.embed_dim, features, kernel_size=1)

    # Fusion blocks (simplified DPT)
    self.fusion1 = self._make_fusion_block(features, config.use_bn)
    self.fusion2 = self._make_fusion_block(features, config.use_bn)
    self.fusion3 = self._make_fusion_block(features, config.use_bn)

    # Output head
    self.output_conv = nn.Sequential(
      nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(features // 2, 1, kernel_size=1),
    )

  def _make_fusion_block(self, features: int, use_bn: bool) -> nn.Module:
    """Create a fusion block for feature refinement."""
    layers = [
      nn.Conv2d(features, features, kernel_size=3, padding=1),
    ]
    if use_bn:
      layers.append(nn.BatchNorm2d(features))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1))
    if use_bn:
      layers.append(nn.BatchNorm2d(features))
    layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    """Predict depth from patch features.

    Args:
      features: Patch features [B, N, D] from vision transformer

    Returns:
      Depth map [B, H, W]
    """
    b, n, d = features.shape

    # Compute spatial dimensions
    h = w = int(math.sqrt(n))

    # Reshape to spatial feature map
    x = features.permute(0, 2, 1)  # [B, D, N]
    x = x.view(b, d, h, w)  # [B, D, H, W]

    # Project to decoder dimension
    x = self.project(x)

    # Progressive upsampling with fusion
    x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    x = self.fusion1(x)

    x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    x = self.fusion2(x)

    x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    x = self.fusion3(x)

    # Final depth prediction
    depth = self.output_conv(x)
    depth = depth.squeeze(1)  # [B, H', W']

    # Resize to target output size
    depth = F.interpolate(
      depth.unsqueeze(1),
      size=self.config.output_size,
      mode="bilinear",
      align_corners=False,
    ).squeeze(1)

    # Scale to depth range using sigmoid
    depth = torch.sigmoid(depth)
    depth = self.config.min_depth + depth * (self.config.max_depth - self.config.min_depth)

    return depth


class MultiScaleDepthHead(_BaseModule):
  """Multi-scale depth head using features from multiple transformer layers.

  Combines features from different depths in the transformer for
  better depth estimation across scales.
  """

  def __init__(
    self,
    embed_dim: int = 768,
    num_layers: int = 4,
    features: int = 256,
    output_size: tuple[int, int] = (128, 256),
    min_depth: float = 0.1,
    max_depth: float = 100.0,
  ):
    """Initialize multi-scale depth head.

    Args:
      embed_dim: Transformer embedding dimension
      num_layers: Number of layer features to use
      features: Hidden feature dimension
      output_size: Output depth map size
      min_depth: Minimum depth value
      max_depth: Maximum depth value
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required")

    super().__init__()
    self.output_size = output_size
    self.min_depth = min_depth
    self.max_depth = max_depth

    # Project each layer's features
    self.projections = nn.ModuleList([nn.Conv2d(embed_dim, features, kernel_size=1) for _ in range(num_layers)])

    # Fusion convolution
    self.fusion = nn.Sequential(
      nn.Conv2d(features * num_layers, features, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(features, features, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
    )

    # Output head
    self.output = nn.Conv2d(features, 1, kernel_size=1)

  def forward(self, layer_features: list[torch.Tensor]) -> torch.Tensor:
    """Predict depth from multi-layer features.

    Args:
      layer_features: List of [B, N, D] features from different layers

    Returns:
      Depth map [B, H, W]
    """
    b, n, d = layer_features[0].shape
    h = w = int(math.sqrt(n))

    # Project and reshape each layer
    projected = []
    for feat, proj in zip(layer_features, self.projections, strict=True):
      x = feat.permute(0, 2, 1).view(b, d, h, w)
      x = proj(x)
      # Upsample all to same size
      x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
      projected.append(x)

    # Concatenate and fuse
    x = torch.cat(projected, dim=1)
    x = self.fusion(x)

    # Predict depth
    depth = self.output(x).squeeze(1)

    # Resize to output
    depth = F.interpolate(
      depth.unsqueeze(1),
      size=self.output_size,
      mode="bilinear",
      align_corners=False,
    ).squeeze(1)

    # Scale to depth range
    depth = torch.sigmoid(depth)
    depth = self.min_depth + depth * (self.max_depth - self.min_depth)

    return depth


def create_depth_head(
  head_type: str = "dpt",
  embed_dim: int = 768,
  output_size: tuple[int, int] = (128, 256),
  **kwargs,
) -> _BaseModule:
  """Factory function to create depth heads.

  Args:
    head_type: Type of head ('linear', 'dpt', 'multiscale')
    embed_dim: Input embedding dimension
    output_size: Output depth map size
    **kwargs: Additional config options

  Returns:
    Initialized depth head
  """
  if head_type == "linear":
    config = LinearDepthConfig(
      embed_dim=embed_dim,
      output_size=output_size,
      **kwargs,
    )
    return LinearDepthHead(config)

  elif head_type == "dpt":
    config = DPTDepthConfig(
      embed_dim=embed_dim,
      output_size=output_size,
      **kwargs,
    )
    return DPTDepthHead(config)

  elif head_type == "multiscale":
    return MultiScaleDepthHead(
      embed_dim=embed_dim,
      output_size=output_size,
      **kwargs,
    )

  else:
    raise ValueError(f"Unknown head type: {head_type}")
