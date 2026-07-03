"""Vision student models for distillation.

Lightweight vision models that can be distilled from DINOv2 and other
large vision transformers.
"""

from __future__ import annotations

from dataclasses import dataclass


# Check PyTorch availability
try:
  import torch
  import torch.nn as nn
  import torch.nn.functional as F

  TORCH_AVAILABLE = True
  _BaseModule = nn.Module
except ImportError:
  TORCH_AVAILABLE = False
  # Placeholder base class when PyTorch not available
  _BaseModule = object


@dataclass
class TinyViTConfig:
  """TinyViT configuration.

  Attributes:
    image_size: Input image size
    patch_size: Patch size for embedding
    embed_dim: Embedding dimension
    depth: Number of transformer blocks
    num_heads: Number of attention heads
    mlp_ratio: MLP hidden dim ratio
    num_classes: Number of output classes (0 for feature extraction)
  """

  image_size: int = 224
  patch_size: int = 16
  embed_dim: int = 192
  depth: int = 6
  num_heads: int = 3
  mlp_ratio: float = 4.0
  num_classes: int = 0


class TinyViT(_BaseModule):
  """Tiny Vision Transformer for distillation.

  A lightweight ViT suitable for distilling from larger models like DINOv2.

  Usage:
    config = TinyViTConfig(embed_dim=192, depth=6)
    model = TinyViT(config)

    features = model(images)
  """

  def __init__(self, config: TinyViTConfig | None = None):
    """Initialize TinyViT.

    Args:
      config: Model configuration
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required for student models")

    super().__init__()
    self.config = config or TinyViTConfig()

    # Patch embedding
    self.patch_embed = nn.Conv2d(
      3,
      self.config.embed_dim,
      kernel_size=self.config.patch_size,
      stride=self.config.patch_size,
    )

    # Positional embedding
    num_patches = (self.config.image_size // self.config.patch_size) ** 2
    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.config.embed_dim))
    self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.embed_dim))

    # Transformer blocks
    self.blocks = nn.ModuleList(
      [
        TransformerBlock(
          dim=self.config.embed_dim,
          num_heads=self.config.num_heads,
          mlp_ratio=self.config.mlp_ratio,
        )
        for _ in range(self.config.depth)
      ]
    )

    self.norm = nn.LayerNorm(self.config.embed_dim)

    # Optional classification head
    if self.config.num_classes > 0:
      self.head = nn.Linear(self.config.embed_dim, self.config.num_classes)
    else:
      self.head = nn.Identity()

    self._init_weights()

  def _init_weights(self) -> None:
    """Initialize weights."""
    nn.init.trunc_normal_(self.pos_embed, std=0.02)
    nn.init.trunc_normal_(self.cls_token, std=0.02)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass.

    Args:
      x: Input images [B, C, H, W]

    Returns:
      Features or logits [B, D] or [B, num_classes]
    """
    B = x.shape[0]

    # Patch embedding
    x = self.patch_embed(x)  # [B, D, H/P, W/P]
    x = x.flatten(2).transpose(1, 2)  # [B, N, D]

    # Add cls token
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat([cls_tokens, x], dim=1)

    # Add position embedding
    x = x + self.pos_embed

    # Transformer blocks
    for block in self.blocks:
      x = block(x)

    x = self.norm(x)

    # Return cls token
    return self.head(x[:, 0])

  def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Forward pass returning all features.

    Args:
      x: Input images [B, C, H, W]

    Returns:
      Dictionary with cls token and patch tokens
    """
    B = x.shape[0]

    # Patch embedding
    x = self.patch_embed(x)
    x = x.flatten(2).transpose(1, 2)

    # Add cls token
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat([cls_tokens, x], dim=1)
    x = x + self.pos_embed

    # Transformer blocks
    for block in self.blocks:
      x = block(x)

    x = self.norm(x)

    return {
      "x_norm_clstoken": x[:, 0],
      "x_norm_patchtokens": x[:, 1:],
    }


class TransformerBlock(_BaseModule):
  """Transformer block with attention and MLP."""

  def __init__(
    self,
    dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
  ):
    """Initialize transformer block.

    Args:
      dim: Input dimension
      num_heads: Number of attention heads
      mlp_ratio: MLP hidden dimension ratio
      dropout: Dropout rate
    """
    super().__init__()

    self.norm1 = nn.LayerNorm(dim)
    self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
    self.norm2 = nn.LayerNorm(dim)

    mlp_dim = int(dim * mlp_ratio)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(mlp_dim, dim),
      nn.Dropout(dropout),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass with residual connections."""
    # Self-attention with residual
    x_norm = self.norm1(x)
    attn_out, _ = self.attn(x_norm, x_norm, x_norm)
    x = x + attn_out

    # MLP with residual
    x = x + self.mlp(self.norm2(x))

    return x


@dataclass
class MobileViTConfig:
  """MobileViT configuration.

  Combines convolutional and transformer layers for mobile efficiency.

  Attributes:
    image_size: Input image size
    patch_size: Patch size for transformer
    dims: Channel dimensions for each stage
    depths: Number of blocks per stage
    expansion: MobileNet expansion ratio
    num_classes: Number of output classes
  """

  image_size: int = 256
  patch_size: int = 2
  dims: tuple[int, ...] = (64, 128, 256)
  depths: tuple[int, ...] = (2, 4, 3)
  expansion: int = 4
  num_classes: int = 0


class MobileViT(_BaseModule):
  """MobileViT - efficient vision transformer for mobile.

  Combines MobileNet-style convolutions with local transformer blocks.

  Usage:
    config = MobileViTConfig()
    model = MobileViT(config)

    features = model(images)
  """

  def __init__(self, config: MobileViTConfig | None = None):
    """Initialize MobileViT.

    Args:
      config: Model configuration
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required for student models")

    super().__init__()
    self.config = config or MobileViTConfig()

    # Initial convolution
    self.stem = nn.Sequential(
      nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.SiLU(),
    )

    # Build stages
    self.stages = nn.ModuleList()
    in_channels = 32

    for dim, depth in zip(self.config.dims, self.config.depths, strict=True):
      stage = nn.Sequential(
        # Downsample
        nn.Conv2d(in_channels, dim, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(dim),
        nn.SiLU(),
        # Transformer blocks
        *[MobileViTBlock(dim, self.config.expansion) for _ in range(depth)],
      )
      self.stages.append(stage)
      in_channels = dim

    # Head
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.flatten = nn.Flatten()

    if self.config.num_classes > 0:
      self.head = nn.Linear(self.config.dims[-1], self.config.num_classes)
    else:
      self.head = nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass.

    Args:
      x: Input images [B, C, H, W]

    Returns:
      Features or logits
    """
    x = self.stem(x)

    for stage in self.stages:
      x = stage(x)

    x = self.pool(x)
    x = self.flatten(x)
    return self.head(x)


class MobileViTBlock(_BaseModule):
  """MobileViT block with local transformer."""

  def __init__(self, dim: int, expansion: int = 4):
    """Initialize MobileViT block.

    Args:
      dim: Channel dimension
      expansion: MLP expansion ratio
    """
    super().__init__()

    # Local representation
    self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(dim)

    # Transformer
    self.transformer = nn.TransformerEncoderLayer(
      d_model=dim,
      nhead=4,
      dim_feedforward=dim * expansion,
      batch_first=True,
    )

    # Projection back
    self.conv2 = nn.Conv2d(dim, dim, 1, bias=False)
    self.bn2 = nn.BatchNorm2d(dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward with residual."""
    residual = x

    # Local conv
    x = F.silu(self.bn1(self.conv1(x)))

    # Reshape for transformer
    B, C, H, W = x.shape
    x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

    # Transformer
    x = self.transformer(x)

    # Reshape back
    x = x.transpose(1, 2).view(B, C, H, W)

    # Project
    x = self.bn2(self.conv2(x))

    return x + residual


@dataclass
class EfficientStudentConfig:
  """EfficientStudent configuration.

  A simple CNN-based student model for basic distillation.

  Attributes:
    image_size: Input image size
    base_channels: Base channel count
    num_stages: Number of downsampling stages
    num_classes: Number of output classes
  """

  image_size: int = 224
  base_channels: int = 32
  num_stages: int = 4
  num_classes: int = 0


class EfficientStudent(_BaseModule):
  """Efficient CNN student model.

  Simple and fast CNN for basic distillation experiments.

  Usage:
    model = EfficientStudent()
    features = model(images)
  """

  def __init__(self, config: EfficientStudentConfig | None = None):
    """Initialize EfficientStudent.

    Args:
      config: Model configuration
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required for student models")

    super().__init__()
    self.config = config or EfficientStudentConfig()

    layers = []
    in_channels = 3
    out_channels = self.config.base_channels

    for _i in range(self.config.num_stages):
      layers.extend(
        [
          nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        ]
      )
      in_channels = out_channels
      out_channels = min(out_channels * 2, 512)

    self.features = nn.Sequential(*layers)
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.flatten = nn.Flatten()

    final_channels = in_channels
    if self.config.num_classes > 0:
      self.head = nn.Linear(final_channels, self.config.num_classes)
    else:
      self.head = nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass."""
    x = self.features(x)
    x = self.pool(x)
    x = self.flatten(x)
    return self.head(x)
