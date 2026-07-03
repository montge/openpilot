"""Detection student models for distillation.

Lightweight detection models that can be distilled from DETR and other
large detection transformers.
"""

from __future__ import annotations

from dataclasses import dataclass


# Check PyTorch availability
try:
  import torch
  import torch.nn as nn

  TORCH_AVAILABLE = True
  _BaseModule = nn.Module
except ImportError:
  TORCH_AVAILABLE = False
  # Placeholder base class when PyTorch not available
  _BaseModule = object


@dataclass
class TinyDETRConfig:
  """TinyDETR configuration.

  Attributes:
    num_classes: Number of detection classes
    hidden_dim: Hidden dimension for transformer
    num_queries: Number of object queries
    num_encoder_layers: Number of encoder layers
    num_decoder_layers: Number of decoder layers
    num_heads: Number of attention heads
    backbone: Backbone type ('resnet18', 'mobilenet')
  """

  num_classes: int = 91
  hidden_dim: int = 128
  num_queries: int = 50
  num_encoder_layers: int = 2
  num_decoder_layers: int = 2
  num_heads: int = 4
  backbone: str = "mobilenet"


class TinyDETR(_BaseModule):
  """Tiny DETR for distillation.

  A lightweight end-to-end detector based on DETR architecture.

  Usage:
    config = TinyDETRConfig(hidden_dim=128, num_queries=50)
    model = TinyDETR(config)

    outputs = model(images)
  """

  def __init__(self, config: TinyDETRConfig | None = None):
    """Initialize TinyDETR.

    Args:
      config: Model configuration
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required for student models")

    super().__init__()
    self.config = config or TinyDETRConfig()

    # Backbone
    if self.config.backbone == "mobilenet":
      self.backbone = MobileNetBackbone()
      backbone_channels = 320
    else:
      self.backbone = ResNet18Backbone()
      backbone_channels = 512

    # Input projection
    self.input_proj = nn.Conv2d(backbone_channels, self.config.hidden_dim, 1)

    # Positional encoding
    self.pos_encoding = PositionalEncoding2D(self.config.hidden_dim)

    # Transformer
    self.transformer = nn.Transformer(
      d_model=self.config.hidden_dim,
      nhead=self.config.num_heads,
      num_encoder_layers=self.config.num_encoder_layers,
      num_decoder_layers=self.config.num_decoder_layers,
      dim_feedforward=self.config.hidden_dim * 4,
      batch_first=True,
    )

    # Object queries
    self.query_embed = nn.Embedding(self.config.num_queries, self.config.hidden_dim)

    # Output heads
    self.class_head = nn.Linear(self.config.hidden_dim, self.config.num_classes + 1)
    self.bbox_head = nn.Sequential(
      nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
      nn.ReLU(),
      nn.Linear(self.config.hidden_dim, 4),
      nn.Sigmoid(),
    )

  def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Forward pass.

    Args:
      x: Input images [B, C, H, W]

    Returns:
      Dictionary with pred_logits [B, N, num_classes+1] and
      pred_boxes [B, N, 4] in cxcywh format
    """
    B = x.shape[0]

    # Backbone features
    features = self.backbone(x)  # [B, C, H', W']

    # Project to hidden dim
    src = self.input_proj(features)  # [B, D, H', W']
    _, _, H, W = src.shape

    # Add positional encoding
    pos = self.pos_encoding(H, W, src.device)  # [1, H*W, D]

    # Flatten spatial dimensions
    src = src.flatten(2).transpose(1, 2)  # [B, H*W, D]
    src = src + pos

    # Object queries
    query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]

    # Transformer
    memory = self.transformer.encoder(src)
    hs = self.transformer.decoder(query_embed, memory)  # [B, N, D]

    # Output predictions
    pred_logits = self.class_head(hs)  # [B, N, num_classes+1]
    pred_boxes = self.bbox_head(hs)  # [B, N, 4]

    return {
      "pred_logits": pred_logits,
      "pred_boxes": pred_boxes,
    }


class MobileNetBackbone(_BaseModule):
  """Lightweight MobileNet-style backbone."""

  def __init__(self):
    """Initialize MobileNet backbone."""
    super().__init__()

    self.features = nn.Sequential(
      # Initial conv
      nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU6(inplace=True),
      # Depthwise separable blocks
      self._make_block(32, 64, stride=2),
      self._make_block(64, 128, stride=2),
      self._make_block(128, 128, stride=1),
      self._make_block(128, 256, stride=2),
      self._make_block(256, 256, stride=1),
      self._make_block(256, 320, stride=2),
    )

  def _make_block(self, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    """Create depthwise separable block."""
    return nn.Sequential(
      # Depthwise
      nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
      nn.BatchNorm2d(in_channels),
      nn.ReLU6(inplace=True),
      # Pointwise
      nn.Conv2d(in_channels, out_channels, 1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU6(inplace=True),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass."""
    return self.features(x)


class ResNet18Backbone(_BaseModule):
  """Lightweight ResNet-18 style backbone."""

  def __init__(self):
    """Initialize ResNet backbone."""
    super().__init__()

    self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

    self.layer1 = self._make_layer(64, 64, 2)
    self.layer2 = self._make_layer(64, 128, 2, stride=2)
    self.layer3 = self._make_layer(128, 256, 2, stride=2)
    self.layer4 = self._make_layer(256, 512, 2, stride=2)

  def _make_layer(
    self,
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int = 1,
  ) -> nn.Sequential:
    """Create ResNet layer."""
    layers = []

    # First block with optional downsample
    downsample = None
    if stride != 1 or in_channels != out_channels:
      downsample = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
      )

    layers.append(ResBlock(in_channels, out_channels, stride, downsample))

    for _ in range(1, num_blocks):
      layers.append(ResBlock(out_channels, out_channels))

    return nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass."""
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x


class ResBlock(_BaseModule):
  """Basic residual block."""

  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    downsample: nn.Module | None = None,
  ):
    """Initialize residual block."""
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.downsample = downsample

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward with residual."""
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


class PositionalEncoding2D(_BaseModule):
  """2D positional encoding for transformers."""

  def __init__(self, hidden_dim: int):
    """Initialize positional encoding.

    Args:
      hidden_dim: Hidden dimension (must be even)
    """
    super().__init__()
    self.hidden_dim = hidden_dim

  def forward(self, h: int, w: int, device: torch.device) -> torch.Tensor:
    """Generate positional encoding.

    Args:
      h: Height
      w: Width
      device: Device to create tensor on

    Returns:
      Positional encoding [1, H*W, D]
    """
    y_embed = torch.arange(h, device=device).float()
    x_embed = torch.arange(w, device=device).float()

    # Normalize to [0, 1]
    y_embed = y_embed / (h - 1 + 1e-6)
    x_embed = x_embed / (w - 1 + 1e-6)

    # Create grid
    y_embed = y_embed.view(h, 1).expand(h, w)
    x_embed = x_embed.view(1, w).expand(h, w)

    # Create sinusoidal encoding
    dim_t = self.hidden_dim // 2
    dim = torch.arange(dim_t, device=device).float()
    dim = 10000 ** (2 * dim / dim_t)

    pos_x = x_embed.flatten()[:, None] / dim
    pos_y = y_embed.flatten()[:, None] / dim

    pos = torch.cat(
      [
        torch.sin(pos_x),
        torch.cos(pos_x),
        torch.sin(pos_y),
        torch.cos(pos_y),
      ],
      dim=1,
    )

    return pos[:, : self.hidden_dim].unsqueeze(0)


@dataclass
class MobileDetectorConfig:
  """MobileDetector configuration.

  Simple anchor-based detector for mobile.

  Attributes:
    num_classes: Number of detection classes
    base_channels: Base channel count
    num_anchors: Anchors per location
  """

  num_classes: int = 91
  base_channels: int = 64
  num_anchors: int = 3


class MobileDetector(_BaseModule):
  """Simple mobile detector.

  Anchor-based single-shot detector for fast inference.

  Usage:
    model = MobileDetector()
    outputs = model(images)
  """

  def __init__(self, config: MobileDetectorConfig | None = None):
    """Initialize MobileDetector.

    Args:
      config: Model configuration
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required for student models")

    super().__init__()
    self.config = config or MobileDetectorConfig()

    # Backbone
    self.backbone = MobileNetBackbone()

    # Detection head
    self.class_head = nn.Conv2d(
      320,
      self.config.num_anchors * (self.config.num_classes + 1),
      3,
      padding=1,
    )
    self.bbox_head = nn.Conv2d(
      320,
      self.config.num_anchors * 4,
      3,
      padding=1,
    )

  def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Forward pass.

    Args:
      x: Input images [B, C, H, W]

    Returns:
      Dictionary with pred_logits and pred_boxes
    """
    # Backbone
    features = self.backbone(x)  # [B, 320, H', W']

    # Detection heads
    class_out = self.class_head(features)  # [B, A*(C+1), H', W']
    bbox_out = self.bbox_head(features)  # [B, A*4, H', W']

    B, _, H, W = class_out.shape
    A = self.config.num_anchors
    C = self.config.num_classes + 1

    # Reshape outputs
    class_out = class_out.view(B, A, C, H, W).permute(0, 3, 4, 1, 2)  # [B, H, W, A, C]
    class_out = class_out.reshape(B, -1, C)  # [B, H*W*A, C]

    bbox_out = bbox_out.view(B, A, 4, H, W).permute(0, 3, 4, 1, 2)  # [B, H, W, A, 4]
    bbox_out = bbox_out.reshape(B, -1, 4)  # [B, H*W*A, 4]
    bbox_out = torch.sigmoid(bbox_out)  # Normalize to [0, 1]

    return {
      "pred_logits": class_out,
      "pred_boxes": bbox_out,
    }
