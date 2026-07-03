"""Tests for student model architectures."""

import pytest

# Check PyTorch availability
try:
  import torch

  TORCH_AVAILABLE = True
except ImportError:
  TORCH_AVAILABLE = False

from openpilot.tools.fair.students.vision import (
  TinyViTConfig,
  MobileViTConfig,
  EfficientStudentConfig,
)
from openpilot.tools.fair.students.detection import (
  TinyDETRConfig,
  MobileDetectorConfig,
)


class TestVisionConfigs:
  """Tests for vision model configurations."""

  def test_tinyvit_config_defaults(self):
    """Test TinyViT default config."""
    config = TinyViTConfig()
    assert config.image_size == 224
    assert config.patch_size == 16
    assert config.embed_dim == 192
    assert config.depth == 6
    assert config.num_heads == 3
    assert config.mlp_ratio == 4.0
    assert config.num_classes == 0

  def test_tinyvit_config_custom(self):
    """Test custom TinyViT config."""
    config = TinyViTConfig(
      image_size=384,
      embed_dim=256,
      depth=12,
      num_heads=4,
      num_classes=1000,
    )
    assert config.image_size == 384
    assert config.embed_dim == 256
    assert config.depth == 12
    assert config.num_classes == 1000

  def test_mobilevit_config_defaults(self):
    """Test MobileViT default config."""
    config = MobileViTConfig()
    assert config.image_size == 256
    assert config.patch_size == 2
    assert config.dims == (64, 128, 256)
    assert config.depths == (2, 4, 3)
    assert config.expansion == 4

  def test_efficientstudent_config_defaults(self):
    """Test EfficientStudent default config."""
    config = EfficientStudentConfig()
    assert config.image_size == 224
    assert config.base_channels == 32
    assert config.num_stages == 4
    assert config.num_classes == 0


class TestDetectionConfigs:
  """Tests for detection model configurations."""

  def test_tinydetr_config_defaults(self):
    """Test TinyDETR default config."""
    config = TinyDETRConfig()
    assert config.num_classes == 91
    assert config.hidden_dim == 128
    assert config.num_queries == 50
    assert config.num_encoder_layers == 2
    assert config.num_decoder_layers == 2
    assert config.num_heads == 4
    assert config.backbone == "mobilenet"

  def test_tinydetr_config_resnet(self):
    """Test TinyDETR with ResNet backbone."""
    config = TinyDETRConfig(backbone="resnet18")
    assert config.backbone == "resnet18"

  def test_mobiledetector_config_defaults(self):
    """Test MobileDetector default config."""
    config = MobileDetectorConfig()
    assert config.num_classes == 91
    assert config.base_channels == 64
    assert config.num_anchors == 3


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTinyViT:
  """Tests for TinyViT model."""

  def test_tinyvit_creation(self):
    """Test TinyViT model creation."""
    from openpilot.tools.fair.students.vision import TinyViT

    model = TinyViT()
    assert model is not None
    assert len(model.blocks) == 6

  def test_tinyvit_forward(self):
    """Test TinyViT forward pass."""
    from openpilot.tools.fair.students.vision import TinyViT

    config = TinyViTConfig(image_size=224, embed_dim=96, depth=3, num_heads=2)
    model = TinyViT(config)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
      out = model(x)

    assert out.shape == (2, 96)

  def test_tinyvit_forward_features(self):
    """Test TinyViT forward_features."""
    from openpilot.tools.fair.students.vision import TinyViT

    config = TinyViTConfig(image_size=224, embed_dim=96, depth=3, num_heads=2)
    model = TinyViT(config)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
      out = model.forward_features(x)

    assert "x_norm_clstoken" in out
    assert "x_norm_patchtokens" in out
    assert out["x_norm_clstoken"].shape == (2, 96)
    # 224/16 = 14, 14*14 = 196 patches
    assert out["x_norm_patchtokens"].shape == (2, 196, 96)

  def test_tinyvit_with_classification(self):
    """Test TinyViT with classification head."""
    from openpilot.tools.fair.students.vision import TinyViT

    config = TinyViTConfig(
      image_size=224,
      embed_dim=96,
      depth=3,
      num_heads=2,
      num_classes=10,
    )
    model = TinyViT(config)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
      out = model(x)

    assert out.shape == (2, 10)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestMobileViT:
  """Tests for MobileViT model."""

  def test_mobilevit_creation(self):
    """Test MobileViT model creation."""
    from openpilot.tools.fair.students.vision import MobileViT

    model = MobileViT()
    assert model is not None

  def test_mobilevit_forward(self):
    """Test MobileViT forward pass."""
    from openpilot.tools.fair.students.vision import MobileViT

    config = MobileViTConfig(
      image_size=128,
      dims=(32, 64, 128),
      depths=(1, 2, 1),
    )
    model = MobileViT(config)
    model.eval()

    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
      out = model(x)

    assert out.shape == (2, 128)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestEfficientStudent:
  """Tests for EfficientStudent model."""

  def test_efficientstudent_creation(self):
    """Test EfficientStudent model creation."""
    from openpilot.tools.fair.students.vision import EfficientStudent

    model = EfficientStudent()
    assert model is not None

  def test_efficientstudent_forward(self):
    """Test EfficientStudent forward pass."""
    from openpilot.tools.fair.students.vision import EfficientStudent

    config = EfficientStudentConfig(
      base_channels=16,
      num_stages=3,
    )
    model = EfficientStudent(config)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
      out = model(x)

    # After 3 stages with doubling, final channels = 16 * 4 = 64
    assert out.shape[0] == 2


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTinyDETR:
  """Tests for TinyDETR model."""

  def test_tinydetr_creation(self):
    """Test TinyDETR model creation."""
    from openpilot.tools.fair.students.detection import TinyDETR

    model = TinyDETR()
    assert model is not None

  def test_tinydetr_forward(self):
    """Test TinyDETR forward pass."""
    from openpilot.tools.fair.students.detection import TinyDETR

    config = TinyDETRConfig(
      hidden_dim=64,
      num_queries=10,
      num_encoder_layers=1,
      num_decoder_layers=1,
      num_heads=2,
    )
    model = TinyDETR(config)
    model.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
      out = model(x)

    assert "pred_logits" in out
    assert "pred_boxes" in out
    assert out["pred_logits"].shape == (2, 10, 92)  # num_classes + 1
    assert out["pred_boxes"].shape == (2, 10, 4)

  def test_tinydetr_resnet_backbone(self):
    """Test TinyDETR with ResNet backbone."""
    from openpilot.tools.fair.students.detection import TinyDETR

    config = TinyDETRConfig(
      backbone="resnet18",
      hidden_dim=64,
      num_queries=5,
      num_encoder_layers=1,
      num_decoder_layers=1,
    )
    model = TinyDETR(config)
    model.eval()

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
      out = model(x)

    assert out["pred_logits"].shape == (1, 5, 92)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestMobileDetector:
  """Tests for MobileDetector model."""

  def test_mobiledetector_creation(self):
    """Test MobileDetector model creation."""
    from openpilot.tools.fair.students.detection import MobileDetector

    model = MobileDetector()
    assert model is not None

  def test_mobiledetector_forward(self):
    """Test MobileDetector forward pass."""
    from openpilot.tools.fair.students.detection import MobileDetector

    config = MobileDetectorConfig(num_anchors=3)
    model = MobileDetector(config)
    model.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
      out = model(x)

    assert "pred_logits" in out
    assert "pred_boxes" in out
    # Output size depends on feature map size (256/64 = 4)
    # Total anchors: 4 * 4 * 3 = 48
    assert out["pred_boxes"].shape[-1] == 4


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestBackbones:
  """Tests for backbone networks."""

  def test_mobilenet_backbone(self):
    """Test MobileNet backbone."""
    from openpilot.tools.fair.students.detection import MobileNetBackbone

    backbone = MobileNetBackbone()
    backbone.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
      out = backbone(x)

    # 256 / 64 (6 strides of 2) = 4
    assert out.shape == (2, 320, 4, 4)

  def test_resnet18_backbone(self):
    """Test ResNet18 backbone."""
    from openpilot.tools.fair.students.detection import ResNet18Backbone

    backbone = ResNet18Backbone()
    backbone.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
      out = backbone(x)

    # 256 / 32 = 8
    assert out.shape == (2, 512, 8, 8)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPositionalEncoding:
  """Tests for positional encoding."""

  def test_positional_encoding_2d(self):
    """Test 2D positional encoding."""
    from openpilot.tools.fair.students.detection import PositionalEncoding2D

    pos_enc = PositionalEncoding2D(hidden_dim=64)

    pos = pos_enc(8, 8, torch.device("cpu"))

    assert pos.shape == (1, 64, 64)  # [1, H*W, D]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTransformerBlock:
  """Tests for transformer block."""

  def test_transformer_block(self):
    """Test transformer block forward."""
    from openpilot.tools.fair.students.vision import TransformerBlock

    block = TransformerBlock(dim=64, num_heads=4, mlp_ratio=4.0)
    block.eval()

    x = torch.randn(2, 16, 64)  # [B, N, D]
    with torch.no_grad():
      out = block(x)

    assert out.shape == x.shape
