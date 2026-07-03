"""Tests for FAIR model heads."""

import pytest

from openpilot.tools.fair.heads.depth import (
  LinearDepthConfig,
  LinearDepthHead,
  DPTDepthConfig,
  DPTDepthHead,
  MultiScaleDepthHead,
  create_depth_head,
)

# Check PyTorch availability
try:
  import torch

  TORCH_AVAILABLE = True
except ImportError:
  TORCH_AVAILABLE = False


class TestLinearDepthConfig:
  """Tests for LinearDepthConfig."""

  def test_default_config(self):
    """Test default configuration."""
    config = LinearDepthConfig()
    assert config.embed_dim == 768
    assert config.hidden_dim == 384
    assert config.output_size == (128, 256)
    assert config.min_depth == 0.1
    assert config.max_depth == 100.0

  def test_custom_config(self):
    """Test custom configuration."""
    config = LinearDepthConfig(
      embed_dim=384,
      hidden_dim=0,
      output_size=(64, 128),
      min_depth=0.5,
      max_depth=50.0,
    )
    assert config.embed_dim == 384
    assert config.hidden_dim == 0


class TestLinearDepthHead:
  """Tests for LinearDepthHead."""

  @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
  def test_initialization(self):
    """Test head initialization."""
    config = LinearDepthConfig()
    head = LinearDepthHead(config)
    assert head.config == config

  @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
  def test_forward(self):
    """Test forward pass."""
    config = LinearDepthConfig(embed_dim=384, output_size=(64, 128))
    head = LinearDepthHead(config)

    # Create dummy features [B, N, D] where N = 16x16 = 256
    features = torch.randn(2, 256, 384)

    depth = head(features)

    assert depth.shape == (2, 64, 128)
    assert (depth >= config.min_depth).all()
    assert (depth <= config.max_depth).all()

  @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
  def test_no_hidden_layer(self):
    """Test with no hidden layer."""
    config = LinearDepthConfig(embed_dim=256, hidden_dim=0)
    head = LinearDepthHead(config)

    features = torch.randn(1, 196, 256)  # 14x14 patches
    depth = head(features)

    assert depth.shape[0] == 1


class TestDPTDepthConfig:
  """Tests for DPTDepthConfig."""

  def test_default_config(self):
    """Test default configuration."""
    config = DPTDepthConfig()
    assert config.embed_dim == 768
    assert config.features == 256
    assert config.use_bn is False

  def test_custom_config(self):
    """Test custom configuration."""
    config = DPTDepthConfig(
      embed_dim=384,
      features=128,
      use_bn=True,
    )
    assert config.features == 128
    assert config.use_bn is True


class TestDPTDepthHead:
  """Tests for DPTDepthHead."""

  @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
  def test_initialization(self):
    """Test head initialization."""
    config = DPTDepthConfig()
    head = DPTDepthHead(config)
    assert head.config == config

  @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
  def test_forward(self):
    """Test forward pass."""
    config = DPTDepthConfig(embed_dim=256, features=64, output_size=(64, 128))
    head = DPTDepthHead(config)

    # 16x16 patch grid
    features = torch.randn(2, 256, 256)

    depth = head(features)

    assert depth.shape == (2, 64, 128)
    assert (depth >= config.min_depth).all()
    assert (depth <= config.max_depth).all()

  @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
  def test_with_batch_norm(self):
    """Test with batch normalization."""
    config = DPTDepthConfig(embed_dim=128, features=32, use_bn=True)
    head = DPTDepthHead(config)

    features = torch.randn(4, 64, 128)  # 8x8 patches
    depth = head(features)

    assert depth.shape[0] == 4


class TestMultiScaleDepthHead:
  """Tests for MultiScaleDepthHead."""

  @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
  def test_initialization(self):
    """Test head initialization."""
    head = MultiScaleDepthHead(embed_dim=256, num_layers=4)
    assert head.output_size == (128, 256)

  @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
  def test_forward(self):
    """Test forward pass with multiple layer features."""
    head = MultiScaleDepthHead(
      embed_dim=128,
      num_layers=3,
      features=64,
      output_size=(32, 64),
    )

    # List of features from different layers
    layer_features = [
      torch.randn(2, 64, 128),  # 8x8 patches
      torch.randn(2, 64, 128),
      torch.randn(2, 64, 128),
    ]

    depth = head(layer_features)

    assert depth.shape == (2, 32, 64)


class TestCreateDepthHead:
  """Tests for create_depth_head factory."""

  @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
  def test_create_linear(self):
    """Test creating linear head."""
    head = create_depth_head(
      head_type="linear",
      embed_dim=384,
      output_size=(64, 128),
    )
    assert isinstance(head, LinearDepthHead)

  @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
  def test_create_dpt(self):
    """Test creating DPT head."""
    head = create_depth_head(
      head_type="dpt",
      embed_dim=768,
      output_size=(128, 256),
    )
    assert isinstance(head, DPTDepthHead)

  @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
  def test_create_multiscale(self):
    """Test creating multiscale head."""
    head = create_depth_head(
      head_type="multiscale",
      embed_dim=512,
      num_layers=3,
    )
    assert isinstance(head, MultiScaleDepthHead)

  @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
  def test_invalid_type(self):
    """Test error for invalid head type."""
    with pytest.raises(ValueError, match="Unknown head type"):
      create_depth_head(head_type="invalid")
