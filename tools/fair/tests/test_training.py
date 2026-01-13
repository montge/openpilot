"""Tests for training utilities."""

import pytest
import numpy as np

# Check PyTorch availability
try:
  import torch
  import torch.nn as nn

  TORCH_AVAILABLE = True
except ImportError:
  TORCH_AVAILABLE = False

from openpilot.tools.fair.training.dora import DoRAConfig
from openpilot.tools.fair.training.multitask import MultiTaskConfig
from openpilot.tools.fair.training.dataset import FrameData


class TestDoRAConfig:
  """Tests for DoRA configuration."""

  def test_default_config(self):
    """Test default DoRA config."""
    config = DoRAConfig()
    assert config.rank == 8
    assert config.alpha == 8.0  # Should equal rank
    assert config.dropout == 0.0
    assert config.use_rslora is True

  def test_custom_config(self):
    """Test custom DoRA config."""
    config = DoRAConfig(rank=16, alpha=32.0, dropout=0.1)
    assert config.rank == 16
    assert config.alpha == 32.0
    assert config.dropout == 0.1

  def test_alpha_defaults_to_rank(self):
    """Test alpha defaults to rank value."""
    config = DoRAConfig(rank=32)
    assert config.alpha == 32.0


class TestMultiTaskConfig:
  """Tests for multi-task configuration."""

  def test_default_config(self):
    """Test default multi-task config."""
    config = MultiTaskConfig()
    assert "depth" in config.tasks
    assert "segmentation" in config.tasks
    assert config.epochs == 100
    assert config.uncertainty_weighting is True

  def test_custom_config(self):
    """Test custom multi-task config."""
    config = MultiTaskConfig(
      tasks=["depth", "detection"],
      task_weights={"depth": 2.0, "detection": 1.0},
      epochs=50,
    )
    assert config.tasks == ["depth", "detection"]
    assert config.task_weights["depth"] == 2.0


class TestFrameData:
  """Tests for frame data structure."""

  def test_frame_data_creation(self):
    """Test creating frame data."""
    image = np.zeros((256, 512, 3), dtype=np.uint8)
    frame = FrameData(image=image, timestamp=1000)

    assert frame.image.shape == (256, 512, 3)
    assert frame.timestamp == 1000
    assert frame.depth is None
    assert frame.segmentation is None

  def test_frame_data_with_labels(self):
    """Test frame data with labels."""
    image = np.zeros((256, 512, 3), dtype=np.uint8)
    depth = np.ones((256, 512), dtype=np.float32)
    segmentation = np.zeros((256, 512), dtype=np.int32)

    frame = FrameData(
      image=image,
      timestamp=1000,
      depth=depth,
      segmentation=segmentation,
    )

    assert frame.depth is not None
    assert frame.depth.shape == (256, 512)
    assert frame.segmentation is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDoRALayer:
  """Tests for DoRA layer."""

  def test_dora_layer_creation(self):
    """Test creating DoRA layer."""
    from openpilot.tools.fair.training.dora import DoRALayer

    base_layer = nn.Linear(64, 128)
    config = DoRAConfig(rank=8)
    dora = DoRALayer(base_layer, config)

    assert dora.lora_A.in_features == 64
    assert dora.lora_A.out_features == 8
    assert dora.lora_B.in_features == 8
    assert dora.lora_B.out_features == 128
    assert dora.magnitude.shape == (128,)

  def test_dora_forward(self):
    """Test DoRA forward pass."""
    from openpilot.tools.fair.training.dora import DoRALayer

    base_layer = nn.Linear(64, 128)
    config = DoRAConfig(rank=8)
    dora = DoRALayer(base_layer, config)

    x = torch.randn(4, 64)
    out = dora(x)

    assert out.shape == (4, 128)

  def test_dora_merge(self):
    """Test merging DoRA layer."""
    from openpilot.tools.fair.training.dora import DoRALayer

    base_layer = nn.Linear(64, 128)
    config = DoRAConfig(rank=8)
    dora = DoRALayer(base_layer, config)

    merged = dora.merge()

    assert isinstance(merged, nn.Linear)
    assert merged.weight.shape == (128, 64)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestApplyDoRA:
  """Tests for applying DoRA to models."""

  def test_apply_dora(self):
    """Test applying DoRA to a model."""
    from openpilot.tools.fair.training.dora import apply_dora, DoRALayer

    model = nn.Sequential(
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
    )

    config = DoRAConfig(rank=4)
    model = apply_dora(model, config, target_modules=["0", "2"])

    # Check that layers were replaced
    assert isinstance(model[0], DoRALayer)
    assert isinstance(model[2], DoRALayer)

  def test_get_dora_parameters(self):
    """Test getting DoRA trainable parameters."""
    from openpilot.tools.fair.training.dora import (
      get_dora_parameters,
    )

    model = nn.Linear(64, 128)
    config = DoRAConfig(rank=4)

    # Wrap directly
    from openpilot.tools.fair.training.dora import DoRALayer

    dora = DoRALayer(model, config)

    params = get_dora_parameters(nn.ModuleList([dora]))
    assert len(params) == 3  # lora_A, lora_B, magnitude


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestMultiTaskHead:
  """Tests for multi-task prediction head."""

  def test_head_creation(self):
    """Test creating multi-task head."""
    from openpilot.tools.fair.training.multitask import MultiTaskHead

    head = MultiTaskHead(256, ["depth", "segmentation"])

    assert "depth" in head.heads
    assert "segmentation" in head.heads

  def test_head_forward(self):
    """Test multi-task head forward."""
    from openpilot.tools.fair.training.multitask import MultiTaskHead

    head = MultiTaskHead(256, ["depth", "segmentation"], num_classes=19)

    features = torch.randn(2, 256, 8, 16)
    outputs = head(features)

    assert "depth" in outputs
    assert "segmentation" in outputs
    assert outputs["depth"].shape == (2, 1, 8, 16)
    assert outputs["segmentation"].shape == (2, 19, 8, 16)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestUncertaintyWeights:
  """Tests for uncertainty weighting."""

  def test_uncertainty_weights(self):
    """Test uncertainty weight learning."""
    from openpilot.tools.fair.training.multitask import UncertaintyWeights

    uw = UncertaintyWeights(["depth", "segmentation"])

    losses = {
      "depth": torch.tensor(1.0),
      "segmentation": torch.tensor(2.0),
    }

    total, weights = uw(losses)

    assert total > 0
    assert "depth" in weights
    assert "segmentation" in weights


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestRouteDataset:
  """Tests for route dataset."""

  def test_dataset_creation(self):
    """Test creating route dataset."""
    from openpilot.tools.fair.training.dataset import RouteDataset

    # Create with empty routes (will have 0 length)
    dataset = RouteDataset(
      route_paths=[],
      camera="road",
      image_size=(256, 512),
    )

    assert len(dataset) == 0

  def test_frame_preprocessing(self):
    """Test frame preprocessing."""
    from openpilot.tools.fair.training.dataset import RouteDataset

    dataset = RouteDataset([], image_size=(128, 256), augment=False)

    # Test preprocessing
    image = np.random.randint(0, 255, (128, 256, 3), dtype=np.uint8)
    processed = dataset._preprocess_image(image)

    assert processed.shape == (3, 128, 256)
    assert processed.min() >= 0
    assert processed.max() <= 1

  def test_depth_preprocessing(self):
    """Test depth preprocessing."""
    from openpilot.tools.fair.training.dataset import RouteDataset

    dataset = RouteDataset([], image_size=(128, 256))

    depth = np.random.uniform(0, 100, (128, 256)).astype(np.float32)
    processed = dataset._preprocess_depth(depth)

    assert processed.shape == (1, 128, 256)
    assert processed.min() >= 0
    assert processed.max() <= 1
