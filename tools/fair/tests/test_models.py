"""Tests for FAIR model wrappers."""

import numpy as np
import pytest

from openpilot.tools.fair.models import (
  DINOV2_AVAILABLE,
)
from openpilot.tools.fair.models.base import ModelConfig
from openpilot.tools.fair.models.dinov2 import DINOv2Config, DINOv2Wrapper
from openpilot.tools.fair.models.sam2 import SAM2Config, SAM2Wrapper, VideoTrackingState
from openpilot.tools.fair.models.cotracker import CoTrackerConfig, CoTrackerWrapper, TrackingResult
from openpilot.tools.fair.models.detr import DETRConfig, DETRWrapper, Detection, DetectionResult


class TestModelConfig:
  """Tests for ModelConfig dataclass."""

  def test_default_config(self):
    """Test default configuration values."""
    config = ModelConfig()
    assert config.model_name == "default"
    assert config.device == "auto"
    assert config.precision == "fp32"
    assert config.cache_dir is None

  def test_custom_config(self):
    """Test custom configuration values."""
    config = ModelConfig(
      model_name="custom_model",
      device="cuda",
      precision="fp16",
      cache_dir="/tmp/models",
    )
    assert config.model_name == "custom_model"
    assert config.device == "cuda"
    assert config.precision == "fp16"
    assert config.cache_dir == "/tmp/models"


class TestDINOv2Config:
  """Tests for DINOv2 configuration."""

  def test_default_config(self):
    """Test default DINOv2 config."""
    config = DINOv2Config()
    assert config.model_name == "dinov2_vitb14"
    assert config.with_registers is False
    assert config.image_size == 518

  def test_custom_config(self):
    """Test custom DINOv2 config."""
    config = DINOv2Config(
      model_name="dinov2_vitl14",
      with_registers=True,
      image_size=1024,
    )
    assert config.model_name == "dinov2_vitl14"
    assert config.with_registers is True


class TestDINOv2Wrapper:
  """Tests for DINOv2 wrapper."""

  def test_wrapper_initialization(self):
    """Test wrapper initializes correctly."""
    wrapper = DINOv2Wrapper()
    assert wrapper.config is not None
    assert wrapper.loaded is False
    assert wrapper._model is None

  def test_wrapper_with_config(self):
    """Test wrapper with custom config."""
    config = DINOv2Config(model_name="dinov2_vits14")
    wrapper = DINOv2Wrapper(config)
    assert wrapper.config.model_name == "dinov2_vits14"

  @pytest.mark.skipif(not DINOV2_AVAILABLE, reason="PyTorch not available")
  def test_device_resolution_auto(self):
    """Test auto device resolution."""
    wrapper = DINOv2Wrapper()
    device = wrapper._resolve_device()
    assert device in ("cuda", "mps", "cpu")

  def test_load_without_pytorch(self):
    """Test load fails gracefully without PyTorch."""
    if DINOV2_AVAILABLE:
      pytest.skip("PyTorch is available")

    wrapper = DINOv2Wrapper()
    with pytest.raises(ImportError):
      wrapper.load()


class TestSAM2Config:
  """Tests for SAM2 configuration."""

  def test_default_config(self):
    """Test default SAM2 config."""
    config = SAM2Config()
    assert config.model_name == "sam2_hiera_base_plus"
    assert config.points_per_side == 32
    assert config.pred_iou_thresh == 0.88
    assert config.stability_score_thresh == 0.95


class TestSAM2Wrapper:
  """Tests for SAM2 wrapper."""

  def test_wrapper_initialization(self):
    """Test wrapper initializes correctly."""
    wrapper = SAM2Wrapper()
    assert wrapper.config is not None
    assert wrapper.loaded is False
    assert wrapper._tracking_state is None


class TestVideoTrackingState:
  """Tests for VideoTrackingState."""

  def test_default_state(self):
    """Test default tracking state."""
    state = VideoTrackingState()
    assert state.object_ids == []
    assert state.masks == {}
    assert state.scores == {}
    assert state.memory is None

  def test_state_with_objects(self):
    """Test tracking state with objects."""
    masks = {1: np.zeros((100, 100)), 2: np.ones((100, 100))}
    state = VideoTrackingState(
      object_ids=[1, 2],
      masks=masks,
      scores={1: 0.9, 2: 0.85},
    )
    assert len(state.object_ids) == 2
    assert 1 in state.masks
    assert state.scores[1] == 0.9


class TestCoTrackerConfig:
  """Tests for CoTracker configuration."""

  def test_default_config(self):
    """Test default CoTracker config."""
    config = CoTrackerConfig()
    assert config.model_name == "cotracker2"
    assert config.grid_size == 10
    assert config.window_len == 8


class TestCoTrackerWrapper:
  """Tests for CoTracker wrapper."""

  def test_wrapper_initialization(self):
    """Test wrapper initializes correctly."""
    wrapper = CoTrackerWrapper()
    assert wrapper.config is not None
    assert wrapper.loaded is False


class TestTrackingResult:
  """Tests for TrackingResult dataclass."""

  def test_tracking_result(self):
    """Test tracking result creation."""
    tracks = np.random.randn(1, 10, 100, 2)
    visibility = np.ones((1, 10, 100), dtype=bool)

    result = TrackingResult(tracks=tracks, visibility=visibility)
    assert result.tracks.shape == (1, 10, 100, 2)
    assert result.visibility.shape == (1, 10, 100)
    assert result.confidence is None

  def test_tracking_result_with_confidence(self):
    """Test tracking result with confidence."""
    result = TrackingResult(
      tracks=np.zeros((1, 5, 50, 2)),
      visibility=np.ones((1, 5, 50), dtype=bool),
      confidence=np.ones((1, 5, 50)) * 0.9,
    )
    assert result.confidence is not None
    assert result.confidence.shape == (1, 5, 50)


class TestDETRConfig:
  """Tests for DETR configuration."""

  def test_default_config(self):
    """Test default DETR config."""
    config = DETRConfig()
    assert config.model_name == "detr_resnet50"
    assert config.confidence_threshold == 0.7
    assert config.num_classes == 91

  def test_custom_config(self):
    """Test custom DETR config."""
    config = DETRConfig(
      model_name="detr_resnet101",
      confidence_threshold=0.5,
    )
    assert config.model_name == "detr_resnet101"
    assert config.confidence_threshold == 0.5


class TestDETRWrapper:
  """Tests for DETR wrapper."""

  def test_wrapper_initialization(self):
    """Test wrapper initializes correctly."""
    wrapper = DETRWrapper()
    assert wrapper.config is not None
    assert wrapper.loaded is False

  def test_vehicle_classes(self):
    """Test vehicle class definition."""
    assert 3 in DETRWrapper.VEHICLE_CLASSES  # car
    assert 8 in DETRWrapper.VEHICLE_CLASSES  # truck
    assert 1 not in DETRWrapper.VEHICLE_CLASSES  # person


class TestDetection:
  """Tests for Detection dataclass."""

  def test_detection_creation(self):
    """Test detection creation."""
    det = Detection(
      box=np.array([100, 200, 300, 400]),
      score=0.95,
      label=3,
      label_name="car",
    )
    assert det.box.shape == (4,)
    assert det.score == 0.95
    assert det.label == 3
    assert det.label_name == "car"


class TestDetectionResult:
  """Tests for DetectionResult dataclass."""

  def test_detection_result(self):
    """Test detection result creation."""
    boxes = np.array([[100, 200, 300, 400], [150, 250, 350, 450]])
    scores = np.array([0.95, 0.85])
    labels = np.array([3, 8])

    result = DetectionResult(
      boxes=boxes,
      scores=scores,
      labels=labels,
      detections=[
        Detection(boxes[0], scores[0], labels[0], "car"),
        Detection(boxes[1], scores[1], labels[1], "truck"),
      ],
    )

    assert result.boxes.shape == (2, 4)
    assert len(result.detections) == 2
    assert result.detections[0].label_name == "car"

  def test_empty_result(self):
    """Test empty detection result."""
    result = DetectionResult(
      boxes=np.array([]).reshape(0, 4),
      scores=np.array([]),
      labels=np.array([], dtype=np.int64),
      detections=[],
    )
    assert len(result.detections) == 0
    assert result.boxes.shape == (0, 4)
