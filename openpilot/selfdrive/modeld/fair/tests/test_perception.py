"""Tests for FAIR perception modules."""

import numpy as np
import pytest

from openpilot.selfdrive.modeld.fair.depth import DepthConfig, DepthEstimator, DepthHead
from openpilot.selfdrive.modeld.fair.segmentation import (
  DrivingClass,
  SegmentationConfig,
  SegmentationModule,
)
from openpilot.selfdrive.modeld.fair.lane_tracking import (
  LaneLine,
  LanePoint,
  LaneTracker,
  LaneTrackingConfig,
)


class TestDepthConfig:
  """Tests for DepthConfig."""

  def test_default_config(self):
    """Test default depth configuration."""
    config = DepthConfig()
    assert config.model_path is None
    assert config.input_size == (256, 512)
    assert config.output_size == (128, 256)
    assert config.max_depth == 100.0
    assert config.min_depth == 0.1
    assert config.use_gpu is True

  def test_custom_config(self):
    """Test custom depth configuration."""
    config = DepthConfig(
      model_path="/path/to/model.onnx",
      input_size=(512, 1024),
      output_size=(256, 512),
      max_depth=200.0,
      min_depth=0.5,
      use_gpu=False,
    )
    assert config.model_path == "/path/to/model.onnx"
    assert config.input_size == (512, 1024)
    assert config.max_depth == 200.0


class TestDepthEstimator:
  """Tests for DepthEstimator."""

  def test_initialization(self):
    """Test depth estimator initialization."""
    estimator = DepthEstimator()
    assert estimator.config is not None
    assert estimator._loaded is False
    assert estimator._model is None

  def test_initialization_with_config(self):
    """Test depth estimator with custom config."""
    config = DepthConfig(max_depth=50.0)
    estimator = DepthEstimator(config)
    assert estimator.config.max_depth == 50.0

  def test_load_unload(self):
    """Test load and unload."""
    estimator = DepthEstimator()

    estimator.load()
    assert estimator._loaded is True

    estimator.unload()
    assert estimator._loaded is False
    assert estimator._model is None

  def test_estimate_without_load(self):
    """Test estimate raises without load."""
    estimator = DepthEstimator()
    image = np.zeros((256, 512, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="Model not loaded"):
      estimator.estimate(image)

  def test_estimate(self):
    """Test depth estimation."""
    estimator = DepthEstimator()
    estimator.load()

    image = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)
    depth = estimator.estimate(image)

    assert depth.shape == estimator.config.output_size
    assert depth.min() >= estimator.config.min_depth
    assert depth.max() <= estimator.config.max_depth

    estimator.unload()

  def test_estimate_batch(self):
    """Test batch depth estimation."""
    estimator = DepthEstimator()
    estimator.load()

    images = np.random.randint(0, 255, (3, 256, 512, 3), dtype=np.uint8)
    depths = estimator.estimate_batch(images)

    assert depths.shape == (3,) + estimator.config.output_size

    estimator.unload()

  def test_context_manager(self):
    """Test context manager usage."""
    config = DepthConfig()
    with DepthEstimator(config) as estimator:
      assert estimator._loaded is True
      image = np.zeros((256, 512, 3), dtype=np.uint8)
      depth = estimator.estimate(image)
      assert depth is not None

    # Should be unloaded after context
    assert estimator._loaded is False


class TestDepthHead:
  """Tests for DepthHead."""

  def test_initialization(self):
    """Test depth head initialization."""
    head = DepthHead(in_channels=256, output_size=(128, 256))
    assert head.in_channels == 256
    assert head.output_size == (128, 256)

  def test_forward(self):
    """Test depth head forward pass."""
    head = DepthHead(in_channels=128, output_size=(64, 128))
    features = np.random.randn(2, 128, 16, 32).astype(np.float32)

    depth = head.forward(features)

    assert depth.shape == (2, 64, 128)


class TestDrivingClass:
  """Tests for DrivingClass enum."""

  def test_class_values(self):
    """Test class values are correct."""
    assert DrivingClass.ROAD == 0
    assert DrivingClass.LANE_MARKING == 1
    assert DrivingClass.VEHICLE == 2
    assert DrivingClass.PEDESTRIAN == 3
    assert DrivingClass.SKY == 7
    assert DrivingClass.UNKNOWN == 255

  def test_class_membership(self):
    """Test class enum membership."""
    assert DrivingClass.ROAD in DrivingClass
    assert 0 in [c.value for c in DrivingClass]


class TestSegmentationConfig:
  """Tests for SegmentationConfig."""

  def test_default_config(self):
    """Test default segmentation configuration."""
    config = SegmentationConfig()
    assert config.model_path is None
    assert config.input_size == (256, 512)
    assert config.output_size == (128, 256)
    assert config.num_classes == 10
    assert config.use_gpu is True

  def test_custom_config(self):
    """Test custom segmentation configuration."""
    config = SegmentationConfig(
      model_path="/path/to/seg.onnx",
      num_classes=20,
      use_gpu=False,
    )
    assert config.model_path == "/path/to/seg.onnx"
    assert config.num_classes == 20
    assert config.use_gpu is False


class TestSegmentationModule:
  """Tests for SegmentationModule."""

  def test_initialization(self):
    """Test segmentation module initialization."""
    module = SegmentationModule()
    assert module.config is not None
    assert module._loaded is False

  def test_load_unload(self):
    """Test load and unload."""
    module = SegmentationModule()

    module.load()
    assert module._loaded is True

    module.unload()
    assert module._loaded is False

  def test_segment_without_load(self):
    """Test segment raises without load."""
    module = SegmentationModule()
    image = np.zeros((256, 512, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="Model not loaded"):
      module.segment(image)

  def test_segment(self):
    """Test segmentation."""
    module = SegmentationModule()
    module.load()

    image = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)
    mask = module.segment(image)

    assert mask.shape == module.config.output_size
    assert mask.dtype == np.uint8

    module.unload()

  def test_segment_probs(self):
    """Test probability segmentation."""
    module = SegmentationModule()
    module.load()

    image = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)
    probs = module.segment_probs(image)

    assert probs.shape[0] == module.config.num_classes
    assert probs.shape[1:] == module.config.output_size
    # Probabilities should sum to 1
    np.testing.assert_array_almost_equal(probs.sum(axis=0), 1.0, decimal=5)

    module.unload()

  def test_get_road_mask(self):
    """Test road mask extraction."""
    module = SegmentationModule()
    module.load()

    image = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)
    road_mask = module.get_road_mask(image)

    assert road_mask.shape == module.config.output_size
    assert road_mask.dtype == bool

    module.unload()

  def test_get_drivable_area(self):
    """Test drivable area mask."""
    module = SegmentationModule()
    module.load()

    image = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)
    drivable = module.get_drivable_area(image)

    assert drivable.shape == module.config.output_size
    assert drivable.dtype == bool

    module.unload()

  def test_context_manager(self):
    """Test context manager usage."""
    with SegmentationModule() as module:
      assert module._loaded is True
      image = np.zeros((256, 512, 3), dtype=np.uint8)
      mask = module.segment(image)
      assert mask is not None


class TestLanePoint:
  """Tests for LanePoint dataclass."""

  def test_default_point(self):
    """Test default lane point creation."""
    point = LanePoint(x=100.0, y=200.0)
    assert point.x == 100.0
    assert point.y == 200.0
    assert point.confidence == 1.0
    assert point.visible is True
    assert point.lane_id == 0

  def test_custom_point(self):
    """Test custom lane point creation."""
    point = LanePoint(
      x=150.5,
      y=250.5,
      confidence=0.85,
      visible=False,
      lane_id=1,
    )
    assert point.x == 150.5
    assert point.confidence == 0.85
    assert point.visible is False
    assert point.lane_id == 1


class TestLaneLine:
  """Tests for LaneLine dataclass."""

  def test_default_lane(self):
    """Test default lane line creation."""
    lane = LaneLine()
    assert lane.points == []
    assert lane.lane_id == 0
    assert lane.lane_type == "solid"
    assert lane.color == "white"
    assert lane.confidence == 1.0

  def test_lane_with_points(self):
    """Test lane line with points."""
    points = [
      LanePoint(x=100, y=200),
      LanePoint(x=110, y=250),
      LanePoint(x=120, y=300),
    ]
    lane = LaneLine(
      points=points,
      lane_id=1,
      lane_type="dashed",
      color="yellow",
      confidence=0.9,
    )
    assert len(lane.points) == 3
    assert lane.lane_type == "dashed"
    assert lane.color == "yellow"


class TestLaneTrackingConfig:
  """Tests for LaneTrackingConfig."""

  def test_default_config(self):
    """Test default lane tracking configuration."""
    config = LaneTrackingConfig()
    assert config.num_points_per_lane == 10
    assert config.max_lanes == 4
    assert config.min_confidence == 0.3
    assert config.temporal_smoothing == 0.7
    assert config.search_radius == 20

  def test_custom_config(self):
    """Test custom lane tracking configuration."""
    config = LaneTrackingConfig(
      num_points_per_lane=20,
      max_lanes=6,
      min_confidence=0.5,
    )
    assert config.num_points_per_lane == 20
    assert config.max_lanes == 6
    assert config.min_confidence == 0.5


class TestLaneTracker:
  """Tests for LaneTracker."""

  def test_initialization(self):
    """Test lane tracker initialization."""
    tracker = LaneTracker()
    assert tracker.config is not None
    assert tracker._initialized is False
    assert tracker._lanes == []
    assert tracker._prev_frame is None

  def test_initialization_with_config(self):
    """Test lane tracker with custom config."""
    config = LaneTrackingConfig(max_lanes=2)
    tracker = LaneTracker(config)
    assert tracker.config.max_lanes == 2

  def test_detect_lanes(self):
    """Test lane detection."""
    tracker = LaneTracker()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    lanes = tracker.detect_lanes(image)

    assert tracker._initialized is True
    assert len(lanes) >= 1
    assert all(isinstance(lane, LaneLine) for lane in lanes)

  def test_update(self):
    """Test lane tracking update."""
    tracker = LaneTracker()
    image1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    image2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # First detection
    lanes1 = tracker.detect_lanes(image1)
    assert len(lanes1) >= 1

    # Update with new frame
    lanes2 = tracker.update(image2)
    assert len(lanes2) >= 1

  def test_get_lane_points(self):
    """Test getting lane points by ID."""
    tracker = LaneTracker()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    tracker.detect_lanes(image)

    points = tracker.get_lane_points(0)
    assert isinstance(points, list)
    if points:
      assert all(isinstance(p, tuple) and len(p) == 2 for p in points)

  def test_get_ego_lanes(self):
    """Test getting ego lanes."""
    tracker = LaneTracker()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    tracker.detect_lanes(image)

    left, right = tracker.get_ego_lanes()
    # At least one should be detected
    assert left is not None or right is not None

  def test_fit_polynomial(self):
    """Test polynomial fitting to lane."""
    tracker = LaneTracker()
    points = [
      LanePoint(x=100, y=200),
      LanePoint(x=105, y=250),
      LanePoint(x=110, y=300),
      LanePoint(x=115, y=350),
    ]
    lane = LaneLine(points=points)

    coeffs = tracker.fit_polynomial(lane, degree=2)
    assert coeffs is not None
    assert len(coeffs) == 3  # degree + 1 coefficients

  def test_fit_polynomial_insufficient_points(self):
    """Test polynomial fitting with insufficient points."""
    tracker = LaneTracker()
    points = [LanePoint(x=100, y=200)]
    lane = LaneLine(points=points)

    coeffs = tracker.fit_polynomial(lane, degree=2)
    assert coeffs is None

  def test_reset(self):
    """Test tracker reset."""
    tracker = LaneTracker()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    tracker.detect_lanes(image)

    assert tracker._initialized is True

    tracker.reset()

    assert tracker._initialized is False
    assert tracker._lanes == []
    assert tracker._prev_frame is None

  def test_update_without_detection(self):
    """Test update initializes tracking if not initialized."""
    tracker = LaneTracker()
    assert tracker._initialized is False

    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    lanes = tracker.update(image)

    assert tracker._initialized is True
    assert len(lanes) >= 1

  def test_temporal_consistency(self):
    """Test temporal consistency across frames."""
    tracker = LaneTracker()

    # Process multiple frames
    for _ in range(5):
      image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
      lanes = tracker.update(image)

      # Should maintain tracking
      assert tracker._initialized is True
      assert len(lanes) >= 1
