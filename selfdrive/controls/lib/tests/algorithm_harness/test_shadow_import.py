"""Tests for shadow device log import functionality."""

from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from openpilot.selfdrive.controls.lib.tests.algorithm_harness.interface import (
  LateralAlgorithmState,
  LongitudinalAlgorithmState,
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.shadow_import import (
  frame_to_lateral_state,
  frame_to_longitudinal_state,
  import_shadow_log,
  import_shadow_segment,
  compare_shadow_to_harness,
  format_shadow_comparison_report,
)
from openpilot.tools.shadow.comparison_logger import FrameData, ComparisonLogger


def make_test_frame(
  frame_id: int,
  timestamp: float,
  steer: float = 0.0,
  accel: float = 0.0,
  v_ego: float = 20.0,
  desired_curvature: float = 0.0,
) -> FrameData:
  """Create a test frame with common defaults."""
  return FrameData(
    frame_id=frame_id,
    timestamp_mono=timestamp,
    timestamp_gps=timestamp,
    controls={
      "steer_torque": steer,
      "accel": accel,
      "steering_angle_deg": steer * 15.0,
    },
    model_outputs={
      "desired_curvature": desired_curvature,
    },
    state={
      "v_ego": v_ego,
      "a_ego": 0.0,
      "lat_active": True,
      "long_active": True,
      "yaw_rate": 0.0,
    },
    trajectory={
      "x": [0.0, 1.0, 2.0],
      "y": [0.0, 0.1, 0.2],
    },
    events=[],
  )


class TestFrameConversion:
  """Tests for frame conversion functions."""

  def test_frame_to_lateral_state_basic(self):
    """Test basic lateral state conversion."""
    frame = make_test_frame(
      frame_id=1,
      timestamp=1.0,
      steer=0.5,
      v_ego=25.0,
      desired_curvature=0.01,
    )

    state = frame_to_lateral_state(frame)

    assert isinstance(state, LateralAlgorithmState)
    assert state.timestamp_ns == int(1.0 * 1e9)
    assert state.v_ego == 25.0
    assert state.desired_curvature == 0.01
    assert state.active is True

  def test_frame_to_lateral_state_inactive(self):
    """Test lateral state with inactive control."""
    frame = make_test_frame(frame_id=1, timestamp=1.0)
    frame.state["lat_active"] = False

    state = frame_to_lateral_state(frame)

    assert state.active is False

  def test_frame_to_longitudinal_state_basic(self):
    """Test basic longitudinal state conversion."""
    frame = make_test_frame(
      frame_id=1,
      timestamp=2.0,
      accel=1.5,
      v_ego=30.0,
    )
    frame.state["a_target"] = 2.0
    frame.state["should_stop"] = False

    state = frame_to_longitudinal_state(frame)

    assert isinstance(state, LongitudinalAlgorithmState)
    assert state.timestamp_ns == int(2.0 * 1e9)
    assert state.v_ego == 30.0
    assert state.a_target == 2.0

  def test_frame_to_longitudinal_state_stopping(self):
    """Test longitudinal state when stopping."""
    frame = make_test_frame(frame_id=1, timestamp=1.0, v_ego=0.5)
    frame.state["should_stop"] = True
    frame.state["cruise_standstill"] = True

    state = frame_to_longitudinal_state(frame)

    assert state.should_stop is True
    assert state.cruise_standstill is True

  def test_missing_fields_use_defaults(self):
    """Test that missing fields use sensible defaults."""
    frame = FrameData(
      frame_id=1,
      timestamp_mono=1.0,
      controls={},
      model_outputs={},
      state={},
      events=[],
      trajectory={},
    )

    lat_state = frame_to_lateral_state(frame)
    assert lat_state.v_ego == 0.0
    assert lat_state.desired_curvature == 0.0
    assert lat_state.active is True  # Default when not specified

    long_state = frame_to_longitudinal_state(frame)
    assert long_state.v_ego == 0.0
    assert long_state.a_target == 0.0


class TestImportShadowLog:
  """Tests for import_shadow_log function."""

  def test_import_lateral_log(self):
    """Test importing frames as lateral scenario."""
    frames = [
      make_test_frame(i, float(i) * 0.01, steer=0.1 * i, desired_curvature=0.001 * i)
      for i in range(100)
    ]

    scenario = import_shadow_log(frames, name="test_lateral", mode="lateral")

    assert scenario.name == "test_lateral"
    assert len(scenario.states) == 100
    assert all(isinstance(s, LateralAlgorithmState) for s in scenario.states)
    assert len(scenario.ground_truth) == 100

  def test_import_longitudinal_log(self):
    """Test importing frames as longitudinal scenario."""
    frames = [
      make_test_frame(i, float(i) * 0.01, accel=0.5)
      for i in range(50)
    ]

    scenario = import_shadow_log(frames, name="test_long", mode="longitudinal")

    assert scenario.name == "test_long"
    assert len(scenario.states) == 50
    assert all(isinstance(s, LongitudinalAlgorithmState) for s in scenario.states)
    # Ground truth should be accel values
    assert all(gt == 0.5 for gt in scenario.ground_truth)

  def test_import_with_description(self):
    """Test importing with custom description."""
    frames = [make_test_frame(0, 0.0)]

    scenario = import_shadow_log(
      frames,
      name="test",
      description="Custom description",
    )

    assert scenario.description == "Custom description"

  def test_import_metadata(self):
    """Test that metadata is populated."""
    frames = [
      make_test_frame(10, 1.0),
      make_test_frame(20, 2.0),
    ]

    scenario = import_shadow_log(frames, name="test")

    assert scenario.metadata["source"] == "shadow_device"
    assert scenario.metadata["frame_count"] == 2
    assert scenario.metadata["first_frame_id"] == 10
    assert scenario.metadata["last_frame_id"] == 20
    assert scenario.metadata["duration_s"] == 1.0

  def test_invalid_mode_raises(self):
    """Test that invalid mode raises ValueError."""
    frames = [make_test_frame(0, 0.0)]

    with pytest.raises(ValueError, match="Unknown mode"):
      import_shadow_log(frames, name="test", mode="invalid")


class TestImportShadowSegment:
  """Tests for import_shadow_segment function."""

  def test_import_from_directory(self, tmp_path: Path):
    """Test importing from a segment directory."""
    # Create test segment
    segment_dir = tmp_path / "test_segment"
    logger = ComparisonLogger(tmp_path, compress=False)
    logger.start_segment("test_segment")

    for i in range(50):
      frame = FrameData(
        frame_id=i,
        timestamp_mono=float(i) * 0.01,
        controls={"steer_torque": 0.1},
        model_outputs={"desired_curvature": 0.001},
        state={"v_ego": 20.0, "a_ego": 0.0, "lat_active": True},
      )
      logger.log_frame(frame)

    logger.end_segment()

    # Import
    scenario = import_shadow_segment(segment_dir, mode="lateral")

    assert len(scenario.states) == 50
    assert scenario.name == "test_segment"

  def test_import_with_subsample(self, tmp_path: Path):
    """Test importing with subsampling."""
    segment_dir = tmp_path / "test_segment"
    logger = ComparisonLogger(tmp_path, compress=False)
    logger.start_segment("test_segment")

    for i in range(100):
      frame = FrameData(
        frame_id=i,
        timestamp_mono=float(i) * 0.01,
        controls={"steer_torque": 0.1},
        state={"v_ego": 20.0},
      )
      logger.log_frame(frame)

    logger.end_segment()

    # Import with subsample
    scenario = import_shadow_segment(segment_dir, subsample=5)

    assert len(scenario.states) == 20  # 100 / 5

  def test_import_with_max_frames(self, tmp_path: Path):
    """Test importing with max frames limit."""
    segment_dir = tmp_path / "test_segment"
    logger = ComparisonLogger(tmp_path, compress=False)
    logger.start_segment("test_segment")

    for i in range(100):
      frame = FrameData(
        frame_id=i,
        timestamp_mono=float(i) * 0.01,
        controls={"steer_torque": 0.1},
        state={"v_ego": 20.0},
      )
      logger.log_frame(frame)

    logger.end_segment()

    # Import with max frames
    scenario = import_shadow_segment(segment_dir, max_frames=30)

    assert len(scenario.states) == 30

  def test_empty_directory_raises(self, tmp_path: Path):
    """Test that empty directory raises ValueError."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No frames found"):
      import_shadow_segment(empty_dir)


class TestCompareShadowToHarness:
  """Tests for compare_shadow_to_harness function."""

  def test_perfect_match(self):
    """Test comparison with perfect match."""
    frames = [
      make_test_frame(i, float(i) * 0.01, steer=0.5)
      for i in range(100)
    ]
    harness_outputs = [0.5] * 100  # Same as shadow

    metrics = compare_shadow_to_harness(frames, harness_outputs, mode="lateral")

    assert metrics["error_rmse"] < 0.001
    assert metrics["error_mae"] < 0.001
    assert metrics["error_max"] < 0.001

  def test_constant_error(self):
    """Test comparison with constant error."""
    frames = [
      make_test_frame(i, float(i) * 0.01, steer=0.5)
      for i in range(100)
    ]
    harness_outputs = [0.6] * 100  # Constant offset of 0.1

    metrics = compare_shadow_to_harness(frames, harness_outputs, mode="lateral")

    assert abs(metrics["error_rmse"] - 0.1) < 0.001
    assert abs(metrics["error_mae"] - 0.1) < 0.001
    assert abs(metrics["error_mean"] - 0.1) < 0.001  # Bias

  def test_longitudinal_comparison(self):
    """Test longitudinal mode comparison."""
    frames = [
      make_test_frame(i, float(i) * 0.01, accel=1.0)
      for i in range(50)
    ]
    harness_outputs = [1.2] * 50

    metrics = compare_shadow_to_harness(frames, harness_outputs, mode="longitudinal")

    assert abs(metrics["error_mean"] - 0.2) < 0.001

  def test_mismatched_lengths_raises(self):
    """Test that mismatched lengths raise ValueError."""
    frames = [make_test_frame(i, float(i)) for i in range(10)]
    harness_outputs = [0.0] * 5

    with pytest.raises(ValueError, match="doesn't match"):
      compare_shadow_to_harness(frames, harness_outputs)

  def test_correlation_computed(self):
    """Test that correlation is computed when there's variance."""
    import random
    random.seed(42)

    frames = [
      make_test_frame(i, float(i) * 0.01, steer=random.random())
      for i in range(100)
    ]
    # Harness outputs are correlated but not identical
    harness_outputs = [f.controls["steer_torque"] * 0.9 + 0.1 for f in frames]

    metrics = compare_shadow_to_harness(frames, harness_outputs)

    assert "correlation" in metrics
    assert metrics["correlation"] > 0.9  # Should be highly correlated


class TestFormatReport:
  """Tests for format_shadow_comparison_report function."""

  def test_basic_format(self):
    """Test basic report formatting."""
    metrics = {
      "n_samples": 100,
      "shadow_mean": 0.5,
      "shadow_std": 0.1,
      "harness_mean": 0.52,
      "harness_std": 0.11,
      "error_rmse": 0.05,
      "error_mae": 0.03,
      "error_max": 0.15,
      "error_mean": 0.02,
      "error_std": 0.04,
    }

    report = format_shadow_comparison_report(metrics, "TestAlgorithm")

    assert "TestAlgorithm" in report
    assert "100" in report
    assert "RMSE" in report
    assert "0.05" in report

  def test_format_with_correlation(self):
    """Test report with correlation metric."""
    metrics = {
      "n_samples": 100,
      "shadow_mean": 0.5,
      "shadow_std": 0.1,
      "harness_mean": 0.5,
      "harness_std": 0.1,
      "error_rmse": 0.01,
      "error_mae": 0.01,
      "error_max": 0.02,
      "error_mean": 0.0,
      "error_std": 0.01,
      "correlation": 0.95,
    }

    report = format_shadow_comparison_report(metrics)

    assert "Correlation" in report
    assert "0.95" in report
