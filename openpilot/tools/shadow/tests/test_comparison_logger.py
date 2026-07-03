"""Unit tests for ComparisonLogger."""

from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path

import pytest

from openpilot.tools.shadow.comparison_logger import ComparisonLogger, FrameData


class TestFrameData:
  """Tests for FrameData dataclass."""

  def test_frame_data_creation(self):
    """Test basic FrameData creation."""
    frame = FrameData(
      frame_id=100,
      timestamp_mono=1234.567,
      timestamp_gps=1700000000.0,
      model_outputs={"desired_curvature": 0.01},
      controls={"steer": 0.5, "accel": 1.0},
    )

    assert frame.frame_id == 100
    assert frame.timestamp_mono == 1234.567
    assert frame.timestamp_gps == 1700000000.0
    assert frame.model_outputs["desired_curvature"] == 0.01
    assert frame.controls["steer"] == 0.5

  def test_frame_data_to_dict(self):
    """Test FrameData serialization to dict."""
    frame = FrameData(
      frame_id=1,
      timestamp_mono=100.0,
      controls={"steer": 0.1},
    )

    d = frame.to_dict()
    assert d["frame_id"] == 1
    assert d["timestamp_mono"] == 100.0
    assert d["controls"]["steer"] == 0.1
    assert d["timestamp_gps"] is None

  def test_frame_data_from_dict(self):
    """Test FrameData deserialization from dict."""
    d = {
      "frame_id": 42,
      "timestamp_mono": 200.0,
      "timestamp_gps": 1700000001.0,
      "model_outputs": {"confidence": 0.95},
      "trajectory": {"x": [0, 1, 2]},
      "controls": {"accel": 2.0},
      "events": ["engage"],
      "state": {"enabled": True},
    }

    frame = FrameData.from_dict(d)
    assert frame.frame_id == 42
    assert frame.timestamp_gps == 1700000001.0
    assert frame.model_outputs["confidence"] == 0.95
    assert frame.trajectory["x"] == [0, 1, 2]
    assert frame.events == ["engage"]

  def test_frame_data_roundtrip(self):
    """Test FrameData serialization roundtrip."""
    original = FrameData(
      frame_id=999,
      timestamp_mono=12345.6789,
      timestamp_gps=1700000002.5,
      model_outputs={"a": 1, "b": [1, 2, 3]},
      trajectory={"x": [0.0, 0.1], "y": [0.0, 0.5]},
      controls={"steer": -0.5, "accel": 0.0},
      events=["override", "steer_saturated"],
      state={"lat_active": True, "long_active": False},
    )

    d = original.to_dict()
    restored = FrameData.from_dict(d)

    assert restored.frame_id == original.frame_id
    assert restored.timestamp_mono == original.timestamp_mono
    assert restored.timestamp_gps == original.timestamp_gps
    assert restored.model_outputs == original.model_outputs
    assert restored.trajectory == original.trajectory
    assert restored.controls == original.controls
    assert restored.events == original.events
    assert restored.state == original.state


class TestComparisonLogger:
  """Tests for ComparisonLogger class."""

  def test_logger_creation(self):
    """Test logger initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
      logger = ComparisonLogger(output_dir=tmpdir, device_id="test_device")

      assert logger.device_id == "test_device"
      assert logger.current_segment is None
      assert logger.total_frames == 0

  def test_start_end_segment(self):
    """Test segment lifecycle."""
    with tempfile.TemporaryDirectory() as tmpdir:
      logger = ComparisonLogger(output_dir=tmpdir)

      logger.start_segment("seg_001")
      assert logger.current_segment == "seg_001"

      logger.end_segment()
      assert logger.current_segment is None

  def test_log_frame_without_segment_raises(self):
    """Test that logging without segment raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
      logger = ComparisonLogger(output_dir=tmpdir)

      with pytest.raises(RuntimeError, match="Must call start_segment"):
        logger.log_frame(FrameData(frame_id=1, timestamp_mono=0.0))

  def test_log_single_frame(self):
    """Test logging a single frame."""
    with tempfile.TemporaryDirectory() as tmpdir:
      logger = ComparisonLogger(output_dir=tmpdir, compress=False)
      logger.start_segment("test_seg")

      frame = FrameData(
        frame_id=1,
        timestamp_mono=100.0,
        controls={"steer": 0.5},
      )
      logger.log_frame(frame)

      assert logger.total_frames == 1

      logger.end_segment()

      # Verify file was created
      seg_dir = Path(tmpdir) / "test_seg"
      files = list(seg_dir.glob("*.json"))
      assert len(files) == 1

  def test_log_multiple_frames(self):
    """Test logging multiple frames."""
    with tempfile.TemporaryDirectory() as tmpdir:
      logger = ComparisonLogger(output_dir=tmpdir, compress=False)
      logger.start_segment("multi_frame")

      for i in range(10):
        frame = FrameData(
          frame_id=i,
          timestamp_mono=float(i * 10),
          controls={"steer": float(i) / 10},
        )
        logger.log_frame(frame)

      assert logger.total_frames == 10
      logger.end_segment()

  def test_log_rotation(self):
    """Test log file rotation when max frames reached."""
    with tempfile.TemporaryDirectory() as tmpdir:
      logger = ComparisonLogger(
        output_dir=tmpdir,
        compress=False,
        max_frames_per_file=5,  # Small for testing
      )
      logger.start_segment("rotation_test")

      # Log 12 frames to trigger rotation
      for i in range(12):
        frame = FrameData(frame_id=i, timestamp_mono=float(i))
        logger.log_frame(frame)

      logger.end_segment()

      # Should have 3 files: 5 + 5 + 2 frames
      seg_dir = Path(tmpdir) / "rotation_test"
      files = sorted(seg_dir.glob("*.json"))
      assert len(files) == 3

  def test_compressed_logging(self):
    """Test gzip compressed logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
      logger = ComparisonLogger(output_dir=tmpdir, compress=True)
      logger.start_segment("compressed_seg")

      for i in range(5):
        frame = FrameData(frame_id=i, timestamp_mono=float(i))
        logger.log_frame(frame)

      logger.end_segment()

      # Verify compressed file exists
      seg_dir = Path(tmpdir) / "compressed_seg"
      gz_files = list(seg_dir.glob("*.json.gz"))
      assert len(gz_files) == 1

      # Verify it's valid gzip
      with gzip.open(gz_files[0], "rb") as f:
        data = json.loads(f.read().decode("utf-8"))
        assert data["frame_count"] == 5

  def test_load_segment(self):
    """Test loading frames from segment directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create and populate segment
      logger = ComparisonLogger(output_dir=tmpdir, compress=False)
      logger.start_segment("load_test")

      original_frames = []
      for i in range(10):
        frame = FrameData(
          frame_id=i,
          timestamp_mono=float(i * 10),
          controls={"steer": float(i) / 10},
        )
        original_frames.append(frame)
        logger.log_frame(frame)

      logger.end_segment()

      # Load and verify
      seg_dir = Path(tmpdir) / "load_test"
      loaded_frames = ComparisonLogger.load_segment(seg_dir)

      assert len(loaded_frames) == 10
      for i, frame in enumerate(loaded_frames):
        assert frame.frame_id == i
        assert frame.controls["steer"] == float(i) / 10

  def test_load_frame_range(self):
    """Test loading specific frame range."""
    with tempfile.TemporaryDirectory() as tmpdir:
      logger = ComparisonLogger(output_dir=tmpdir, compress=False)
      logger.start_segment("range_test")

      for i in range(20):
        frame = FrameData(frame_id=i, timestamp_mono=float(i))
        logger.log_frame(frame)

      logger.end_segment()

      # Load range
      seg_dir = Path(tmpdir) / "range_test"
      frames = ComparisonLogger.load_frame_range(seg_dir, 5, 15)

      assert len(frames) == 10
      assert frames[0].frame_id == 5
      assert frames[-1].frame_id == 14

  def test_convenience_log_model_output(self):
    """Test log_model_output convenience method."""
    with tempfile.TemporaryDirectory() as tmpdir:
      logger = ComparisonLogger(output_dir=tmpdir, compress=False)
      logger.start_segment("model_test")

      frame = logger.log_model_output(
        frame_id=1,
        desired_curvature=0.05,
        model_confidence=0.92,
        lead_car={"distance": 50.0},
      )

      assert frame.frame_id == 1
      assert frame.model_outputs["desired_curvature"] == 0.05
      assert frame.model_outputs["confidence"] == 0.92

      logger.end_segment()

  def test_convenience_log_control_command(self):
    """Test log_control_command convenience method."""
    with tempfile.TemporaryDirectory() as tmpdir:
      logger = ComparisonLogger(output_dir=tmpdir, compress=False)
      logger.start_segment("control_test")

      frame = logger.log_control_command(
        frame_id=1,
        steer_torque=0.3,
        accel=1.5,
        steering_angle_deg=15.0,
        lat_active=True,
        gps_time=1700000000.0,
      )

      assert frame.controls["steer_torque"] == 0.3
      assert frame.controls["accel"] == 1.5
      assert frame.state["lat_active"] is True
      assert frame.timestamp_gps == 1700000000.0

      logger.end_segment()

  def test_convenience_log_trajectory(self):
    """Test log_trajectory convenience method."""
    with tempfile.TemporaryDirectory() as tmpdir:
      logger = ComparisonLogger(output_dir=tmpdir, compress=False)
      logger.start_segment("traj_test")

      frame = logger.log_trajectory(
        frame_id=1,
        x_points=[0.0, 1.0, 2.0],
        y_points=[0.0, 0.1, 0.2],
        v_points=[10.0, 11.0, 12.0],
      )

      assert frame.trajectory["x"] == [0.0, 1.0, 2.0]
      assert frame.trajectory["y"] == [0.0, 0.1, 0.2]
      assert frame.trajectory["v"] == [10.0, 11.0, 12.0]

      logger.end_segment()

  def test_segment_auto_end_on_new_start(self):
    """Test that starting new segment ends previous one."""
    with tempfile.TemporaryDirectory() as tmpdir:
      logger = ComparisonLogger(output_dir=tmpdir, compress=False)

      logger.start_segment("seg1")
      logger.log_frame(FrameData(frame_id=1, timestamp_mono=0.0))

      # Starting new segment should end previous
      logger.start_segment("seg2")

      # seg1 should have been flushed
      seg1_dir = Path(tmpdir) / "seg1"
      assert seg1_dir.exists()
      assert len(list(seg1_dir.glob("*.json"))) == 1

      logger.end_segment()
