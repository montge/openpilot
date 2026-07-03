"""Unit tests for metrics module."""

from __future__ import annotations


from openpilot.tools.shadow.align import AlignedPair, AlignmentResult
from openpilot.tools.shadow.comparison_logger import FrameData
from openpilot.tools.shadow.metrics import (
  compute_all_metrics,
  compute_control_metrics,
  compute_model_metrics,
  compute_time_series,
  compute_trajectory_metrics,
  format_report_markdown,
)


def make_pair(
  frame_id: int,
  shadow_steer: float,
  prod_steer: float,
  shadow_accel: float = 0.0,
  prod_accel: float = 0.0,
  shadow_curv: float | None = None,
  prod_curv: float | None = None,
) -> AlignedPair:
  """Helper to create test aligned pairs."""
  shadow_frame = FrameData(
    frame_id=frame_id,
    timestamp_mono=float(frame_id),
    controls={"steer": shadow_steer, "accel": shadow_accel},
    model_outputs={"desired_curvature": shadow_curv} if shadow_curv is not None else {},
  )
  prod_frame = FrameData(
    frame_id=frame_id,
    timestamp_mono=float(frame_id),
    controls={"steer": prod_steer, "accel": prod_accel},
    model_outputs={"desired_curvature": prod_curv} if prod_curv is not None else {},
  )
  return AlignedPair(
    shadow_frame=shadow_frame,
    production_frame=prod_frame,
    time_offset_ms=0.0,
    alignment_quality=1.0,
  )


class TestControlMetrics:
  """Tests for control metrics computation."""

  def test_identical_controls(self):
    """Test metrics when shadow and production are identical."""
    pairs = [
      make_pair(1, 0.5, 0.5, 1.0, 1.0),
      make_pair(2, 0.3, 0.3, 0.5, 0.5),
    ]

    metrics = compute_control_metrics(pairs)

    assert metrics.steer_rmse == 0.0
    assert metrics.steer_mae == 0.0
    assert metrics.accel_rmse == 0.0
    assert metrics.n_samples == 2

  def test_steer_error(self):
    """Test steer error computation."""
    pairs = [
      make_pair(1, 0.5, 0.4),  # 0.1 error
      make_pair(2, 0.3, 0.2),  # 0.1 error
    ]

    metrics = compute_control_metrics(pairs)

    assert abs(metrics.steer_rmse - 0.1) < 0.001
    assert abs(metrics.steer_mae - 0.1) < 0.001
    assert abs(metrics.steer_max_error - 0.1) < 0.001

  def test_accel_error(self):
    """Test accel error computation."""
    pairs = [
      make_pair(1, 0.0, 0.0, 1.0, 1.5),  # -0.5 error
      make_pair(2, 0.0, 0.0, 2.0, 1.5),  # 0.5 error
    ]

    metrics = compute_control_metrics(pairs)

    assert abs(metrics.accel_rmse - 0.5) < 0.001
    assert abs(metrics.accel_mae - 0.5) < 0.001

  def test_empty_pairs(self):
    """Test with no pairs."""
    metrics = compute_control_metrics([])

    assert metrics.steer_rmse == 0.0
    assert metrics.n_samples == 0


class TestTrajectoryMetrics:
  """Tests for trajectory metrics computation."""

  def test_trajectory_with_data(self):
    """Test trajectory metrics with trajectory data."""
    shadow_frame = FrameData(
      frame_id=1,
      timestamp_mono=1.0,
      trajectory={"y": [0.0, 0.1, 0.2], "v": [10.0, 11.0, 12.0]},
    )
    prod_frame = FrameData(
      frame_id=1,
      timestamp_mono=1.0,
      trajectory={"y": [0.0, 0.05, 0.1], "v": [10.0, 10.5, 11.0]},
    )

    pair = AlignedPair(shadow_frame, prod_frame, 0.0, 1.0)
    metrics = compute_trajectory_metrics([pair])

    assert metrics is not None
    assert metrics.path_rmse > 0
    assert metrics.speed_rmse > 0

  def test_trajectory_no_data(self):
    """Test trajectory metrics without trajectory data."""
    pairs = [make_pair(1, 0.5, 0.5)]
    metrics = compute_trajectory_metrics(pairs)

    assert metrics is None


class TestModelMetrics:
  """Tests for model output metrics computation."""

  def test_model_metrics_with_curvature(self):
    """Test model metrics with curvature data."""
    pairs = [
      make_pair(1, 0.0, 0.0, shadow_curv=0.01, prod_curv=0.01),
      make_pair(2, 0.0, 0.0, shadow_curv=0.02, prod_curv=0.02),
    ]

    metrics = compute_model_metrics(pairs)

    assert metrics is not None
    assert metrics.curvature_rmse == 0.0
    assert metrics.curvature_mae == 0.0

  def test_model_metrics_with_error(self):
    """Test model metrics with curvature error."""
    pairs = [
      make_pair(1, 0.0, 0.0, shadow_curv=0.01, prod_curv=0.02),
      make_pair(2, 0.0, 0.0, shadow_curv=0.03, prod_curv=0.02),
    ]

    metrics = compute_model_metrics(pairs)

    assert metrics is not None
    assert metrics.curvature_rmse > 0

  def test_model_metrics_no_data(self):
    """Test model metrics without curvature data."""
    pairs = [make_pair(1, 0.5, 0.5)]
    metrics = compute_model_metrics(pairs)

    assert metrics is None

  def test_model_correlation(self):
    """Test curvature correlation computation."""
    # Perfect correlation
    pairs = [make_pair(i, 0.0, 0.0, shadow_curv=float(i) * 0.001, prod_curv=float(i) * 0.001) for i in range(10)]

    metrics = compute_model_metrics(pairs)

    assert metrics is not None
    assert metrics.curvature_correlation > 0.99


class TestComputeAllMetrics:
  """Tests for complete metrics computation."""

  def test_compute_all_metrics(self):
    """Test computing all metrics from alignment result."""
    pairs = [make_pair(i, float(i) * 0.1, float(i) * 0.1 + 0.01, shadow_curv=0.01, prod_curv=0.01) for i in range(5)]

    result = AlignmentResult(
      pairs=pairs,
      shadow_only=[],
      production_only=[],
      mean_time_offset_ms=0.0,
      alignment_quality=0.95,
      method="gps",
    )

    report = compute_all_metrics(result)

    assert report.n_aligned_pairs == 5
    assert report.control_metrics.n_samples == 5
    assert report.alignment_quality == 0.95


class TestFormatReport:
  """Tests for report formatting."""

  def test_format_markdown(self):
    """Test markdown report formatting."""
    pairs = [make_pair(1, 0.5, 0.4)]

    result = AlignmentResult(
      pairs=pairs,
      shadow_only=[],
      production_only=[],
      mean_time_offset_ms=0.0,
      alignment_quality=0.95,
      method="gps",
    )

    report = compute_all_metrics(result)
    md = format_report_markdown(report)

    assert "# Shadow Device Comparison Report" in md
    assert "## Control Metrics" in md
    assert "RMSE" in md


class TestTimeSeries:
  """Tests for time series extraction."""

  def test_steer_time_series(self):
    """Test extracting steer time series."""
    pairs = [
      make_pair(1, 0.5, 0.4),
      make_pair(2, 0.6, 0.5),
      make_pair(3, 0.7, 0.6),
    ]

    ts = compute_time_series(pairs, "steer")

    assert len(ts["time"]) == 3
    assert len(ts["shadow"]) == 3
    assert len(ts["production"]) == 3
    assert len(ts["error"]) == 3
    assert ts["shadow"] == [0.5, 0.6, 0.7]
    assert ts["production"] == [0.4, 0.5, 0.6]

  def test_curvature_time_series(self):
    """Test extracting curvature time series."""
    pairs = [
      make_pair(1, 0.0, 0.0, shadow_curv=0.01, prod_curv=0.02),
      make_pair(2, 0.0, 0.0, shadow_curv=0.03, prod_curv=0.04),
    ]

    ts = compute_time_series(pairs, "curvature")

    assert ts["shadow"] == [0.01, 0.03]
    assert ts["production"] == [0.02, 0.04]
