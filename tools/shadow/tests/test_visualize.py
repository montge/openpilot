"""Tests for shadow device visualization tools."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openpilot.tools.shadow.align import AlignedPair, AlignmentResult
from openpilot.tools.shadow.comparison_logger import ComparisonFrame
from openpilot.tools.shadow.metrics import ComparisonReport, ControlMetrics


def make_test_frame(
  frame_id: int,
  timestamp: float,
  steer: float = 0.0,
  accel: float = 0.0,
) -> ComparisonFrame:
  """Create a test comparison frame."""
  return ComparisonFrame(
    frame_id=frame_id,
    timestamp_gps=timestamp,
    timestamp_mono=timestamp,
    controls={"steer_torque": steer, "accel": accel},
    trajectory={"y": [0.0, 0.1, 0.2], "v": [10.0, 10.1, 10.2]},
    model_outputs={"desired_curvature": 0.001},
    events=[],
    state={},
  )


def make_test_pairs(n: int = 100) -> list[AlignedPair]:
  """Create test aligned pairs."""
  import random
  pairs = []
  for i in range(n):
    t = float(i) * 0.05  # 20Hz
    shadow = make_test_frame(i, t, steer=0.1 * random.random(), accel=0.5 * random.random())
    prod = make_test_frame(i, t, steer=0.1 * random.random(), accel=0.5 * random.random())
    pairs.append(AlignedPair(shadow_frame=shadow, production_frame=prod, time_offset=0.0))
  return pairs


def make_test_result() -> AlignmentResult:
  """Create a test alignment result."""
  return AlignmentResult(
    pairs=make_test_pairs(100),
    shadow_only=[],
    production_only=[],
    alignment_quality=0.95,
    method="frame_id",
  )


def make_test_report() -> ComparisonReport:
  """Create a test comparison report."""
  return ComparisonReport(
    control_metrics=ControlMetrics(
      steer_rmse=0.05,
      steer_mae=0.03,
      steer_max_error=0.15,
      accel_rmse=0.2,
      accel_mae=0.1,
      accel_max_error=0.5,
      n_samples=100,
    ),
    trajectory_metrics=None,
    model_metrics=None,
    alignment_quality=0.95,
    n_aligned_pairs=100,
    n_shadow_only=5,
    n_production_only=3,
    alignment_method="frame_id",
  )


class TestVisualizationImports:
  """Test visualization module imports."""

  def test_matplotlib_check(self):
    """Test matplotlib availability check."""
    from openpilot.tools.shadow import visualize
    # Should either work or raise ImportError
    assert hasattr(visualize, 'MATPLOTLIB_AVAILABLE')


class TestTimeSeries:
  """Test time series plotting."""

  @pytest.mark.skipif(
    not __import__('openpilot.tools.shadow.visualize', fromlist=['MATPLOTLIB_AVAILABLE']).MATPLOTLIB_AVAILABLE,
    reason="matplotlib not available"
  )
  def test_plot_time_series_steer(self, tmp_path: Path):
    """Test steer time series plot."""
    from openpilot.tools.shadow.visualize import plot_time_series

    pairs = make_test_pairs(50)
    output_path = tmp_path / "steer_timeseries.png"

    plot_time_series(pairs, "steer", output_path=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0

  @pytest.mark.skipif(
    not __import__('openpilot.tools.shadow.visualize', fromlist=['MATPLOTLIB_AVAILABLE']).MATPLOTLIB_AVAILABLE,
    reason="matplotlib not available"
  )
  def test_plot_time_series_accel(self, tmp_path: Path):
    """Test accel time series plot."""
    from openpilot.tools.shadow.visualize import plot_time_series

    pairs = make_test_pairs(50)
    output_path = tmp_path / "accel_timeseries.png"

    plot_time_series(pairs, "accel", output_path=output_path)

    assert output_path.exists()


class TestHistogram:
  """Test histogram plotting."""

  @pytest.mark.skipif(
    not __import__('openpilot.tools.shadow.visualize', fromlist=['MATPLOTLIB_AVAILABLE']).MATPLOTLIB_AVAILABLE,
    reason="matplotlib not available"
  )
  def test_plot_error_histogram(self, tmp_path: Path):
    """Test error histogram plot."""
    from openpilot.tools.shadow.visualize import plot_error_histogram

    pairs = make_test_pairs(50)
    output_path = tmp_path / "steer_histogram.png"

    plot_error_histogram(pairs, "steer", output_path=output_path)

    assert output_path.exists()


class TestHeatmap:
  """Test heatmap plotting."""

  @pytest.mark.skipif(
    not __import__('openpilot.tools.shadow.visualize', fromlist=['MATPLOTLIB_AVAILABLE']).MATPLOTLIB_AVAILABLE,
    reason="matplotlib not available"
  )
  def test_plot_control_heatmap(self, tmp_path: Path):
    """Test control error heatmap."""
    from openpilot.tools.shadow.visualize import plot_control_heatmap

    pairs = make_test_pairs(100)
    output_path = tmp_path / "control_heatmap.png"

    plot_control_heatmap(pairs, output_path=output_path)

    assert output_path.exists()


class TestCorrelation:
  """Test correlation scatter plotting."""

  @pytest.mark.skipif(
    not __import__('openpilot.tools.shadow.visualize', fromlist=['MATPLOTLIB_AVAILABLE']).MATPLOTLIB_AVAILABLE,
    reason="matplotlib not available"
  )
  def test_plot_correlation_scatter(self, tmp_path: Path):
    """Test correlation scatter plot."""
    from openpilot.tools.shadow.visualize import plot_correlation_scatter

    pairs = make_test_pairs(50)
    output_path = tmp_path / "steer_correlation.png"

    plot_correlation_scatter(pairs, "steer", output_path=output_path)

    assert output_path.exists()


class TestDashboard:
  """Test summary dashboard plotting."""

  @pytest.mark.skipif(
    not __import__('openpilot.tools.shadow.visualize', fromlist=['MATPLOTLIB_AVAILABLE']).MATPLOTLIB_AVAILABLE,
    reason="matplotlib not available"
  )
  def test_plot_summary_dashboard(self, tmp_path: Path):
    """Test summary dashboard generation."""
    from openpilot.tools.shadow.visualize import plot_summary_dashboard

    result = make_test_result()
    report = make_test_report()
    output_path = tmp_path / "dashboard.png"

    plot_summary_dashboard(result, report, output_path=output_path)

    assert output_path.exists()
    # Dashboard should be larger than individual plots
    assert output_path.stat().st_size > 10000


class TestGenerateAll:
  """Test generate all plots function."""

  @pytest.mark.skipif(
    not __import__('openpilot.tools.shadow.visualize', fromlist=['MATPLOTLIB_AVAILABLE']).MATPLOTLIB_AVAILABLE,
    reason="matplotlib not available"
  )
  def test_generate_all_plots(self, tmp_path: Path):
    """Test generating all visualization plots."""
    from openpilot.tools.shadow.visualize import generate_all_plots

    result = make_test_result()
    report = make_test_report()
    output_dir = tmp_path / "plots"

    generated = generate_all_plots(result, report, output_dir)

    assert len(generated) > 0
    assert output_dir.exists()
    for path in generated:
      assert path.exists()

    # Check expected files
    assert (output_dir / "dashboard.png").exists()
    assert (output_dir / "timeseries_steer.png").exists()
    assert (output_dir / "control_heatmap.png").exists()
