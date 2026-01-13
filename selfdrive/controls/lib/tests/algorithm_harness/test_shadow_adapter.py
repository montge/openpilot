"""Tests for shadow log adapter."""

from __future__ import annotations

import pytest

try:
  import pandas as pd  # noqa: F401

  PANDAS_AVAILABLE = True
except ImportError:
  PANDAS_AVAILABLE = False

from openpilot.tools.shadow.comparison_logger import FrameData

# Skip all tests if pandas not available
pytestmark = pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")


class TestShadowFramesToDataframe:
  """Tests for converting shadow frames to DataFrame."""

  def test_basic_conversion(self):
    """Test basic frame to DataFrame conversion."""
    from openpilot.selfdrive.controls.lib.tests.algorithm_harness.shadow_adapter import (
      shadow_frames_to_dataframe,
    )

    frames = [
      FrameData(
        frame_id=1,
        timestamp_mono=100.0,
        controls={"steer": 0.5, "accel": 1.0},
        model_outputs={"desired_curvature": 0.01},
        state={"lat_active": True},
      ),
      FrameData(
        frame_id=2,
        timestamp_mono=100.01,
        controls={"steer": 0.6, "accel": 1.1},
        model_outputs={"desired_curvature": 0.02},
        state={"lat_active": True},
      ),
    ]

    df = shadow_frames_to_dataframe(frames)

    assert len(df) == 2
    assert "timestamp_ns" in df.columns
    assert "frame_id" in df.columns
    assert "gt_steer_cmd" in df.columns
    assert df.iloc[0]["gt_steer_cmd"] == 0.5
    assert df.iloc[1]["gt_steer_cmd"] == 0.6

  def test_empty_frames(self):
    """Test conversion with empty frame list."""
    from openpilot.selfdrive.controls.lib.tests.algorithm_harness.shadow_adapter import (
      shadow_frames_to_dataframe,
    )

    df = shadow_frames_to_dataframe([])
    assert len(df) == 0


class TestCreateShadowScenario:
  """Tests for creating scenarios from shadow logs."""

  def test_create_scenario(self):
    """Test scenario creation from frames."""
    from openpilot.selfdrive.controls.lib.tests.algorithm_harness.shadow_adapter import (
      create_shadow_scenario,
    )

    frames = [
      FrameData(
        frame_id=i,
        timestamp_mono=100.0 + i * 0.01,
        controls={"steer": float(i) * 0.1},
      )
      for i in range(10)
    ]

    df, metadata = create_shadow_scenario(
      frames,
      name="test_scenario",
      description="Test shadow scenario",
    )

    assert len(df) == 10
    assert metadata.name == "test_scenario"
    assert metadata.source == "shadow_device"
    assert metadata.num_steps == 10

  def test_create_scenario_empty_raises(self):
    """Test that empty frames raises error."""
    from openpilot.selfdrive.controls.lib.tests.algorithm_harness.shadow_adapter import (
      create_shadow_scenario,
    )

    with pytest.raises(ValueError, match="No frames provided"):
      create_shadow_scenario([], name="empty")
