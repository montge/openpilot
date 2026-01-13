"""Unit tests for Stone Soup adapters."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from openpilot.tools.stonesoup.adapters import (
  LeadData,
  PoseState,
  RadarDetection,
  STONESOUP_AVAILABLE,
)


class TestRadarDetection:
  """Tests for RadarDetection dataclass."""

  def test_creation(self):
    """Test basic RadarDetection creation."""
    det = RadarDetection(
      d_rel=50.0,
      v_rel=-5.0,
      a_rel=0.0,
      y_rel=0.5,
      timestamp=100.0,
    )

    assert det.d_rel == 50.0
    assert det.v_rel == -5.0
    assert det.y_rel == 0.5


class TestLeadData:
  """Tests for LeadData dataclass."""

  def test_creation(self):
    """Test basic LeadData creation."""
    lead = LeadData(
      d_rel=30.0,
      v_rel=-2.0,
      a_rel=-0.5,
      y_rel=0.0,
      d_std=1.0,
      v_std=0.5,
      prob=0.95,
    )

    assert lead.d_rel == 30.0
    assert lead.prob == 0.95


class TestPoseState:
  """Tests for PoseState dataclass."""

  def test_creation(self):
    """Test basic PoseState creation."""
    pose = PoseState(
      x=100.0,
      y=50.0,
      z=0.0,
      vx=10.0,
      vy=0.0,
      vz=0.0,
      roll=0.0,
      pitch=0.0,
      yaw=0.5,
      timestamp=1000.0,
    )

    assert pose.x == 100.0
    assert pose.vx == 10.0
    assert pose.yaw == 0.5


# Skip Stone Soup dependent tests if not installed
pytestmark_stonesoup = pytest.mark.skipif(
  not STONESOUP_AVAILABLE,
  reason="Stone Soup not installed",
)


@pytestmark_stonesoup
class TestOpenpilotAdapterWithStonesoup:
  """Tests requiring Stone Soup installation."""

  def test_adapter_creation(self):
    """Test adapter can be created when Stone Soup is available."""
    from openpilot.tools.stonesoup.adapters import OpenpilotAdapter

    adapter = OpenpilotAdapter()
    assert adapter is not None

  def test_radar_to_stonesoup(self):
    """Test radar detection conversion."""
    from openpilot.tools.stonesoup.adapters import OpenpilotAdapter

    adapter = OpenpilotAdapter()

    det = RadarDetection(
      d_rel=50.0,
      v_rel=-5.0,
      a_rel=-0.5,
      y_rel=0.5,
      timestamp=100.0,
    )

    ss_det = adapter.radar_to_stonesoup(det)

    # Verify state vector
    assert float(ss_det.state_vector[0, 0]) == 50.0  # d_rel
    assert float(ss_det.state_vector[1, 0]) == -5.0  # v_rel
    assert float(ss_det.state_vector[2, 0]) == 0.5  # y_rel

    # Verify metadata
    assert ss_det.metadata["a_rel"] == -0.5
    assert ss_det.metadata["source"] == "radar"

  def test_stonesoup_to_lead(self):
    """Test Stone Soup state to LeadData conversion."""
    from stonesoup.types.state import GaussianState, StateVector

    from openpilot.tools.stonesoup.adapters import OpenpilotAdapter

    adapter = OpenpilotAdapter()

    # Create a Stone Soup state
    state_vector = StateVector([[40.0], [-3.0], [0.2], [-0.1]])
    covar = np.diag([1.0, 0.5, 0.1, 0.1])
    ss_state = GaussianState(
      state_vector=state_vector,
      covar=covar,
      timestamp=datetime.now(),
    )

    lead = adapter.stonesoup_to_lead(ss_state)

    assert lead.d_rel == 40.0
    assert lead.v_rel == -3.0
    assert lead.y_rel == 0.2
    assert lead.d_std == 1.0
    assert abs(lead.v_std - np.sqrt(0.5)) < 0.001

  def test_lead_to_groundtruth(self):
    """Test LeadData to GroundTruthState conversion."""
    from openpilot.tools.stonesoup.adapters import OpenpilotAdapter

    adapter = OpenpilotAdapter()

    lead = LeadData(
      d_rel=25.0,
      v_rel=-1.5,
      a_rel=0.0,
      y_rel=0.0,
      d_std=0.5,
      v_std=0.2,
      prob=0.99,
    )

    gt_state = adapter.lead_to_groundtruth(lead, datetime.now())

    assert float(gt_state.state_vector[0, 0]) == 25.0
    assert float(gt_state.state_vector[1, 0]) == -1.5

  def test_pose_roundtrip(self):
    """Test pose state conversion roundtrip."""
    from openpilot.tools.stonesoup.adapters import OpenpilotAdapter

    adapter = OpenpilotAdapter()

    original = PoseState(
      x=100.0,
      y=50.0,
      z=0.0,
      vx=10.0,
      vy=0.5,
      vz=0.0,
      roll=0.0,
      pitch=0.0,
      yaw=0.5,
      timestamp=1000.0,
    )

    ss_state = adapter.pose_to_stonesoup(original)
    restored = adapter.stonesoup_to_pose(ss_state, original.timestamp)

    assert restored.x == original.x
    assert restored.vx == original.vx
    assert restored.y == original.y
    assert restored.vy == original.vy
    assert restored.yaw == original.yaw

  def test_create_constant_velocity_model(self):
    """Test constant velocity transition model creation."""
    from openpilot.tools.stonesoup.adapters import create_constant_velocity_model

    model = create_constant_velocity_model(noise_diffusion=0.1)
    assert model is not None
    assert model.ndim_state == 4  # x, vx, y, vy

  def test_create_position_measurement_model(self):
    """Test position measurement model creation."""
    from openpilot.tools.stonesoup.adapters import create_position_measurement_model

    model = create_position_measurement_model()
    assert model is not None
    assert model.ndim_state == 4
    assert model.ndim_meas == 2


class TestAdapterWithoutStonesoup:
  """Tests that work without Stone Soup."""

  def test_stonesoup_availability_flag(self):
    """Test that STONESOUP_AVAILABLE reflects actual availability."""
    # This test always passes - it just documents the behavior
    assert isinstance(STONESOUP_AVAILABLE, bool)

  def test_adapter_raises_without_stonesoup(self):
    """Test adapter raises ImportError when Stone Soup not available."""
    if STONESOUP_AVAILABLE:
      pytest.skip("Stone Soup is installed")

    from openpilot.tools.stonesoup.adapters import OpenpilotAdapter

    with pytest.raises(ImportError, match="Stone Soup is required"):
      OpenpilotAdapter()
