from datetime import timedelta

import numpy as np
from stonesoup.types.array import StateVector
from stonesoup.types.state import GaussianState

from openpilot.tools.stonesoup.adapters import OpenpilotAdapter


class TestOpenpilotAdapter:
  def test_mono_time_to_datetime(self):
    adapter = OpenpilotAdapter()
    dt = adapter.mono_time_to_datetime(3600.0)  # 1 hour
    expected = adapter.epoch + timedelta(hours=1)
    assert dt == expected

  def test_radar_point_to_detection(self):
    adapter = OpenpilotAdapter()
    det = adapter.radar_point_to_detection(
      d_rel=50.0, y_rel=-1.5, v_rel=-5.0,
      timestamp=100.0, track_id=42
    )
    assert det.state_vector[0] == 50.0   # dRel
    assert det.state_vector[1] == -1.5   # yRel
    assert det.state_vector[2] == -5.0   # vRel
    assert det.metadata['track_id'] == 42

  def test_radar_point_default_metadata(self):
    adapter = OpenpilotAdapter()
    det = adapter.radar_point_to_detection(
      d_rel=30.0, y_rel=0.0, v_rel=-10.0,
      timestamp=50.0
    )
    assert det.metadata['track_id'] is None
    assert det.metadata['measured'] is True

  def test_gaussian_state_to_lead_dict(self):
    adapter = OpenpilotAdapter()
    state = GaussianState(
      state_vector=StateVector([50.0, -1.5, -5.0]),
      covar=np.eye(3),
      timestamp=adapter.mono_time_to_datetime(100.0),
    )
    lead = adapter.gaussian_state_to_lead_dict(state, v_ego=20.0)

    assert lead['dRel'] == 50.0
    assert lead['yRel'] == -1.5
    assert lead['vRel'] == -5.0
    assert lead['vLead'] == 15.0  # v_ego + v_rel
    assert lead['vLeadK'] == 15.0
    assert lead['aLeadK'] == 0.0  # no acceleration in 3-state
    assert lead['status'] is True
    assert lead['radar'] is True

  def test_gaussian_state_to_lead_dict_with_accel(self):
    adapter = OpenpilotAdapter()
    state = GaussianState(
      state_vector=StateVector([50.0, -1.5, -5.0, -2.0]),  # 4-state with accel
      covar=np.eye(4),
      timestamp=adapter.mono_time_to_datetime(100.0),
    )
    lead = adapter.gaussian_state_to_lead_dict(state, v_ego=20.0, track_id=7)

    assert lead['aLeadK'] == -2.0
    assert lead['radarTrackId'] == 7

  def test_pose_to_gaussian_state(self):
    adapter = OpenpilotAdapter()
    gs = adapter.pose_to_gaussian_state(
      x=100.0, y=5.0, yaw=0.1,
      vx=20.0, vy=0.5, yaw_rate=0.01,
      timestamp=50.0
    )

    assert gs.state_vector[0] == 100.0  # x
    assert gs.state_vector[1] == 5.0    # y
    assert gs.state_vector[2] == 0.1    # yaw
    assert gs.state_vector[3] == 20.0   # vx
    assert gs.state_vector[4] == 0.5    # vy
    assert gs.state_vector[5] == 0.01   # yaw_rate
    assert gs.covar.shape == (6, 6)

  def test_pose_to_gaussian_state_custom_covariance(self):
    adapter = OpenpilotAdapter()
    custom_cov = np.eye(6) * 2.0
    gs = adapter.pose_to_gaussian_state(
      x=0.0, y=0.0, yaw=0.0,
      vx=0.0, vy=0.0, yaw_rate=0.0,
      timestamp=0.0,
      covariance=custom_cov
    )
    np.testing.assert_array_equal(gs.covar, custom_cov)

  def test_pose_roundtrip(self):
    adapter = OpenpilotAdapter()
    gs = adapter.pose_to_gaussian_state(
      x=100.0, y=5.0, yaw=0.1,
      vx=20.0, vy=0.5, yaw_rate=0.01,
      timestamp=50.0
    )
    pose = adapter.gaussian_state_to_pose_dict(gs)

    assert pose['x'] == 100.0
    assert pose['y'] == 5.0
    assert pose['yaw'] == 0.1
    assert pose['vx'] == 20.0
    assert pose['vy'] == 0.5
    assert pose['yaw_rate'] == 0.01

  def test_custom_epoch(self):
    from datetime import datetime
    custom_epoch = datetime(2020, 1, 1)
    adapter = OpenpilotAdapter(epoch=custom_epoch)

    assert adapter.epoch == custom_epoch
    dt = adapter.mono_time_to_datetime(0.0)
    assert dt == custom_epoch
