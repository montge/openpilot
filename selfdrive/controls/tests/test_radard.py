"""Tests for radard.py - radar data processing and lead detection."""

from collections import deque

import pytest

from openpilot.selfdrive.controls.radard import (
  KalmanParams,
  Track,
  RadarD,
  laplacian_pdf,
  match_vision_to_track,
  get_RadarState_from_vision,
  get_lead,
  RADAR_TO_CAMERA,
  _LEAD_ACCEL_TAU,
)


class TestKalmanParams:
  """Test KalmanParams initialization and interpolation."""

  def test_valid_dt_range(self):
    """Test that valid dt values create params correctly."""
    for dt in [0.02, 0.05, 0.1, 0.15]:
      params = KalmanParams(dt)
      assert len(params.A) == 2
      assert len(params.A[0]) == 2
      assert params.A[0][0] == 1.0
      assert params.A[0][1] == dt
      assert params.A[1][0] == 0.0
      assert params.A[1][1] == 1.0
      assert params.C == [1.0, 0.0]
      assert len(params.K) == 2

  def test_invalid_dt_too_low(self):
    """Test that dt below 0.01 raises assertion."""
    with pytest.raises(AssertionError):
      KalmanParams(0.005)

  def test_invalid_dt_too_high(self):
    """Test that dt above 0.2 raises assertion."""
    with pytest.raises(AssertionError):
      KalmanParams(0.25)

  def test_k_interpolation(self):
    """Test that K values are interpolated correctly."""
    params_low = KalmanParams(0.02)
    params_high = KalmanParams(0.15)
    # K0 should increase with dt
    assert params_low.K[0][0] < params_high.K[0][0]
    # K1 should decrease with dt
    assert params_low.K[1][0] > params_high.K[1][0]


class TestTrack:
  """Test Track class for radar point tracking."""

  @pytest.fixture
  def kalman_params(self):
    return KalmanParams(0.05)

  def test_track_initialization(self, kalman_params):
    """Test Track is initialized correctly."""
    track = Track(identifier=42, v_lead=10.0, kalman_params=kalman_params)
    assert track.identifier == 42
    assert track.cnt == 0
    assert track.kf is not None
    assert track.aLeadTau.x == pytest.approx(_LEAD_ACCEL_TAU)

  def test_track_update_first_call(self, kalman_params):
    """Test first update doesn't call kf.update."""
    track = Track(identifier=1, v_lead=10.0, kalman_params=kalman_params)
    track.update(d_rel=50.0, y_rel=-1.0, v_rel=-5.0, v_lead=15.0, measured=1.0)

    assert track.dRel == 50.0
    assert track.yRel == -1.0
    assert track.vRel == -5.0
    assert track.vLead == 15.0
    assert track.measured == 1.0
    assert track.cnt == 1

  def test_track_update_subsequent_calls(self, kalman_params):
    """Test subsequent updates call kf.update."""
    track = Track(identifier=1, v_lead=10.0, kalman_params=kalman_params)

    # First update
    track.update(d_rel=50.0, y_rel=-1.0, v_rel=-5.0, v_lead=15.0, measured=1.0)

    # Second update - should update Kalman filter
    track.update(d_rel=48.0, y_rel=-0.9, v_rel=-5.1, v_lead=14.9, measured=1.0)
    assert track.cnt == 2
    assert isinstance(track.vLeadK, float)
    assert isinstance(track.aLeadK, float)

  def test_track_aLeadTau_reset_on_low_accel(self, kalman_params):
    """Test aLeadTau resets when acceleration is low."""
    track = Track(identifier=1, v_lead=10.0, kalman_params=kalman_params)
    track.update(d_rel=50.0, y_rel=0.0, v_rel=0.0, v_lead=10.0, measured=1.0)

    # After update with low accel, aLeadTau should be reset to default
    assert track.aLeadTau.x == pytest.approx(_LEAD_ACCEL_TAU)

  def test_get_RadarState(self, kalman_params):
    """Test get_RadarState returns correct dict."""
    track = Track(identifier=1, v_lead=10.0, kalman_params=kalman_params)
    track.update(d_rel=50.0, y_rel=-1.0, v_rel=-5.0, v_lead=15.0, measured=1.0)

    state = track.get_RadarState(model_prob=0.8)

    assert state['dRel'] == 50.0
    assert state['yRel'] == -1.0
    assert state['vRel'] == -5.0
    assert state['vLead'] == 15.0
    assert state['status'] is True
    assert state['radar'] is True
    assert state['radarTrackId'] == 1
    assert state['modelProb'] == 0.8

  def test_get_RadarState_fcw(self, kalman_params):
    """Test FCW is triggered at high model probability."""
    track = Track(identifier=1, v_lead=10.0, kalman_params=kalman_params)
    track.update(d_rel=50.0, y_rel=0.0, v_rel=0.0, v_lead=10.0, measured=1.0)

    # FCW should be True when model_prob > 0.9
    state_high = track.get_RadarState(model_prob=0.95)
    assert state_high['fcw'] is True

    state_low = track.get_RadarState(model_prob=0.85)
    assert state_low['fcw'] is False

  def test_potential_low_speed_lead_true(self, kalman_params):
    """Test low speed lead detection when conditions are met."""
    track = Track(identifier=1, v_lead=2.0, kalman_params=kalman_params)
    track.update(d_rel=10.0, y_rel=0.5, v_rel=-1.0, v_lead=2.0, measured=1.0)

    # v_ego < V_EGO_STATIONARY, yRel < 1, dRel in range
    assert track.potential_low_speed_lead(v_ego=2.0) is True

  def test_potential_low_speed_lead_false_high_speed(self, kalman_params):
    """Test low speed lead not detected at high ego speed."""
    track = Track(identifier=1, v_lead=10.0, kalman_params=kalman_params)
    track.update(d_rel=10.0, y_rel=0.5, v_rel=-5.0, v_lead=10.0, measured=1.0)

    assert track.potential_low_speed_lead(v_ego=20.0) is False

  def test_potential_low_speed_lead_false_too_close(self, kalman_params):
    """Test low speed lead not detected when too close (glitch filter)."""
    track = Track(identifier=1, v_lead=2.0, kalman_params=kalman_params)
    track.update(d_rel=0.5, y_rel=0.0, v_rel=0.0, v_lead=2.0, measured=1.0)

    assert track.potential_low_speed_lead(v_ego=2.0) is False

  def test_potential_low_speed_lead_false_too_far(self, kalman_params):
    """Test low speed lead not detected when too far."""
    track = Track(identifier=1, v_lead=2.0, kalman_params=kalman_params)
    track.update(d_rel=30.0, y_rel=0.0, v_rel=0.0, v_lead=2.0, measured=1.0)

    assert track.potential_low_speed_lead(v_ego=2.0) is False

  def test_potential_low_speed_lead_false_too_lateral(self, kalman_params):
    """Test low speed lead not detected when too far laterally."""
    track = Track(identifier=1, v_lead=2.0, kalman_params=kalman_params)
    track.update(d_rel=10.0, y_rel=2.0, v_rel=0.0, v_lead=2.0, measured=1.0)

    assert track.potential_low_speed_lead(v_ego=2.0) is False

  def test_str_representation(self, kalman_params):
    """Test string representation of track."""
    track = Track(identifier=1, v_lead=10.0, kalman_params=kalman_params)
    track.update(d_rel=50.0, y_rel=-1.0, v_rel=-5.0, v_lead=15.0, measured=1.0)

    s = str(track)
    assert 'x:' in s
    assert 'y:' in s
    assert 'v:' in s
    assert 'a:' in s


class TestLaplacianPdf:
  """Test laplacian_pdf function."""

  def test_laplacian_pdf_at_mean(self):
    """Test PDF is 1 at the mean."""
    result = laplacian_pdf(5.0, 5.0, 1.0)
    assert result == pytest.approx(1.0)

  def test_laplacian_pdf_symmetric(self):
    """Test PDF is symmetric around mean."""
    left = laplacian_pdf(3.0, 5.0, 1.0)
    right = laplacian_pdf(7.0, 5.0, 1.0)
    assert left == pytest.approx(right)

  def test_laplacian_pdf_decreases_with_distance(self):
    """Test PDF decreases as distance from mean increases."""
    close = laplacian_pdf(4.0, 5.0, 1.0)
    far = laplacian_pdf(2.0, 5.0, 1.0)
    assert close > far

  def test_laplacian_pdf_small_b_clamped(self):
    """Test that very small b is clamped to prevent division by zero."""
    result = laplacian_pdf(5.0, 5.0, 0.0)
    assert result == pytest.approx(1.0)

  def test_laplacian_pdf_larger_b_wider_spread(self):
    """Test that larger b gives wider spread (higher probability at distance)."""
    narrow = laplacian_pdf(7.0, 5.0, 0.5)
    wide = laplacian_pdf(7.0, 5.0, 2.0)
    assert narrow < wide


class TestMatchVisionToTrack:
  """Test match_vision_to_track function."""

  @pytest.fixture
  def kalman_params(self):
    return KalmanParams(0.05)

  def _make_lead_msg(self, mocker, x, y, v, x_std=1.0, y_std=1.0, v_std=1.0):
    """Create a mock lead message."""
    lead = mocker.MagicMock()
    lead.x = [x]
    lead.y = [y]
    lead.v = [v]
    lead.xStd = [x_std]
    lead.yStd = [y_std]
    lead.vStd = [v_std]
    return lead

  def test_match_finds_best_track(self, mocker, kalman_params):
    """Test that the best matching track is found."""
    # Create two tracks
    track1 = Track(1, 10.0, kalman_params)
    track1.update(d_rel=50.0 - RADAR_TO_CAMERA, y_rel=0.0, v_rel=-5.0, v_lead=15.0, measured=1.0)

    track2 = Track(2, 10.0, kalman_params)
    track2.update(d_rel=30.0 - RADAR_TO_CAMERA, y_rel=-2.0, v_rel=-3.0, v_lead=17.0, measured=1.0)

    tracks = {1: track1, 2: track2}

    # Lead message matches track1 better
    lead = self._make_lead_msg(mocker, x=50.0, y=0.0, v=15.0)

    result = match_vision_to_track(v_ego=20.0, lead=lead, tracks=tracks)
    assert result.identifier == 1

  def test_match_returns_none_for_insane_distance(self, mocker, kalman_params):
    """Test that None is returned when distance is too different."""
    track = Track(1, 10.0, kalman_params)
    track.update(d_rel=10.0, y_rel=0.0, v_rel=-5.0, v_lead=15.0, measured=1.0)

    tracks = {1: track}
    lead = self._make_lead_msg(mocker, x=100.0, y=0.0, v=15.0)  # Very different distance

    result = match_vision_to_track(v_ego=20.0, lead=lead, tracks=tracks)
    assert result is None

  def test_match_returns_none_for_insane_velocity(self, mocker, kalman_params):
    """Test that None is returned when velocity is too different."""
    track = Track(1, 10.0, kalman_params)
    track.update(d_rel=50.0 - RADAR_TO_CAMERA, y_rel=0.0, v_rel=-20.0, v_lead=0.0, measured=1.0)

    tracks = {1: track}
    lead = self._make_lead_msg(mocker, x=50.0, y=0.0, v=30.0)  # Very different velocity

    result = match_vision_to_track(v_ego=20.0, lead=lead, tracks=tracks)
    assert result is None


class TestGetRadarStateFromVision:
  """Test get_RadarState_from_vision function."""

  def test_creates_correct_state(self, mocker):
    """Test that vision-only state is created correctly."""
    lead = mocker.MagicMock()
    lead.x = [50.0]
    lead.y = [1.0]
    lead.v = [15.0]
    lead.a = [-1.0]
    lead.prob = 0.8

    state = get_RadarState_from_vision(lead, v_ego=20.0, model_v_ego=20.0)

    assert state['dRel'] == pytest.approx(50.0 - RADAR_TO_CAMERA)
    assert state['yRel'] == pytest.approx(-1.0)
    assert state['vRel'] == pytest.approx(-5.0)  # 15 - 20 = -5
    assert state['vLead'] == pytest.approx(15.0)  # 20 + (-5) = 15
    assert state['aLeadK'] == pytest.approx(-1.0)
    assert state['aLeadTau'] == 0.3
    assert state['fcw'] is False
    assert state['modelProb'] == 0.8
    assert state['status'] is True
    assert state['radar'] is False
    assert state['radarTrackId'] == -1


class TestGetLead:
  """Test get_lead function."""

  @pytest.fixture
  def kalman_params(self):
    return KalmanParams(0.05)

  def _make_lead_msg(self, mocker, x, y, v, prob, a=0.0):
    """Create a mock lead message."""
    lead = mocker.MagicMock()
    lead.x = [x]
    lead.y = [y]
    lead.v = [v]
    lead.a = [a]
    lead.prob = prob
    lead.xStd = [1.0]
    lead.yStd = [1.0]
    lead.vStd = [1.0]
    return lead

  def test_no_tracks_returns_vision_only(self, mocker, kalman_params):
    """Test that with no tracks, vision-only lead is returned."""
    lead = self._make_lead_msg(mocker, x=50.0, y=0.0, v=15.0, prob=0.8)
    result = get_lead(v_ego=20.0, ready=True, tracks={}, lead_msg=lead, model_v_ego=20.0)

    assert result['status'] is True
    assert result['radar'] is False

  def test_low_prob_returns_no_lead(self, mocker, kalman_params):
    """Test that low probability returns no lead."""
    lead = self._make_lead_msg(mocker, x=50.0, y=0.0, v=15.0, prob=0.3)
    result = get_lead(v_ego=20.0, ready=True, tracks={}, lead_msg=lead, model_v_ego=20.0)

    assert result['status'] is False

  def test_not_ready_returns_no_lead(self, mocker, kalman_params):
    """Test that not ready returns no lead."""
    lead = self._make_lead_msg(mocker, x=50.0, y=0.0, v=15.0, prob=0.8)
    result = get_lead(v_ego=20.0, ready=False, tracks={}, lead_msg=lead, model_v_ego=20.0)

    assert result['status'] is False

  def test_matching_track_returns_radar_lead(self, mocker, kalman_params):
    """Test that matching track returns radar-based lead."""
    track = Track(1, 15.0, kalman_params)
    track.update(d_rel=50.0 - RADAR_TO_CAMERA, y_rel=0.0, v_rel=-5.0, v_lead=15.0, measured=1.0)
    tracks = {1: track}

    lead = self._make_lead_msg(mocker, x=50.0, y=0.0, v=15.0, prob=0.8)
    result = get_lead(v_ego=20.0, ready=True, tracks=tracks, lead_msg=lead, model_v_ego=20.0)

    assert result['status'] is True
    assert result['radar'] is True
    assert result['radarTrackId'] == 1

  def test_low_speed_override_selects_closer_track(self, mocker, kalman_params):
    """Test that low speed override selects closer track."""
    # Create a track that would be a low speed lead
    track = Track(1, 2.0, kalman_params)
    track.update(d_rel=5.0, y_rel=0.0, v_rel=-1.0, v_lead=2.0, measured=1.0)
    tracks = {1: track}

    lead = self._make_lead_msg(mocker, x=50.0, y=0.0, v=15.0, prob=0.3)  # Low prob, no vision lead
    result = get_lead(v_ego=2.0, ready=True, tracks=tracks, lead_msg=lead, model_v_ego=2.0, low_speed_override=True)

    assert result['status'] is True
    assert result['radarTrackId'] == 1

  def test_low_speed_override_disabled(self, mocker, kalman_params):
    """Test that low speed override can be disabled."""
    track = Track(1, 2.0, kalman_params)
    track.update(d_rel=5.0, y_rel=0.0, v_rel=-1.0, v_lead=2.0, measured=1.0)
    tracks = {1: track}

    lead = self._make_lead_msg(mocker, x=50.0, y=0.0, v=15.0, prob=0.3)
    result = get_lead(v_ego=2.0, ready=True, tracks=tracks, lead_msg=lead, model_v_ego=2.0, low_speed_override=False)

    assert result['status'] is False


class TestRadarD:
  """Test RadarD class."""

  def test_radard_initialization(self):
    """Test RadarD initializes correctly."""
    rd = RadarD(delay=0.1)
    assert rd.current_time == 0.0
    assert rd.v_ego == 0.0
    assert rd.ready is False
    assert len(rd.tracks) == 0
    assert rd.kalman_params is not None

  def test_radard_initialization_with_delay(self):
    """Test RadarD initializes v_ego_hist based on delay."""
    rd = RadarD(delay=0.2)
    # maxlen should be based on delay / DT_MDL + 1
    assert isinstance(rd.v_ego_hist, deque)
    assert len(rd.v_ego_hist) == 1  # Initial value

  def test_radard_tracks_dict_empty(self):
    """Test RadarD starts with empty tracks."""
    rd = RadarD()
    assert rd.tracks == {}

  def test_radard_radar_state_initially_none(self):
    """Test RadarD radar_state is None initially."""
    rd = RadarD()
    assert rd.radar_state is None
    assert rd.radar_state_valid is False
