"""Tests for monitoring/helpers.py - driver monitoring classes and functions."""

from openpilot.selfdrive.monitoring.helpers import (
  DRIVER_MONITOR_SETTINGS,
  DistractedType,
  DriverPose,
  DriverProb,
  DriverBlink,
  face_orientation_from_net,
  DriverMonitoring,
)
from openpilot.common.realtime import DT_DMON


class TestDriverMonitorSettings:
  """Test DRIVER_MONITOR_SETTINGS class."""

  def test_settings_initialization_default(self):
    """Test settings initialize with default device type."""
    settings = DRIVER_MONITOR_SETTINGS(device_type='tici')

    assert settings._DT_DMON == DT_DMON
    assert settings._AWARENESS_TIME == 30.0
    assert settings._DISTRACTED_TIME == 11.0
    assert settings._FACE_THRESHOLD == 0.7
    assert settings._PHONE_THRESH == 0.4

  def test_settings_initialization_mici(self):
    """Test settings with mici device type."""
    settings = DRIVER_MONITOR_SETTINGS(device_type='mici')

    assert settings._PHONE_THRESH == 0.75

  def test_settings_has_all_expected_attributes(self):
    """Test that all expected settings attributes exist."""
    settings = DRIVER_MONITOR_SETTINGS(device_type='tici')

    expected_attrs = [
      '_AWARENESS_TIME',
      '_AWARENESS_PRE_TIME_TILL_TERMINAL',
      '_DISTRACTED_TIME',
      '_DISTRACTED_PRE_TIME_TILL_TERMINAL',
      '_FACE_THRESHOLD',
      '_EYE_THRESHOLD',
      '_SG_THRESHOLD',
      '_BLINK_THRESHOLD',
      '_PHONE_THRESH',
      '_POSE_PITCH_THRESHOLD',
      '_POSE_YAW_THRESHOLD',
      '_PITCH_NATURAL_OFFSET',
      '_YAW_NATURAL_OFFSET',
      '_MAX_TERMINAL_ALERTS',
      '_MAX_TERMINAL_DURATION',
    ]

    for attr in expected_attrs:
      assert hasattr(settings, attr), f"Missing attribute: {attr}"


class TestDistractedType:
  """Test DistractedType constants."""

  def test_not_distracted_is_zero(self):
    """Test NOT_DISTRACTED is 0."""
    assert DistractedType.NOT_DISTRACTED == 0

  def test_distracted_types_are_bit_flags(self):
    """Test distracted types can be combined as bit flags."""
    combined = DistractedType.DISTRACTED_POSE | DistractedType.DISTRACTED_BLINK
    assert combined & DistractedType.DISTRACTED_POSE
    assert combined & DistractedType.DISTRACTED_BLINK
    assert not (combined & DistractedType.DISTRACTED_PHONE)

  def test_distracted_types_are_unique(self):
    """Test distracted types are unique powers of 2."""
    types = [DistractedType.DISTRACTED_POSE, DistractedType.DISTRACTED_BLINK, DistractedType.DISTRACTED_PHONE]
    for i, t in enumerate(types):
      assert t == 1 << i


class TestDriverPose:
  """Test DriverPose class."""

  def setup_method(self):
    self.settings = DRIVER_MONITOR_SETTINGS(device_type='tici')

  def test_driver_pose_initialization(self):
    """Test DriverPose initializes correctly."""
    pose = DriverPose(self.settings)

    assert pose.yaw == 0.0
    assert pose.pitch == 0.0
    assert pose.roll == 0.0
    assert pose.yaw_std == 0.0
    assert pose.pitch_std == 0.0
    assert pose.roll_std == 0.0
    assert not pose.calibrated
    assert pose.low_std
    assert pose.cfactor_pitch == 1.0
    assert pose.cfactor_yaw == 1.0

  def test_driver_pose_has_offseters(self):
    """Test DriverPose has pitch and yaw offseters."""
    pose = DriverPose(self.settings)

    assert pose.pitch_offseter is not None
    assert pose.yaw_offseter is not None


class TestDriverProb:
  """Test DriverProb class."""

  def test_driver_prob_initialization(self):
    """Test DriverProb initializes correctly."""
    raw_priors = (0.05, 0.015, 2)
    prob = DriverProb(raw_priors=raw_priors, max_trackable=100)

    assert prob.prob == 0.0
    assert prob.prob_offseter is not None
    assert not prob.prob_calibrated


class TestDriverBlink:
  """Test DriverBlink class."""

  def test_driver_blink_initialization(self):
    """Test DriverBlink initializes correctly."""
    blink = DriverBlink()

    assert blink.left == 0.0
    assert blink.right == 0.0


class TestFaceOrientationFromNet:
  """Test face_orientation_from_net function."""

  def test_face_orientation_at_center(self):
    """Test face orientation when face is at center."""
    angles_desc = [0.0, 0.0, 0.0]  # pitch, yaw, roll from net
    pos_desc = [0.0, 0.0]  # centered position
    rpy_calib = [0.0, 0.0, 0.0]

    roll, pitch, yaw = face_orientation_from_net(angles_desc, pos_desc, rpy_calib)

    assert abs(roll - 0.0) < 1e-5
    # Pitch and yaw will have focal angle offset for center position
    assert isinstance(pitch, float)
    assert isinstance(yaw, float)

  def test_face_orientation_applies_calibration(self):
    """Test that calibration is applied to pitch and yaw."""
    angles_desc = [0.1, 0.1, 0.1]
    pos_desc = [0.0, 0.0]
    rpy_calib_zero = [0.0, 0.0, 0.0]
    rpy_calib_nonzero = [0.0, 0.1, 0.1]

    _, pitch_zero, yaw_zero = face_orientation_from_net(angles_desc, pos_desc, rpy_calib_zero)
    _, pitch_cal, yaw_cal = face_orientation_from_net(angles_desc, pos_desc, rpy_calib_nonzero)

    # Calibration should affect pitch and yaw
    assert abs(pitch_zero - pitch_cal) > 1e-5
    assert abs(yaw_zero - yaw_cal) > 1e-5

  def test_face_orientation_roll_unaffected_by_calib(self):
    """Test that roll is not affected by calibration."""
    angles_desc = [0.1, 0.2, 0.3]
    pos_desc = [0.0, 0.0]
    rpy_calib = [0.5, 0.5, 0.5]

    roll, _, _ = face_orientation_from_net(angles_desc, pos_desc, rpy_calib)

    assert abs(roll - 0.3) < 1e-5  # roll_net unchanged


class TestDriverMonitoringInit:
  """Test DriverMonitoring initialization."""

  def test_driver_monitoring_initialization(self, mocker):
    """Test DriverMonitoring initializes correctly."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring(rhd_saved=False, always_on=False)

    assert dm.settings is not None
    assert dm.wheelpos is not None
    assert dm.phone is not None
    assert dm.pose is not None
    assert dm.blink is not None
    assert not dm.always_on
    assert dm.awareness == 1.0
    assert not dm.face_detected
    assert not dm.driver_distracted
    assert dm.terminal_alert_cnt == 0

  def test_driver_monitoring_rhd_saved(self, mocker):
    """Test DriverMonitoring with RHD saved setting."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring(rhd_saved=True)

    assert dm.wheel_on_right_default

  def test_driver_monitoring_always_on(self, mocker):
    """Test DriverMonitoring with always_on enabled."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring(always_on=True)

    assert dm.always_on


class TestDriverMonitoringResetAwareness:
  """Test DriverMonitoring._reset_awareness."""

  def test_reset_awareness_sets_all_to_one(self, mocker):
    """Test _reset_awareness sets all awareness values to 1."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = 0.5
    dm.awareness_active = 0.3
    dm.awareness_passive = 0.7

    dm._reset_awareness()

    assert dm.awareness == 1.0
    assert dm.awareness_active == 1.0
    assert dm.awareness_passive == 1.0


class TestDriverMonitoringSetTimers:
  """Test DriverMonitoring._set_timers."""

  def test_set_timers_active_mode(self, mocker):
    """Test _set_timers for active monitoring mode."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm._set_timers(active_monitoring=True)

    assert dm.active_monitoring_mode
    assert dm.step_change > 0

  def test_set_timers_passive_mode(self, mocker):
    """Test _set_timers for passive monitoring mode."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm._set_timers(active_monitoring=False)

    assert not dm.active_monitoring_mode
    # Passive mode has different timing thresholds
    assert dm.threshold_pre < 1.0


class TestDriverMonitoringSetPolicy:
  """Test DriverMonitoring._set_policy."""

  def test_set_policy_adjusts_cfactors(self, mocker):
    """Test _set_policy adjusts pose cfactors."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()

    dm._set_policy(brake_disengage_prob=0.8, car_speed=30.0)

    # cfactors should be updated based on brake disengage probability
    assert isinstance(dm.pose.cfactor_pitch, float)
    assert isinstance(dm.pose.cfactor_yaw, float)


class TestDriverMonitoringGetDistractedTypes:
  """Test DriverMonitoring._get_distracted_types."""

  def test_get_distracted_types_empty_when_not_distracted(self, mocker):
    """Test returns empty when not distracted."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    # Default pose is centered, not distracted
    dm.pose.pitch = 0.0
    dm.pose.yaw = 0.0
    dm.blink.left = 0.0
    dm.blink.right = 0.0
    dm.phone.prob = 0.0

    types = dm._get_distracted_types()

    assert types == []

  def test_get_distracted_types_pose_distracted(self, mocker):
    """Test detects pose distraction."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.pose.pitch = -0.5  # Looking down
    dm.pose.yaw = 0.0

    types = dm._get_distracted_types()

    assert DistractedType.DISTRACTED_POSE in types

  def test_get_distracted_types_blink_distracted(self, mocker):
    """Test detects blink distraction (eyes closed)."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.blink.left = 0.9
    dm.blink.right = 0.9

    types = dm._get_distracted_types()

    assert DistractedType.DISTRACTED_BLINK in types

  def test_get_distracted_types_phone_distracted(self, mocker):
    """Test detects phone distraction."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.phone.prob = 0.8  # High phone probability

    types = dm._get_distracted_types()

    assert DistractedType.DISTRACTED_PHONE in types


class TestDriverMonitoringGetStatePacket:
  """Test DriverMonitoring.get_state_packet."""

  def test_get_state_packet_returns_message(self, mocker):
    """Test get_state_packet returns a valid message."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    packet = dm.get_state_packet(valid=True)

    assert packet is not None
    assert packet.driverMonitoringState is not None

  def test_get_state_packet_contains_expected_fields(self, mocker):
    """Test get_state_packet contains expected fields."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    packet = dm.get_state_packet()

    state = packet.driverMonitoringState
    # Check some key fields exist
    assert state.awarenessStatus is not None
    assert state.faceDetected is not None
    assert state.isDistracted is not None


class TestDriverMonitoringUpdateEvents:
  """Test DriverMonitoring._update_events method."""

  def test_update_events_resets_on_driver_engaged(self, mocker):
    """Test awareness resets when driver is engaged in passive mode."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = 0.5
    dm.active_monitoring_mode = False

    dm._update_events(driver_engaged=True, op_engaged=False, standstill=False, wrong_gear=False, car_speed=20.0)

    assert dm.awareness == 1.0

  def test_update_events_decrements_awareness_when_distracted(self, mocker):
    """Test awareness decreases when driver is distracted."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = 1.0
    dm.face_detected = False  # Not seeing face counts as maybe distracted
    dm.active_monitoring_mode = True

    initial_awareness = dm.awareness
    dm._update_events(driver_engaged=False, op_engaged=True, standstill=False, wrong_gear=False, car_speed=20.0)

    assert dm.awareness < initial_awareness

  def test_update_events_terminal_alert_too_distracted(self, mocker):
    """Test too_distracted flag is set after max terminal alerts."""
    mock_params = mocker.patch('openpilot.selfdrive.monitoring.helpers.Params')
    mock_params.return_value.get_bool.return_value = False
    mock_params.return_value.put_bool_nonblocking = mocker.MagicMock()

    dm = DriverMonitoring()
    dm.terminal_alert_cnt = dm.settings._MAX_TERMINAL_ALERTS

    dm._update_events(driver_engaged=False, op_engaged=True, standstill=False, wrong_gear=False, car_speed=20.0)

    assert dm.too_distracted

  def test_update_events_awareness_recovery(self, mocker):
    """Test awareness recovers when driver is attentive."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = 0.5
    dm.face_detected = True
    dm.pose.low_std = True
    dm.driver_distraction_filter.x = 0.1  # Attentive
    dm.active_monitoring_mode = True

    initial_awareness = dm.awareness
    dm._update_events(driver_engaged=False, op_engaged=True, standstill=False, wrong_gear=False, car_speed=20.0)

    assert dm.awareness > initial_awareness

  def test_update_events_resets_on_disengage(self, mocker):
    """Test awareness resets when openpilot disengages in normal mode."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = 0.3
    dm.always_on = False
    dm.active_monitoring_mode = True

    dm._update_events(driver_engaged=False, op_engaged=False, standstill=False, wrong_gear=False, car_speed=20.0)

    assert dm.awareness == 1.0


class TestDriverMonitoringUpdateStates:
  """Test DriverMonitoring._update_states method."""

  def test_update_states_face_detection(self, mocker):
    """Test face detection updates correctly."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    # Initially no face detected
    assert not dm.face_detected
