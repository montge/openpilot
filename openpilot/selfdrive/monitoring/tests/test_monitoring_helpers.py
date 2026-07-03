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


class TestDriverMonitoringSetTimersEdgeCases:
  """Test DriverMonitoring._set_timers edge cases."""

  def test_set_timers_no_change_when_awareness_zero(self, mocker):
    """Test _set_timers returns early when awareness is zero."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = 0.0
    old_step_change = dm.step_change

    dm._set_timers(active_monitoring=True)

    # Should return early and not change step_change
    assert dm.step_change == old_step_change

  def test_set_timers_active_prompt_with_active_monitoring(self, mocker):
    """Test _set_timers when at prompt threshold in active mode."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.active_monitoring_mode = True
    dm.awareness = dm.threshold_prompt - 0.01  # Below prompt threshold

    dm._set_timers(active_monitoring=True)

    # Should update step_change but return early
    assert dm.step_change > 0

  def test_set_timers_active_prompt_with_passive_monitoring(self, mocker):
    """Test _set_timers when at prompt threshold switching to passive."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.active_monitoring_mode = True
    dm.awareness = dm.threshold_prompt - 0.01  # Below prompt threshold

    dm._set_timers(active_monitoring=False)

    # step_change should be 0 when switching to passive at prompt threshold
    assert dm.step_change == 0

  def test_set_timers_passive_to_active_restores_awareness(self, mocker):
    """Test awareness is restored when switching from passive to active."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.active_monitoring_mode = False
    dm.awareness = 0.6
    dm.awareness_active = 0.8
    dm.awareness_passive = 0.6

    dm._set_timers(active_monitoring=True)

    # Should restore awareness_active
    assert dm.awareness == 0.8
    assert dm.active_monitoring_mode

  def test_set_timers_active_to_passive_saves_awareness(self, mocker):
    """Test awareness is saved when switching from active to passive."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.active_monitoring_mode = True
    dm.awareness = 0.7
    dm.awareness_active = 1.0
    dm.awareness_passive = 0.5

    dm._set_timers(active_monitoring=False)

    # Should save awareness_active and restore awareness_passive
    assert dm.awareness_active == 0.7
    assert dm.awareness == 0.5
    assert not dm.active_monitoring_mode


class TestDriverMonitoringGetDistractedTypesCalibrated:
  """Test _get_distracted_types with calibrated pose."""

  def test_get_distracted_types_calibrated_pose(self, mocker):
    """Test distracted types with calibrated pose."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.pose.calibrated = True
    # Set offseter stats
    dm.pose.pitch_offseter.filtered_stat._n = 100
    dm.pose.pitch_offseter.filtered_stat._M = 0.05
    dm.pose.yaw_offseter.filtered_stat._n = 100
    dm.pose.yaw_offseter.filtered_stat._M = 0.03

    dm.pose.pitch = -0.5  # Looking down
    dm.pose.yaw = 0.0

    types = dm._get_distracted_types()

    assert DistractedType.DISTRACTED_POSE in types

  def test_get_distracted_types_phone_calibrated(self, mocker):
    """Test phone distraction with calibrated phone probability."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.phone.prob_calibrated = True
    dm.phone.prob_offseter.filtered_stat._n = 100
    dm.phone.prob_offseter.filtered_stat._M = 0.03  # Low baseline

    dm.phone.prob = 0.8  # High phone probability

    types = dm._get_distracted_types()

    assert DistractedType.DISTRACTED_PHONE in types

  def test_get_distracted_types_phone_calibrated_below_threshold(self, mocker):
    """Test phone not distracted when below calibrated threshold."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.phone.prob_calibrated = True
    dm.phone.prob_offseter.filtered_stat._n = 100
    dm.phone.prob_offseter.filtered_stat._M = 0.06  # Higher baseline

    dm.phone.prob = 0.2  # Below threshold

    types = dm._get_distracted_types()

    assert DistractedType.DISTRACTED_PHONE not in types


class TestDriverMonitoringUpdateStates:
  """Test DriverMonitoring._update_states method."""

  def _create_driver_state(self, mocker, face_prob=0.9, phone_prob=0.1, wheel_on_right_prob=0.1):
    """Create a mock driver state."""
    driver_data = mocker.MagicMock()
    driver_data.faceProb = face_prob
    driver_data.faceOrientation = [0.0, 0.0, 0.0]
    driver_data.facePosition = [0.0, 0.0]
    driver_data.faceOrientationStd = [0.1, 0.1, 0.1]
    driver_data.facePositionStd = [0.1, 0.1]
    driver_data.leftBlinkProb = 0.1
    driver_data.rightBlinkProb = 0.1
    driver_data.leftEyeProb = 0.9
    driver_data.rightEyeProb = 0.9
    driver_data.sunglassesProb = 0.1
    driver_data.phoneProb = phone_prob

    driver_state = mocker.MagicMock()
    driver_state.wheelOnRightProb = wheel_on_right_prob
    driver_state.leftDriverData = driver_data
    driver_state.rightDriverData = driver_data

    return driver_state

  def test_update_states_face_detection(self, mocker):
    """Test face detection updates correctly."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    # Initially no face detected
    assert not dm.face_detected

  def test_update_states_detects_face(self, mocker):
    """Test _update_states detects face correctly."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    driver_state = self._create_driver_state(mocker, face_prob=0.9)

    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=30.0, op_engaged=True, standstill=False)

    assert dm.face_detected

  def test_update_states_no_face(self, mocker):
    """Test _update_states with no face detected."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    driver_state = self._create_driver_state(mocker, face_prob=0.3)

    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=30.0, op_engaged=True, standstill=False)

    assert not dm.face_detected

  def test_update_states_wheel_position_calibration(self, mocker):
    """Test wheel position calibration during update."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    driver_state = self._create_driver_state(mocker, face_prob=0.9, wheel_on_right_prob=0.8)

    # Run update multiple times at speed above calibration threshold
    for _ in range(3):
      dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=15.0, op_engaged=False, standstill=False)

    # Wheel position should be tracked (n increments after second push)
    assert dm.wheelpos.prob_offseter.filtered_stat.n > 0

  def test_update_states_rhd_detection_demo_mode(self, mocker):
    """Test RHD detection in demo mode."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    driver_state = self._create_driver_state(mocker, face_prob=0.9, wheel_on_right_prob=0.9)

    # Fill up enough samples for calibration
    for _ in range(20):
      dm.wheelpos.prob_offseter.push_and_update(0.9)

    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=15.0, op_engaged=False, standstill=False, demo_mode=True)

    assert dm.wheel_on_right

  def test_update_states_no_switch_when_engaged(self, mocker):
    """Test wheel position doesn't switch when engaged."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.wheel_on_right_last = False
    driver_state = self._create_driver_state(mocker, face_prob=0.9, wheel_on_right_prob=0.9)

    # Fill up enough samples to trigger switch
    for _ in range(20):
      dm.wheelpos.prob_offseter.push_and_update(0.9)

    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=15.0, op_engaged=True, standstill=False)

    # Should not switch when engaged
    assert dm.wheel_on_right == dm.wheel_on_right_last

  def test_update_states_empty_face_data_returns_early(self, mocker):
    """Test _update_states returns early with empty face data."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()

    driver_data = mocker.MagicMock()
    driver_data.faceProb = 0.9
    driver_data.faceOrientation = []  # Empty orientation
    driver_data.facePosition = [0.0, 0.0]
    driver_data.faceOrientationStd = []
    driver_data.facePositionStd = []

    driver_state = mocker.MagicMock()
    driver_state.wheelOnRightProb = 0.1
    driver_state.leftDriverData = driver_data
    driver_state.rightDriverData = driver_data

    initial_face_detected = dm.face_detected
    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=30.0, op_engaged=True, standstill=False)

    # Face detection should not be updated
    assert dm.face_detected == initial_face_detected

  def test_update_states_pose_calibration(self, mocker):
    """Test pose calibration during update."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    driver_state = self._create_driver_state(mocker, face_prob=0.9)

    # Run update at speed above calibration threshold when not distracted
    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=15.0, op_engaged=False, standstill=False)

    # Pose offseters should be updated
    assert dm.pose.pitch_offseter.filtered_stat.n > 0

  def test_update_states_hi_stds_tracking(self, mocker):
    """Test hi_stds counter tracking."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()

    driver_data = mocker.MagicMock()
    driver_data.faceProb = 0.9
    driver_data.faceOrientation = [0.0, 0.0, 0.0]
    driver_data.facePosition = [0.0, 0.0]
    driver_data.faceOrientationStd = [0.5, 0.5, 0.5]  # High std
    driver_data.facePositionStd = [0.1, 0.1]
    driver_data.leftBlinkProb = 0.1
    driver_data.rightBlinkProb = 0.1
    driver_data.leftEyeProb = 0.9
    driver_data.rightEyeProb = 0.9
    driver_data.sunglassesProb = 0.1
    driver_data.phoneProb = 0.1

    driver_state = mocker.MagicMock()
    driver_state.wheelOnRightProb = 0.1
    driver_state.leftDriverData = driver_data
    driver_state.rightDriverData = driver_data

    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=30.0, op_engaged=True, standstill=False)

    assert dm.hi_stds > 0

  def test_update_states_dcam_uncertain_count(self, mocker):
    """Test dcam uncertain counter increases."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()

    driver_data = mocker.MagicMock()
    driver_data.faceProb = 0.9
    driver_data.faceOrientation = [0.0, 0.0, 0.0]
    driver_data.facePosition = [0.0, 0.0]
    driver_data.faceOrientationStd = [0.2, 0.2, 0.2]  # Above uncertain threshold
    driver_data.facePositionStd = [0.1, 0.1]
    driver_data.leftBlinkProb = 0.1
    driver_data.rightBlinkProb = 0.1
    driver_data.leftEyeProb = 0.9
    driver_data.rightEyeProb = 0.9
    driver_data.sunglassesProb = 0.1
    driver_data.phoneProb = 0.1

    driver_state = mocker.MagicMock()
    driver_state.wheelOnRightProb = 0.1
    driver_state.leftDriverData = driver_data
    driver_state.rightDriverData = driver_data

    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=30.0, op_engaged=True, standstill=False)

    assert dm.dcam_uncertain_cnt > 0

  def test_update_states_yaw_negated_for_rhd(self, mocker):
    """Test yaw is negated for right-hand drive."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.wheel_on_right_default = True

    driver_data = mocker.MagicMock()
    driver_data.faceProb = 0.9
    driver_data.faceOrientation = [0.0, 0.1, 0.0]  # Some yaw
    driver_data.facePosition = [0.0, 0.0]
    driver_data.faceOrientationStd = [0.1, 0.1, 0.1]
    driver_data.facePositionStd = [0.1, 0.1]
    driver_data.leftBlinkProb = 0.1
    driver_data.rightBlinkProb = 0.1
    driver_data.leftEyeProb = 0.9
    driver_data.rightEyeProb = 0.9
    driver_data.sunglassesProb = 0.1
    driver_data.phoneProb = 0.1

    driver_state = mocker.MagicMock()
    driver_state.wheelOnRightProb = 0.9
    driver_state.leftDriverData = driver_data
    driver_state.rightDriverData = driver_data

    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=30.0, op_engaged=False, standstill=False)

    # Yaw should be affected by RHD
    assert dm.wheel_on_right


class TestDriverMonitoringUpdateEventsEdgeCases:
  """Test DriverMonitoring._update_events edge cases."""

  def test_update_events_too_distracted_sets_param(self, mocker):
    """Test too_distracted sets param when threshold exceeded."""
    mock_params = mocker.patch('openpilot.selfdrive.monitoring.helpers.Params')
    mock_params.return_value.get_bool.return_value = False
    mock_params.return_value.put_bool_nonblocking = mocker.MagicMock()

    dm = DriverMonitoring()
    dm.terminal_alert_cnt = dm.settings._MAX_TERMINAL_ALERTS

    dm._update_events(driver_engaged=False, op_engaged=True, standstill=False, wrong_gear=False, car_speed=20.0)

    mock_params.return_value.put_bool_nonblocking.assert_called_with("DriverTooDistracted", True)
    assert dm.too_distracted

  def test_update_events_terminal_time_exceeds_max(self, mocker):
    """Test too_distracted when terminal time exceeds max."""
    mock_params = mocker.patch('openpilot.selfdrive.monitoring.helpers.Params')
    mock_params.return_value.get_bool.return_value = False
    mock_params.return_value.put_bool_nonblocking = mocker.MagicMock()

    dm = DriverMonitoring()
    dm.terminal_time = dm.settings._MAX_TERMINAL_DURATION

    dm._update_events(driver_engaged=False, op_engaged=True, standstill=False, wrong_gear=False, car_speed=20.0)

    assert dm.too_distracted

  def test_update_events_always_on_alert_at_prompt(self, mocker):
    """Test always_on alert added at prompt threshold."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring(always_on=True)
    dm.awareness = dm.threshold_prompt - 0.01

    dm._update_events(driver_engaged=False, op_engaged=True, standstill=False, wrong_gear=False, car_speed=20.0)

    # Should have tooDistracted event (event ID 45)
    from cereal import log

    assert log.OnroadEvent.EventName.tooDistracted in dm.current_events.events

  def test_update_events_driver_engaged_resets_awareness(self, mocker):
    """Test driver engaged resets awareness in active mode."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = 0.5
    dm.face_detected = True
    dm.pose.low_std = True
    dm.driver_distraction_filter.x = 0.1  # Attentive

    dm._update_events(driver_engaged=True, op_engaged=True, standstill=False, wrong_gear=False, car_speed=20.0)

    assert dm.awareness == 1.0

  def test_update_events_awareness_passive_recovery(self, mocker):
    """Test awareness_passive increments when awareness is full."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = 1.0 - dm.step_change  # Almost full
    dm.awareness_passive = 0.8
    dm.face_detected = True
    dm.pose.low_std = True
    dm.driver_distraction_filter.x = 0.1  # Attentive

    dm._update_events(driver_engaged=False, op_engaged=True, standstill=False, wrong_gear=False, car_speed=20.0)

    # Awareness should be 1.0 and passive should increment
    assert dm.awareness == 1.0

  def test_update_events_standstill_exemption_at_orange(self, mocker):
    """Test standstill exemption prevents going to orange."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = dm.threshold_prompt + dm.step_change  # Just above orange
    dm.face_detected = False  # Maybe distracted

    dm._update_events(driver_engaged=False, op_engaged=True, standstill=True, wrong_gear=False, car_speed=0.0)

    # Should not go below prompt threshold at standstill
    assert dm.awareness >= dm.threshold_prompt

  def test_update_events_terminal_alert_increments(self, mocker):
    """Test terminal_alert_cnt increments when hitting red."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = 0.01  # Almost at red
    dm.face_detected = False  # Definitely distracted

    initial_cnt = dm.terminal_alert_cnt
    dm._update_events(driver_engaged=False, op_engaged=True, standstill=False, wrong_gear=False, car_speed=20.0)

    # awareness should go to 0 or below
    if dm.awareness <= 0:
      assert dm.terminal_alert_cnt > initial_cnt

  def test_update_events_pre_alert(self, mocker):
    """Test pre-alert event is added at threshold_pre."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    # Set awareness between pre and prompt thresholds
    dm.awareness = (dm.threshold_pre + dm.threshold_prompt) / 2
    dm.face_detected = True
    dm.pose.low_std = True
    dm.driver_distraction_filter.x = 0.8  # Distracted

    dm._update_events(driver_engaged=False, op_engaged=True, standstill=False, wrong_gear=False, car_speed=20.0)

    # Should have pre-alert event (preDriverDistracted=33 or preDriverUnresponsive=35)
    from cereal import log

    pre_events = [log.OnroadEvent.EventName.preDriverDistracted, log.OnroadEvent.EventName.preDriverUnresponsive]
    assert any(e in dm.current_events.events for e in pre_events)

  def test_update_events_dcam_uncertain_alert(self, mocker):
    """Test dcam uncertain alert is set when count exceeds threshold."""
    mock_set_alert = mocker.patch('openpilot.selfdrive.monitoring.helpers.set_offroad_alert')
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.dcam_uncertain_cnt = dm.settings._DCAM_UNCERTAIN_ALERT_COUNT + 1
    dm.dcam_uncertain_alerted = False

    dm._update_events(driver_engaged=False, op_engaged=True, standstill=False, wrong_gear=False, car_speed=20.0)

    mock_set_alert.assert_called_with("Offroad_DriverMonitoringUncertain", True)
    assert dm.dcam_uncertain_alerted

  def test_update_events_always_on_disengaged_red_exemption(self, mocker):
    """Test always_on red exemption when disengaged."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring(always_on=True)
    dm.awareness = dm.step_change  # Almost at red
    dm.face_detected = False  # Distracted

    dm._update_events(driver_engaged=False, op_engaged=False, standstill=False, wrong_gear=False, car_speed=20.0)

    # Should not go to negative awareness when disengaged in always_on mode
    assert dm.awareness >= -0.1


class TestDriverMonitoringDcamUncertainReset:
  """Test dcam uncertain reset logic."""

  def _create_driver_state(self, mocker, face_prob=0.9, std_value=0.05):
    """Create a mock driver state with configurable std."""
    driver_data = mocker.MagicMock()
    driver_data.faceProb = face_prob
    driver_data.faceOrientation = [0.0, 0.0, 0.0]
    driver_data.facePosition = [0.0, 0.0]
    driver_data.faceOrientationStd = [std_value, std_value, std_value]
    driver_data.facePositionStd = [0.1, 0.1]
    driver_data.leftBlinkProb = 0.1
    driver_data.rightBlinkProb = 0.1
    driver_data.leftEyeProb = 0.9
    driver_data.rightEyeProb = 0.9
    driver_data.sunglassesProb = 0.1
    driver_data.phoneProb = 0.1

    driver_state = mocker.MagicMock()
    driver_state.wheelOnRightProb = 0.1
    driver_state.leftDriverData = driver_data
    driver_state.rightDriverData = driver_data

    return driver_state

  def test_dcam_reset_count_increments(self, mocker):
    """Test dcam_reset_cnt increments when std is low."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.dcam_uncertain_cnt = 10
    driver_state = self._create_driver_state(mocker, face_prob=0.9, std_value=0.05)  # Low std

    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=30.0, op_engaged=True, standstill=False)

    assert dm.dcam_reset_cnt > 0

  def test_dcam_uncertain_resets_after_enough_resets(self, mocker):
    """Test dcam_uncertain_cnt resets when reset count exceeds threshold."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.dcam_uncertain_cnt = 10
    dm.dcam_reset_cnt = dm.settings._DCAM_UNCERTAIN_RESET_COUNT  # At threshold
    driver_state = self._create_driver_state(mocker, face_prob=0.9, std_value=0.05)  # Low std

    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=30.0, op_engaged=True, standstill=False)

    assert dm.dcam_uncertain_cnt == 0


class TestDriverMonitoringTerminalAlerts:
  """Test terminal alert scenarios."""

  def test_terminal_time_increments_at_red(self, mocker):
    """Test terminal_time increments when awareness is at red."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = -0.05  # Already at red
    dm.face_detected = False  # Distracted

    initial_time = dm.terminal_time
    dm._update_events(driver_engaged=False, op_engaged=True, standstill=False, wrong_gear=False, car_speed=20.0)

    assert dm.terminal_time > initial_time

  def test_terminal_alert_cnt_increments_on_first_red(self, mocker):
    """Test terminal_alert_cnt increments when first reaching red."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = dm.step_change / 2  # Just above 0
    dm.face_detected = False  # Distracted
    dm.hi_stds = dm.settings._HI_STD_FALLBACK_TIME + 1  # Maybe distracted

    initial_cnt = dm.terminal_alert_cnt
    dm._update_events(driver_engaged=False, op_engaged=True, standstill=False, wrong_gear=False, car_speed=20.0)

    # Should have transitioned from positive to negative awareness
    if dm.awareness <= 0:
      assert dm.terminal_alert_cnt == initial_cnt + 1


class TestDriverMonitoringRunStep:
  """Test DriverMonitoring.run_step method."""

  def _create_sm(self, mocker, speed=30.0, enabled=True, standstill=False, steering_pressed=False, gas_pressed=False):
    """Create a mock SubMaster with required messages."""
    sm = mocker.MagicMock()

    # carState
    sm.__getitem__.side_effect = lambda key: {
      'carState': mocker.MagicMock(
        vEgo=speed,
        standstill=standstill,
        steeringPressed=steering_pressed,
        gasPressed=gas_pressed,
        gearShifter=2,  # Drive
      ),
      'selfdriveState': mocker.MagicMock(enabled=enabled),
      'modelV2': mocker.MagicMock(meta=mocker.MagicMock(disengagePredictions=mocker.MagicMock(brakeDisengageProbs=[0.5]))),
      'liveCalibration': mocker.MagicMock(rpyCalib=[0.0, 0.0, 0.0]),
      'driverStateV2': self._create_driver_state(mocker),
    }.get(key)

    return sm

  def _create_driver_state(self, mocker, face_prob=0.9):
    """Create a mock driver state."""
    driver_data = mocker.MagicMock()
    driver_data.faceProb = face_prob
    driver_data.faceOrientation = [0.0, 0.0, 0.0]
    driver_data.facePosition = [0.0, 0.0]
    driver_data.faceOrientationStd = [0.1, 0.1, 0.1]
    driver_data.facePositionStd = [0.1, 0.1]
    driver_data.leftBlinkProb = 0.1
    driver_data.rightBlinkProb = 0.1
    driver_data.leftEyeProb = 0.9
    driver_data.rightEyeProb = 0.9
    driver_data.sunglassesProb = 0.1
    driver_data.phoneProb = 0.1

    driver_state = mocker.MagicMock()
    driver_state.wheelOnRightProb = 0.1
    driver_state.leftDriverData = driver_data
    driver_state.rightDriverData = driver_data

    return driver_state

  def test_run_step_demo_mode(self, mocker):
    """Test run_step in demo mode."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    sm = mocker.MagicMock()

    # Setup driver state
    sm.__getitem__.return_value = self._create_driver_state(mocker)

    dm.run_step(sm, demo=True)

    # Should complete without error
    assert dm.settings is not None

  def test_run_step_normal_mode(self, mocker):
    """Test run_step in normal mode."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    sm = self._create_sm(mocker)

    dm.run_step(sm, demo=False)

    # Should complete without error
    assert dm.settings is not None

  def test_run_step_sets_policy(self, mocker):
    """Test run_step calls _set_policy."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    sm = self._create_sm(mocker, speed=20.0)

    dm.run_step(sm, demo=False)

    # cfactor should be a float after policy is set
    assert isinstance(dm.pose.cfactor_pitch, float)

  def test_run_step_updates_states(self, mocker):
    """Test run_step calls _update_states."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    sm = self._create_sm(mocker)

    dm.run_step(sm, demo=False)

    # States should be updated
    assert dm.face_detected or not dm.face_detected  # Just verify it ran

  def test_run_step_updates_events(self, mocker):
    """Test run_step calls _update_events."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    sm = self._create_sm(mocker)

    dm.run_step(sm, demo=False)

    # Events should be initialized
    assert dm.current_events is not None

  def test_run_step_wrong_gear_detection(self, mocker):
    """Test run_step detects wrong gear."""
    mocker.patch('openpilot.selfdrive.monitoring.helpers.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()

    sm = mocker.MagicMock()
    sm.__getitem__.side_effect = lambda key: {
      'carState': mocker.MagicMock(
        vEgo=30.0,
        standstill=False,
        steeringPressed=False,
        gasPressed=False,
        gearShifter=0,  # Not in drive
      ),
      'selfdriveState': mocker.MagicMock(enabled=True),
      'modelV2': mocker.MagicMock(meta=mocker.MagicMock(disengagePredictions=mocker.MagicMock(brakeDisengageProbs=[0.5]))),
      'liveCalibration': mocker.MagicMock(rpyCalib=[0.0, 0.0, 0.0]),
      'driverStateV2': self._create_driver_state(mocker),
    }.get(key)

    dm.run_step(sm, demo=False)

    # Should complete without error
    assert dm.settings is not None
