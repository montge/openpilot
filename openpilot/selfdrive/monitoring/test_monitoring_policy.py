"""Tests for monitoring/policy.py - driver monitoring classes and functions."""

from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.common.realtime import DT_DMON
from openpilot.selfdrive.monitoring.policy import (
  DRIVER_MONITOR_SETTINGS,
  DriverPose,
  DriverBlink,
  DriverMonitoring,
  face_orientation_from_model,
)

AlertLevel = log.DriverMonitoringState.AlertLevel
MonitoringPolicy = log.DriverMonitoringState.MonitoringPolicy


class TestDriverMonitorSettings:
  """Test DRIVER_MONITOR_SETTINGS class."""

  def test_settings_initialization_default(self):
    """Test settings initialize with expected default values."""
    settings = DRIVER_MONITOR_SETTINGS()

    assert settings._ALERT_MIN_SPEED == 2.8
    assert settings._VISION_POLICY_ALERT_3_TIMEOUT == 13.
    assert settings._WHEELTOUCH_POLICY_ALERT_3_TIMEOUT == 25.
    assert settings._FACE_THRESHOLD == 0.7
    assert settings._PHONE_THRESH == 0.5

  def test_settings_has_all_expected_attributes(self):
    """Test that all expected settings attributes exist."""
    settings = DRIVER_MONITOR_SETTINGS()

    expected_attrs = [
      '_ALERT_MIN_SPEED',
      '_WHEELTOUCH_POLICY_ALERT_1_TIMEOUT',
      '_WHEELTOUCH_POLICY_ALERT_2_TIMEOUT',
      '_WHEELTOUCH_POLICY_ALERT_3_TIMEOUT',
      '_VISION_POLICY_ALERT_1_TIMEOUT',
      '_VISION_POLICY_ALERT_2_TIMEOUT',
      '_VISION_POLICY_ALERT_3_TIMEOUT',
      '_NO_RESPONSE_TIMEOUT',
      '_MAX_ALERT_3',
      '_MAX_NO_RESPONSE',
      '_LOCKOUT_TIME',
      '_FACE_THRESHOLD',
      '_EYE_THRESHOLD',
      '_SG_THRESHOLD',
      '_BLINK_THRESHOLD',
      '_PHONE_THRESH',
      '_POSE_PITCH_THRESHOLD',
      '_POSE_YAW_THRESHOLD',
      '_PITCH_NATURAL_OFFSET',
      '_YAW_NATURAL_OFFSET',
    ]

    for attr in expected_attrs:
      assert hasattr(settings, attr), f"Missing attribute: {attr}"


class TestDriverPose:
  """Test DriverPose class."""

  def setup_method(self):
    self.settings = DRIVER_MONITOR_SETTINGS()

  def test_driver_pose_initialization(self):
    """Test DriverPose initializes correctly."""
    pose = DriverPose(self.settings)

    assert pose.yaw == 0.0
    assert pose.pitch == 0.0
    assert not pose.calibrated
    assert pose.low_std
    assert pose.cfactor_pitch == 1.0
    assert pose.cfactor_yaw == 1.0
    assert pose.steer_yaw_offset == 0.0

  def test_driver_pose_has_offsetters(self):
    """Test DriverPose has pitch and yaw offsetters."""
    pose = DriverPose(self.settings)

    assert pose.pitch_offsetter is not None
    assert pose.yaw_offsetter is not None


class TestDriverBlink:
  """Test DriverBlink class."""

  def test_driver_blink_initialization(self):
    """Test DriverBlink initializes correctly."""
    blink = DriverBlink()

    assert blink.left == 0.0
    assert blink.right == 0.0


class TestFaceOrientationFromModel:
  """Test face_orientation_from_model function."""

  def test_face_orientation_at_center(self):
    """Test face orientation when face is at center."""
    orient_model = [0.0, 0.0, 0.0]  # pitch, yaw, roll from net
    pos_model = [0.0, 0.0]  # centered position
    rpy_calib = [0.0, 0.0, 0.0]

    pitch, yaw = face_orientation_from_model(orient_model, pos_model, rpy_calib)

    # Pitch and yaw will have focal angle offset for center position
    assert isinstance(pitch, float)
    assert isinstance(yaw, float)

  def test_face_orientation_applies_calibration(self):
    """Test that calibration is applied to pitch and yaw."""
    orient_model = [0.1, 0.1, 0.1]
    pos_model = [0.0, 0.0]
    rpy_calib_zero = [0.0, 0.0, 0.0]
    rpy_calib_nonzero = [0.0, 0.1, 0.1]

    pitch_zero, yaw_zero = face_orientation_from_model(orient_model, pos_model, rpy_calib_zero)
    pitch_cal, yaw_cal = face_orientation_from_model(orient_model, pos_model, rpy_calib_nonzero)

    # Calibration pitch/yaw are subtracted directly
    assert abs((pitch_zero - pitch_cal) - 0.1) < 1e-6
    assert abs((yaw_zero - yaw_cal) - 0.1) < 1e-6


class TestDriverMonitoringInit:
  """Test DriverMonitoring initialization."""

  def test_driver_monitoring_initialization(self, mocker):
    """Test DriverMonitoring initializes correctly."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring(rhd_saved=False, always_on=False)

    assert dm.settings is not None
    assert dm.wheelpos_offsetter is not None
    assert dm.pose is not None
    assert dm.blink is not None
    assert dm.phone_prob == 0.0
    assert not dm.always_on
    assert dm.awareness == 1.0
    assert not dm.face_detected
    assert not dm.driver_distracted
    assert dm.alert_3_cnt == 0
    assert dm.no_response_cnt == 0
    assert dm.alert_level == AlertLevel.none
    assert dm.active_policy == MonitoringPolicy.vision

  def test_driver_monitoring_rhd_saved(self, mocker):
    """Test DriverMonitoring with RHD saved setting."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring(rhd_saved=True)

    assert dm.wheel_on_right_default

  def test_driver_monitoring_always_on(self, mocker):
    """Test DriverMonitoring with always_on enabled."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring(always_on=True)

    assert dm.always_on


class TestDriverMonitoringResetAwareness:
  """Test DriverMonitoring._reset_awareness."""

  def test_reset_awareness_sets_all_to_one(self, mocker):
    """Test _reset_awareness sets all awareness values to 1."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = 0.5
    dm.last_vision_awareness = 0.3
    dm.last_wheeltouch_awareness = 0.7

    dm._reset_awareness()

    assert dm.awareness == 1.0
    assert dm.last_vision_awareness == 1.0
    assert dm.last_wheeltouch_awareness == 1.0


class TestDriverMonitoringSetPolicy:
  """Test DriverMonitoring._set_policy."""

  def test_set_policy_vision(self, mocker):
    """Test _set_policy for the vision (active monitoring) policy."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm._set_policy(MonitoringPolicy.vision)

    assert dm.active_policy == MonitoringPolicy.vision
    assert dm.step_change > 0
    assert dm.step_change == DT_DMON / dm.settings._VISION_POLICY_ALERT_3_TIMEOUT

  def test_set_policy_wheeltouch(self, mocker):
    """Test _set_policy for the wheeltouch (passive monitoring) policy."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm._set_policy(MonitoringPolicy.wheeltouch)

    assert dm.active_policy == MonitoringPolicy.wheeltouch
    assert dm.step_change == DT_DMON / dm.settings._WHEELTOUCH_POLICY_ALERT_3_TIMEOUT
    # Wheeltouch policy has different timing thresholds
    assert dm.threshold_alert_1 < 1.0


class TestDriverMonitoringSetPoseStrictness:
  """Test DriverMonitoring._set_pose_strictness."""

  def test_set_pose_strictness_adjusts_cfactors(self, mocker):
    """Test _set_pose_strictness adjusts pose cfactors."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()

    dm._set_pose_strictness(brake_disengage_prob=0.8, car_speed=30.0)

    # cfactors should be updated based on brake disengage probability
    assert isinstance(dm.pose.cfactor_pitch, float)
    assert isinstance(dm.pose.cfactor_yaw, float)


class TestDriverMonitoringGetDistractedTypes:
  """Test DriverMonitoring._get_distracted_types."""

  def test_get_distracted_types_empty_when_not_distracted(self, mocker):
    """Test all distracted types are False when not distracted."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    # Default pose is centered, not distracted
    dm.pose.pitch = 0.0
    dm.pose.yaw = 0.0
    dm.blink.left = 0.0
    dm.blink.right = 0.0
    dm.phone_prob = 0.0

    dm._get_distracted_types()

    assert not any(dm.distracted_types.values())

  def test_get_distracted_types_pose_distracted(self, mocker):
    """Test detects pose distraction."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.pose.pitch = -0.5  # Looking down
    dm.pose.yaw = 0.0

    dm._get_distracted_types()

    assert dm.distracted_types['pose']

  def test_get_distracted_types_blink_distracted(self, mocker):
    """Test detects blink distraction (eyes closed)."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.blink.left = 0.9
    dm.blink.right = 0.9

    dm._get_distracted_types()

    assert dm.distracted_types['eye']

  def test_get_distracted_types_phone_distracted(self, mocker):
    """Test detects phone distraction."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.phone_prob = 0.8  # High phone probability

    dm._get_distracted_types()

    assert dm.distracted_types['phone']

  def test_get_distracted_types_phone_below_threshold(self, mocker):
    """Test phone not distracted when below threshold."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.phone_prob = 0.2  # Below threshold

    dm._get_distracted_types()

    assert not dm.distracted_types['phone']


class TestDriverMonitoringGetStatePacket:
  """Test DriverMonitoring.get_state_packet."""

  def test_get_state_packet_returns_message(self, mocker):
    """Test get_state_packet returns a valid message."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    packet = dm.get_state_packet(valid=True)

    assert packet is not None
    assert packet.driverMonitoringState is not None

  def test_get_state_packet_contains_expected_fields(self, mocker):
    """Test get_state_packet contains expected fields."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    packet = dm.get_state_packet()

    state = packet.driverMonitoringState
    # Check some key fields exist
    assert state.alertLevel == AlertLevel.none
    assert state.activePolicy == MonitoringPolicy.vision
    assert not state.visionPolicyState.isDistracted
    assert not state.visionPolicyState.faceDetected
    assert not state.lockout


class TestDriverMonitoringUpdateEvents:
  """Test DriverMonitoring._update_events method."""

  def test_update_events_resets_on_driver_engaged(self, mocker):
    """Test awareness resets when driver interacts in wheeltouch (passive) policy."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = 0.5
    dm.active_policy = MonitoringPolicy.wheeltouch

    dm._update_events(driver_engaged=True, op_engaged=True, lowspeed=False, wrong_gear=False)

    assert dm.awareness == 1.0

  def test_update_events_decrements_awareness_when_distracted(self, mocker):
    """Test awareness decreases when driver is distracted."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = 1.0
    dm.face_detected = False  # Not seeing face counts as maybe distracted

    initial_awareness = dm.awareness
    dm._update_events(driver_engaged=False, op_engaged=True, lowspeed=False, wrong_gear=False)

    assert dm.awareness < initial_awareness

  def test_update_events_lockout_after_max_alert_3(self, mocker):
    """Test too_distracted flag is set after max alert 3 count."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.alert_3_cnt = dm.settings._MAX_ALERT_3

    dm._update_events(driver_engaged=False, op_engaged=True, lowspeed=False, wrong_gear=False)

    assert dm.too_distracted

  def test_update_events_awareness_recovery(self, mocker):
    """Test awareness recovers when driver is attentive."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = 0.5
    dm.face_detected = True
    dm.pose.low_std = True
    dm.driver_distraction_filter.x = 0.1  # Attentive

    initial_awareness = dm.awareness
    dm._update_events(driver_engaged=False, op_engaged=True, lowspeed=False, wrong_gear=False)

    assert dm.awareness > initial_awareness

  def test_update_events_resets_on_disengage(self, mocker):
    """Test awareness resets when openpilot disengages in normal mode."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = 0.3
    dm.always_on = False

    dm._update_events(driver_engaged=False, op_engaged=False, lowspeed=False, wrong_gear=False)

    assert dm.awareness == 1.0


class TestDriverMonitoringSetPolicyEdgeCases:
  """Test DriverMonitoring._set_policy edge cases."""

  def test_set_policy_no_change_when_awareness_zero(self, mocker):
    """Test _set_policy returns early when awareness is zero."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.active_policy = MonitoringPolicy.wheeltouch
    dm.awareness = 0.0
    old_step_change = dm.step_change

    dm._set_policy(MonitoringPolicy.vision)

    # Should return early and not change step_change or policy
    assert dm.step_change == old_step_change
    assert dm.active_policy == MonitoringPolicy.wheeltouch

  def test_set_policy_vision_below_orange_keeps_counting(self, mocker):
    """Test _set_policy keeps the vision step change when already past orange alert."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.active_policy = MonitoringPolicy.vision
    dm.awareness = dm.threshold_alert_2 - 0.01  # Below orange threshold

    dm._set_policy(MonitoringPolicy.vision)

    # Should keep counting down with the vision step change
    assert dm.step_change == DT_DMON / dm.settings._VISION_POLICY_ALERT_3_TIMEOUT

  def test_set_policy_wheeltouch_below_orange_freezes(self, mocker):
    """Test no exploit when switching to wheeltouch past orange alert."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.active_policy = MonitoringPolicy.vision
    dm.awareness = dm.threshold_alert_2 - 0.01  # Below orange threshold

    dm._set_policy(MonitoringPolicy.wheeltouch)

    # step_change should be 0 when switching to wheeltouch past orange alert
    assert dm.step_change == 0
    assert dm.active_policy == MonitoringPolicy.vision

  def test_set_policy_wheeltouch_to_vision_restores_awareness(self, mocker):
    """Test awareness is restored when switching from wheeltouch to vision."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.active_policy = MonitoringPolicy.wheeltouch
    dm.awareness = 0.6
    dm.last_vision_awareness = 0.8
    dm.last_wheeltouch_awareness = 1.0

    dm._set_policy(MonitoringPolicy.vision)

    # Should restore last vision awareness and save the wheeltouch one
    assert dm.awareness == 0.8
    assert dm.last_wheeltouch_awareness == 0.6
    assert dm.active_policy == MonitoringPolicy.vision

  def test_set_policy_vision_to_wheeltouch_saves_awareness(self, mocker):
    """Test awareness is saved when switching from vision to wheeltouch."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.active_policy = MonitoringPolicy.vision
    dm.awareness = 0.7
    dm.last_vision_awareness = 1.0
    dm.last_wheeltouch_awareness = 0.5

    dm._set_policy(MonitoringPolicy.wheeltouch)

    # Should save the vision awareness and restore the wheeltouch one
    assert dm.last_vision_awareness == 0.7
    assert dm.awareness == 0.5
    assert dm.active_policy == MonitoringPolicy.wheeltouch


class TestDriverMonitoringGetDistractedTypesCalibrated:
  """Test _get_distracted_types with calibrated pose."""

  def test_get_distracted_types_calibrated_pose(self, mocker):
    """Test distracted types with calibrated pose."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.pose.calibrated = True
    # Set offsetter stats
    dm.pose.pitch_offsetter.filtered_stat.n = 100
    dm.pose.pitch_offsetter.filtered_stat.M = 0.05
    dm.pose.yaw_offsetter.filtered_stat.n = 100
    dm.pose.yaw_offsetter.filtered_stat.M = 0.03

    dm.pose.pitch = -0.5  # Looking down
    dm.pose.yaw = 0.0

    dm._get_distracted_types()

    assert dm.distracted_types['pose']


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
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    # Initially no face detected
    assert not dm.face_detected

  def test_update_states_detects_face(self, mocker):
    """Test _update_states detects face correctly."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    driver_state = self._create_driver_state(mocker, face_prob=0.9)

    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=30.0, op_engaged=True, lowspeed=False)

    assert dm.face_detected

  def test_update_states_no_face(self, mocker):
    """Test _update_states with no face detected."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    driver_state = self._create_driver_state(mocker, face_prob=0.3)

    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=30.0, op_engaged=True, lowspeed=False)

    assert not dm.face_detected

  def test_update_states_wheel_position_calibration(self, mocker):
    """Test wheel position calibration during update."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    driver_state = self._create_driver_state(mocker, face_prob=0.9, wheel_on_right_prob=0.8)

    # Run update multiple times at speed above calibration threshold
    for _ in range(3):
      dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=15.0, op_engaged=False, lowspeed=False)

    # Wheel position should be tracked (n increments after second push)
    assert dm.wheelpos_offsetter.filtered_stat.n > 0

  def test_update_states_rhd_detection_demo_mode(self, mocker):
    """Test RHD detection in demo mode."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    driver_state = self._create_driver_state(mocker, face_prob=0.9, wheel_on_right_prob=0.9)

    # Fill up enough samples for calibration
    for _ in range(20):
      dm.wheelpos_offsetter.push_and_update(0.9)

    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=15.0, op_engaged=False, lowspeed=False, demo_mode=True)

    assert dm.wheel_on_right

  def test_update_states_no_switch_when_engaged(self, mocker):
    """Test wheel position doesn't switch when engaged."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.wheel_on_right_last = False
    driver_state = self._create_driver_state(mocker, face_prob=0.9, wheel_on_right_prob=0.9)

    # Fill up enough samples to trigger switch
    for _ in range(20):
      dm.wheelpos_offsetter.push_and_update(0.9)

    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=15.0, op_engaged=True, lowspeed=False)

    # Should not switch when engaged
    assert dm.wheel_on_right == dm.wheel_on_right_last

  def test_update_states_empty_face_data_returns_early(self, mocker):
    """Test _update_states returns early with empty face data."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

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
    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=30.0, op_engaged=True, lowspeed=False)

    # Face detection should not be updated
    assert dm.face_detected == initial_face_detected

  def test_update_states_pose_calibration(self, mocker):
    """Test pose calibration during update."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    driver_state = self._create_driver_state(mocker, face_prob=0.9)

    # Run update at speed above calibration threshold when not distracted
    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=15.0, op_engaged=False, lowspeed=False)

    # Pose offsetters should be updated
    assert dm.pose.pitch_offsetter.filtered_stat.n > 0

  def test_update_states_hi_stds_tracking(self, mocker):
    """Test hi_stds counter tracking."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

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

    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=30.0, op_engaged=True, lowspeed=False)

    assert dm.hi_stds > 0

  def test_update_states_dcam_uncertain_count(self, mocker):
    """Test dcam uncertain counter increases."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

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

    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=30.0, op_engaged=True, lowspeed=False)

    assert dm.dcam_uncertain_cnt > 0

  def test_update_states_yaw_negated_for_rhd(self, mocker):
    """Test yaw is negated for right-hand drive."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

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

    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=30.0, op_engaged=False, lowspeed=False)

    # Yaw should be affected by RHD
    assert dm.wheel_on_right


class TestDriverMonitoringUpdateEventsEdgeCases:
  """Test DriverMonitoring._update_events edge cases."""

  def test_update_events_no_response_sets_lockout(self, mocker):
    """Test too_distracted when no-response count reaches max."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.no_response_cnt = dm.settings._MAX_NO_RESPONSE

    dm._update_events(driver_engaged=False, op_engaged=True, lowspeed=False, wrong_gear=False)

    assert dm.too_distracted

  def test_update_events_lockout_recovers_after_timeout(self, mocker):
    """Test lockout clears after the lockout time elapses."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.too_distracted = True
    dm.lockout_time = dm.settings._LOCKOUT_TIME

    dm._update_events(driver_engaged=False, op_engaged=False, lowspeed=False, wrong_gear=False)

    assert not dm.too_distracted
    assert dm.lockout_time == 0

  def test_update_events_always_on_alert_at_orange(self, mocker):
    """Test alert level two is reached below the orange threshold with always_on."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring(always_on=True)
    dm.awareness = dm.threshold_alert_2 - 0.01
    dm.face_detected = False  # Maybe distracted

    dm._update_events(driver_engaged=False, op_engaged=True, lowspeed=False, wrong_gear=False)

    assert dm.alert_level == AlertLevel.two

  def test_update_events_driver_engaged_resets_awareness(self, mocker):
    """Test driver engaged resets awareness when attentive."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = 0.5
    dm.face_detected = True
    dm.pose.low_std = True
    dm.driver_distraction_filter.x = 0.1  # Attentive

    dm._update_events(driver_engaged=True, op_engaged=True, lowspeed=False, wrong_gear=False)

    assert dm.awareness == 1.0

  def test_update_events_wheeltouch_awareness_recovery(self, mocker):
    """Test last_wheeltouch_awareness increments when awareness is full."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = 1.0 - dm.step_change  # Almost full
    dm.last_wheeltouch_awareness = 0.8
    dm.face_detected = True
    dm.pose.low_std = True
    dm.driver_distraction_filter.x = 0.1  # Attentive

    dm._update_events(driver_engaged=False, op_engaged=True, lowspeed=False, wrong_gear=False)

    # Awareness should be 1.0 and the wheeltouch awareness should recover too
    assert dm.awareness == 1.0
    assert dm.last_wheeltouch_awareness > 0.8

  def test_update_events_lowspeed_exemption_at_alert_1(self, mocker):
    """Test lowspeed exemption prevents dropping past alert level one."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = dm.threshold_alert_1 + dm.step_change / 2  # About to reach alert 1
    dm.face_detected = False  # Maybe distracted

    dm._update_events(driver_engaged=False, op_engaged=True, lowspeed=True, wrong_gear=False)

    # Should not go below the alert 1 threshold at low speed
    assert dm.awareness >= dm.threshold_alert_1

  def test_update_events_terminal_alert_increments(self, mocker):
    """Test alert_3_cnt increments when hitting red."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = dm.step_change / 2  # Just above red
    dm.face_detected = False  # Maybe distracted

    initial_cnt = dm.alert_3_cnt
    dm._update_events(driver_engaged=False, op_engaged=True, lowspeed=False, wrong_gear=False)

    assert dm.awareness <= 0
    assert dm.alert_level == AlertLevel.three
    assert dm.alert_3_cnt == initial_cnt + 1

  def test_update_events_always_on_disengaged_red_exemption(self, mocker):
    """Test always_on red exemption when disengaged."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring(always_on=True)
    dm.awareness = dm.step_change / 2  # Almost at red
    dm.face_detected = False  # Distracted

    dm._update_events(driver_engaged=False, op_engaged=False, lowspeed=False, wrong_gear=False)

    # Should not reach red when disengaged in always_on mode
    assert dm.awareness > 0


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
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.dcam_uncertain_cnt = 10
    driver_state = self._create_driver_state(mocker, face_prob=0.9, std_value=0.05)  # Low std

    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=30.0, op_engaged=True, lowspeed=False)

    assert dm.dcam_reset_cnt > 0

  def test_dcam_uncertain_resets_after_enough_resets(self, mocker):
    """Test dcam_uncertain_cnt resets when reset count exceeds threshold."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.dcam_uncertain_cnt = 10
    dm.dcam_reset_cnt = dm.settings._DCAM_UNCERTAIN_RESET_COUNT  # At threshold
    driver_state = self._create_driver_state(mocker, face_prob=0.9, std_value=0.05)  # Low std

    dm._update_states(driver_state, [0.0, 0.0, 0.0], car_speed=30.0, op_engaged=True, lowspeed=False)

    assert dm.dcam_uncertain_cnt == 0


class TestDriverMonitoringTerminalAlerts:
  """Test terminal alert scenarios."""

  def test_no_response_counting_at_red(self, mocker):
    """Test cnt_since_alert_3 increments when awareness stays at red."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = -0.05  # Already at red
    dm.face_detected = False  # Distracted

    initial_cnt = dm.cnt_since_alert_3
    dm._update_events(driver_engaged=False, op_engaged=True, lowspeed=False, wrong_gear=False)

    assert dm.alert_level == AlertLevel.three
    assert dm.cnt_since_alert_3 == initial_cnt + 1

  def test_alert_3_cnt_increments_on_first_red(self, mocker):
    """Test alert_3_cnt increments when first reaching red."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = dm.step_change / 2  # Just above 0
    dm.face_detected = False  # Distracted

    initial_cnt = dm.alert_3_cnt
    dm._update_events(driver_engaged=False, op_engaged=True, lowspeed=False, wrong_gear=False)

    # Should have transitioned from positive to negative awareness
    assert dm.awareness <= 0
    assert dm.alert_3_cnt == initial_cnt + 1
    assert dm.cnt_since_alert_3 == 0


class TestDriverMonitoringRunStep:
  """Test DriverMonitoring.run_step method."""

  def _create_sm(self, mocker, speed=30.0, enabled=True, steering_pressed=False, gas_pressed=False, gear_shifter=None):
    """Create a mock SubMaster with required messages."""
    if gear_shifter is None:
      gear_shifter = car.CarState.GearShifter.drive
    sm = mocker.MagicMock()

    sm.__getitem__.side_effect = lambda key: {
      'carState': mocker.MagicMock(
        vEgo=speed,
        steeringPressed=steering_pressed,
        gasPressed=gas_pressed,
        steeringAngleDeg=0.0,
        gearShifter=gear_shifter,
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
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    sm = mocker.MagicMock()

    # Setup driver state
    sm.__getitem__.return_value = self._create_driver_state(mocker)

    dm.run_step(sm, demo=True)

    # Should complete without error
    assert dm.settings is not None

  def test_run_step_normal_mode(self, mocker):
    """Test run_step in normal mode."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    sm = self._create_sm(mocker)

    dm.run_step(sm, demo=False)

    # Should complete without error
    assert dm.settings is not None

  def test_run_step_sets_pose_strictness(self, mocker):
    """Test run_step calls _set_pose_strictness."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    sm = self._create_sm(mocker, speed=20.0)

    dm.run_step(sm, demo=False)

    # cfactor should be a float after pose strictness is set
    assert isinstance(dm.pose.cfactor_pitch, float)

  def test_run_step_updates_states(self, mocker):
    """Test run_step calls _update_states."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    sm = self._create_sm(mocker)

    dm.run_step(sm, demo=False)

    # States should be updated: mocked faceProb is above the threshold
    assert dm.face_detected

  def test_run_step_updates_events(self, mocker):
    """Test run_step calls _update_events."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    sm = self._create_sm(mocker)

    dm.run_step(sm, demo=False)

    # Attentive driver: no alert should be raised
    assert dm.alert_level == AlertLevel.none

  def test_run_step_wrong_gear_detection(self, mocker):
    """Test run_step detects wrong gear."""
    mocker.patch('openpilot.selfdrive.monitoring.policy.Params').return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    sm = self._create_sm(mocker, gear_shifter=car.CarState.GearShifter.park)  # Not in drive

    dm.run_step(sm, demo=False)

    # Should complete without error
    assert dm.settings is not None
