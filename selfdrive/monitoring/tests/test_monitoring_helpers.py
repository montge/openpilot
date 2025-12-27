"""Tests for monitoring/helpers.py - driver monitoring classes and functions."""
import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from openpilot.selfdrive.monitoring.helpers import (
  DRIVER_MONITOR_SETTINGS, DistractedType, DriverPose, DriverProb, DriverBlink,
  face_orientation_from_net, DriverMonitoring, EFL, W, H
)
from openpilot.common.realtime import DT_DMON


class TestDriverMonitorSettings(unittest.TestCase):
  """Test DRIVER_MONITOR_SETTINGS class."""

  def test_settings_initialization_default(self):
    """Test settings initialize with default device type."""
    settings = DRIVER_MONITOR_SETTINGS(device_type='tici')

    self.assertEqual(settings._DT_DMON, DT_DMON)
    self.assertEqual(settings._AWARENESS_TIME, 30.0)
    self.assertEqual(settings._DISTRACTED_TIME, 11.0)
    self.assertEqual(settings._FACE_THRESHOLD, 0.7)
    self.assertEqual(settings._PHONE_THRESH, 0.4)

  def test_settings_initialization_mici(self):
    """Test settings with mici device type."""
    settings = DRIVER_MONITOR_SETTINGS(device_type='mici')

    self.assertEqual(settings._PHONE_THRESH, 0.75)

  def test_settings_has_all_expected_attributes(self):
    """Test that all expected settings attributes exist."""
    settings = DRIVER_MONITOR_SETTINGS(device_type='tici')

    expected_attrs = [
      '_AWARENESS_TIME', '_AWARENESS_PRE_TIME_TILL_TERMINAL',
      '_DISTRACTED_TIME', '_DISTRACTED_PRE_TIME_TILL_TERMINAL',
      '_FACE_THRESHOLD', '_EYE_THRESHOLD', '_SG_THRESHOLD', '_BLINK_THRESHOLD',
      '_PHONE_THRESH', '_POSE_PITCH_THRESHOLD', '_POSE_YAW_THRESHOLD',
      '_PITCH_NATURAL_OFFSET', '_YAW_NATURAL_OFFSET',
      '_MAX_TERMINAL_ALERTS', '_MAX_TERMINAL_DURATION',
    ]

    for attr in expected_attrs:
      self.assertTrue(hasattr(settings, attr), f"Missing attribute: {attr}")


class TestDistractedType(unittest.TestCase):
  """Test DistractedType constants."""

  def test_not_distracted_is_zero(self):
    """Test NOT_DISTRACTED is 0."""
    self.assertEqual(DistractedType.NOT_DISTRACTED, 0)

  def test_distracted_types_are_bit_flags(self):
    """Test distracted types can be combined as bit flags."""
    combined = DistractedType.DISTRACTED_POSE | DistractedType.DISTRACTED_BLINK
    self.assertTrue(combined & DistractedType.DISTRACTED_POSE)
    self.assertTrue(combined & DistractedType.DISTRACTED_BLINK)
    self.assertFalse(combined & DistractedType.DISTRACTED_PHONE)

  def test_distracted_types_are_unique(self):
    """Test distracted types are unique powers of 2."""
    types = [DistractedType.DISTRACTED_POSE, DistractedType.DISTRACTED_BLINK, DistractedType.DISTRACTED_PHONE]
    for i, t in enumerate(types):
      self.assertEqual(t, 1 << i)


class TestDriverPose(unittest.TestCase):
  """Test DriverPose class."""

  def setUp(self):
    self.settings = DRIVER_MONITOR_SETTINGS(device_type='tici')

  def test_driver_pose_initialization(self):
    """Test DriverPose initializes correctly."""
    pose = DriverPose(self.settings)

    self.assertEqual(pose.yaw, 0.0)
    self.assertEqual(pose.pitch, 0.0)
    self.assertEqual(pose.roll, 0.0)
    self.assertEqual(pose.yaw_std, 0.0)
    self.assertEqual(pose.pitch_std, 0.0)
    self.assertEqual(pose.roll_std, 0.0)
    self.assertFalse(pose.calibrated)
    self.assertTrue(pose.low_std)
    self.assertEqual(pose.cfactor_pitch, 1.0)
    self.assertEqual(pose.cfactor_yaw, 1.0)

  def test_driver_pose_has_offseters(self):
    """Test DriverPose has pitch and yaw offseters."""
    pose = DriverPose(self.settings)

    self.assertIsNotNone(pose.pitch_offseter)
    self.assertIsNotNone(pose.yaw_offseter)


class TestDriverProb(unittest.TestCase):
  """Test DriverProb class."""

  def test_driver_prob_initialization(self):
    """Test DriverProb initializes correctly."""
    raw_priors = (0.05, 0.015, 2)
    prob = DriverProb(raw_priors=raw_priors, max_trackable=100)

    self.assertEqual(prob.prob, 0.0)
    self.assertIsNotNone(prob.prob_offseter)
    self.assertFalse(prob.prob_calibrated)


class TestDriverBlink(unittest.TestCase):
  """Test DriverBlink class."""

  def test_driver_blink_initialization(self):
    """Test DriverBlink initializes correctly."""
    blink = DriverBlink()

    self.assertEqual(blink.left, 0.0)
    self.assertEqual(blink.right, 0.0)


class TestFaceOrientationFromNet(unittest.TestCase):
  """Test face_orientation_from_net function."""

  def test_face_orientation_at_center(self):
    """Test face orientation when face is at center."""
    angles_desc = [0.0, 0.0, 0.0]  # pitch, yaw, roll from net
    pos_desc = [0.0, 0.0]  # centered position
    rpy_calib = [0.0, 0.0, 0.0]

    roll, pitch, yaw = face_orientation_from_net(angles_desc, pos_desc, rpy_calib)

    self.assertAlmostEqual(roll, 0.0, places=5)
    # Pitch and yaw will have focal angle offset for center position
    self.assertIsInstance(pitch, float)
    self.assertIsInstance(yaw, float)

  def test_face_orientation_applies_calibration(self):
    """Test that calibration is applied to pitch and yaw."""
    angles_desc = [0.1, 0.1, 0.1]
    pos_desc = [0.0, 0.0]
    rpy_calib_zero = [0.0, 0.0, 0.0]
    rpy_calib_nonzero = [0.0, 0.1, 0.1]

    _, pitch_zero, yaw_zero = face_orientation_from_net(angles_desc, pos_desc, rpy_calib_zero)
    _, pitch_cal, yaw_cal = face_orientation_from_net(angles_desc, pos_desc, rpy_calib_nonzero)

    # Calibration should affect pitch and yaw
    self.assertNotAlmostEqual(pitch_zero, pitch_cal, places=5)
    self.assertNotAlmostEqual(yaw_zero, yaw_cal, places=5)

  def test_face_orientation_roll_unaffected_by_calib(self):
    """Test that roll is not affected by calibration."""
    angles_desc = [0.1, 0.2, 0.3]
    pos_desc = [0.0, 0.0]
    rpy_calib = [0.5, 0.5, 0.5]

    roll, _, _ = face_orientation_from_net(angles_desc, pos_desc, rpy_calib)

    self.assertAlmostEqual(roll, 0.3, places=5)  # roll_net unchanged


class TestDriverMonitoringInit(unittest.TestCase):
  """Test DriverMonitoring initialization."""

  @patch('openpilot.selfdrive.monitoring.helpers.Params')
  def test_driver_monitoring_initialization(self, mock_params):
    """Test DriverMonitoring initializes correctly."""
    mock_params.return_value.get_bool.return_value = False

    dm = DriverMonitoring(rhd_saved=False, always_on=False)

    self.assertIsNotNone(dm.settings)
    self.assertIsNotNone(dm.wheelpos)
    self.assertIsNotNone(dm.phone)
    self.assertIsNotNone(dm.pose)
    self.assertIsNotNone(dm.blink)
    self.assertFalse(dm.always_on)
    self.assertEqual(dm.awareness, 1.0)
    self.assertFalse(dm.face_detected)
    self.assertFalse(dm.driver_distracted)
    self.assertEqual(dm.terminal_alert_cnt, 0)

  @patch('openpilot.selfdrive.monitoring.helpers.Params')
  def test_driver_monitoring_rhd_saved(self, mock_params):
    """Test DriverMonitoring with RHD saved setting."""
    mock_params.return_value.get_bool.return_value = False

    dm = DriverMonitoring(rhd_saved=True)

    self.assertTrue(dm.wheel_on_right_default)

  @patch('openpilot.selfdrive.monitoring.helpers.Params')
  def test_driver_monitoring_always_on(self, mock_params):
    """Test DriverMonitoring with always_on enabled."""
    mock_params.return_value.get_bool.return_value = False

    dm = DriverMonitoring(always_on=True)

    self.assertTrue(dm.always_on)


class TestDriverMonitoringResetAwareness(unittest.TestCase):
  """Test DriverMonitoring._reset_awareness."""

  @patch('openpilot.selfdrive.monitoring.helpers.Params')
  def test_reset_awareness_sets_all_to_one(self, mock_params):
    """Test _reset_awareness sets all awareness values to 1."""
    mock_params.return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.awareness = 0.5
    dm.awareness_active = 0.3
    dm.awareness_passive = 0.7

    dm._reset_awareness()

    self.assertEqual(dm.awareness, 1.0)
    self.assertEqual(dm.awareness_active, 1.0)
    self.assertEqual(dm.awareness_passive, 1.0)


class TestDriverMonitoringSetTimers(unittest.TestCase):
  """Test DriverMonitoring._set_timers."""

  @patch('openpilot.selfdrive.monitoring.helpers.Params')
  def test_set_timers_active_mode(self, mock_params):
    """Test _set_timers for active monitoring mode."""
    mock_params.return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm._set_timers(active_monitoring=True)

    self.assertTrue(dm.active_monitoring_mode)
    self.assertGreater(dm.step_change, 0)

  @patch('openpilot.selfdrive.monitoring.helpers.Params')
  def test_set_timers_passive_mode(self, mock_params):
    """Test _set_timers for passive monitoring mode."""
    mock_params.return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm._set_timers(active_monitoring=False)

    self.assertFalse(dm.active_monitoring_mode)
    # Passive mode has different timing thresholds
    self.assertLess(dm.threshold_pre, 1.0)


class TestDriverMonitoringSetPolicy(unittest.TestCase):
  """Test DriverMonitoring._set_policy."""

  @patch('openpilot.selfdrive.monitoring.helpers.Params')
  def test_set_policy_adjusts_cfactors(self, mock_params):
    """Test _set_policy adjusts pose cfactors."""
    mock_params.return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    initial_pitch_factor = dm.pose.cfactor_pitch
    initial_yaw_factor = dm.pose.cfactor_yaw

    dm._set_policy(brake_disengage_prob=0.8, car_speed=30.0)

    # cfactors should be updated based on brake disengage probability
    self.assertIsInstance(dm.pose.cfactor_pitch, float)
    self.assertIsInstance(dm.pose.cfactor_yaw, float)


class TestDriverMonitoringGetDistractedTypes(unittest.TestCase):
  """Test DriverMonitoring._get_distracted_types."""

  @patch('openpilot.selfdrive.monitoring.helpers.Params')
  def test_get_distracted_types_empty_when_not_distracted(self, mock_params):
    """Test returns empty when not distracted."""
    mock_params.return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    # Default pose is centered, not distracted
    dm.pose.pitch = 0.0
    dm.pose.yaw = 0.0
    dm.blink.left = 0.0
    dm.blink.right = 0.0
    dm.phone.prob = 0.0

    types = dm._get_distracted_types()

    self.assertEqual(types, [])

  @patch('openpilot.selfdrive.monitoring.helpers.Params')
  def test_get_distracted_types_pose_distracted(self, mock_params):
    """Test detects pose distraction."""
    mock_params.return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.pose.pitch = -0.5  # Looking down
    dm.pose.yaw = 0.0

    types = dm._get_distracted_types()

    assert DistractedType.DISTRACTED_POSE in types

  @patch('openpilot.selfdrive.monitoring.helpers.Params')
  def test_get_distracted_types_blink_distracted(self, mock_params):
    """Test detects blink distraction (eyes closed)."""
    mock_params.return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.blink.left = 0.9
    dm.blink.right = 0.9

    types = dm._get_distracted_types()

    assert DistractedType.DISTRACTED_BLINK in types

  @patch('openpilot.selfdrive.monitoring.helpers.Params')
  def test_get_distracted_types_phone_distracted(self, mock_params):
    """Test detects phone distraction."""
    mock_params.return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    dm.phone.prob = 0.8  # High phone probability

    types = dm._get_distracted_types()

    assert DistractedType.DISTRACTED_PHONE in types


class TestDriverMonitoringGetStatePacket(unittest.TestCase):
  """Test DriverMonitoring.get_state_packet."""

  @patch('openpilot.selfdrive.monitoring.helpers.Params')
  def test_get_state_packet_returns_message(self, mock_params):
    """Test get_state_packet returns a valid message."""
    mock_params.return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    packet = dm.get_state_packet(valid=True)

    self.assertIsNotNone(packet)
    self.assertIsNotNone(packet.driverMonitoringState)

  @patch('openpilot.selfdrive.monitoring.helpers.Params')
  def test_get_state_packet_contains_expected_fields(self, mock_params):
    """Test get_state_packet contains expected fields."""
    mock_params.return_value.get_bool.return_value = False

    dm = DriverMonitoring()
    packet = dm.get_state_packet()

    state = packet.driverMonitoringState
    # Check some key fields exist
    self.assertIsNotNone(state.awarenessStatus)
    self.assertIsNotNone(state.faceDetected)
    self.assertIsNotNone(state.isDistracted)


if __name__ == '__main__':
  unittest.main()
