"""Tests for selfdrive/controls/lib/longitudinal_planner.py - longitudinal MPC planner."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  LongitudinalPlanner, get_max_accel, get_coast_accel, limit_accel_in_turns,
  A_CRUISE_MAX_VALS, A_CRUISE_MAX_BP, ALLOW_THROTTLE_THRESHOLD,
  MIN_ALLOW_THROTTLE_SPEED, LON_MPC_STEP,
)
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET


def create_mock_cp(steer_ratio=15.0, wheelbase=2.7, openpilot_long=True,
                   actuator_delay=0.2, v_ego_stopping=0.5):
  """Create a mock CarParams for testing."""
  CP = MagicMock()
  CP.steerRatio = steer_ratio
  CP.wheelbase = wheelbase
  CP.openpilotLongitudinalControl = openpilot_long
  CP.longitudinalActuatorDelay = actuator_delay
  CP.vEgoStopping = v_ego_stopping
  return CP


def create_mock_model_msg(valid=True, throttle_prob=1.0, desired_accel=0.0, should_stop=False):
  """Create a mock modelV2 message."""
  model = MagicMock()

  if valid:
    # Create valid position/velocity/acceleration arrays
    model.position.x = list(np.linspace(0, 100, ModelConstants.IDX_N))
    model.velocity.x = list(np.linspace(10, 15, ModelConstants.IDX_N))
    model.acceleration.x = list(np.zeros(ModelConstants.IDX_N))
  else:
    model.position.x = []
    model.velocity.x = []
    model.acceleration.x = []

  model.meta.disengagePredictions.gasPressProbs = [0.0, throttle_prob]
  # Set scalar values explicitly (not MagicMock) to avoid array comparison issues
  model.action = MagicMock()
  model.action.desiredAcceleration = float(desired_accel)
  model.action.shouldStop = bool(should_stop)

  return model


def create_mock_sm(v_ego=15.0, v_cruise=100.0, experimental_mode=False,
                   long_control_off=False, enabled=True, force_decel=False,
                   standstill=False, a_ego=0.0, steering_angle=0.0,
                   angle_offset=0.0, orientation_ned=None, model_msg=None):
  """Create a mock SubMaster for testing."""
  from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState

  # Create individual message mocks
  car_state = MagicMock()
  car_state.vEgo = v_ego
  car_state.vCruise = v_cruise
  car_state.aEgo = a_ego
  car_state.standstill = standstill
  car_state.steeringAngleDeg = steering_angle

  selfdrive_state = MagicMock()
  selfdrive_state.experimentalMode = experimental_mode
  selfdrive_state.enabled = enabled
  selfdrive_state.personality = 0

  controls_state = MagicMock()
  controls_state.longControlState = LongCtrlState.off if long_control_off else LongCtrlState.pid
  controls_state.forceDecel = force_decel

  live_params = MagicMock()
  live_params.angleOffsetDeg = angle_offset

  car_control = MagicMock()
  if orientation_ned is not None:
    car_control.orientationNED = orientation_ned
  else:
    car_control.orientationNED = [0.0, 0.0, 0.0]

  if model_msg is None:
    model_msg = create_mock_model_msg()

  radar_state = MagicMock()
  radar_state.leadOne.status = False

  # Create sm mock with proper __getitem__ behavior
  messages = {
    'carState': car_state,
    'selfdriveState': selfdrive_state,
    'controlsState': controls_state,
    'liveParameters': live_params,
    'carControl': car_control,
    'modelV2': model_msg,
    'radarState': radar_state,
  }

  sm = MagicMock()
  sm.__getitem__ = lambda self, key: messages[key]
  sm.logMonoTime = {'modelV2': 1000000}
  sm.all_checks = MagicMock(return_value=True)

  return sm


class TestGetMaxAccel(unittest.TestCase):
  """Test get_max_accel function."""

  def test_zero_speed(self):
    """Test max accel at zero speed."""
    result = get_max_accel(0.0)
    self.assertAlmostEqual(result, A_CRUISE_MAX_VALS[0])

  def test_high_speed(self):
    """Test max accel at high speed."""
    result = get_max_accel(40.0)
    self.assertAlmostEqual(result, A_CRUISE_MAX_VALS[-1])

  def test_interpolation(self):
    """Test accel interpolates between breakpoints."""
    result = get_max_accel(17.5)  # Between 10 and 25
    # Should be between 1.2 and 0.8
    self.assertLess(result, 1.2)
    self.assertGreater(result, 0.8)

  def test_above_max_speed(self):
    """Test accel clamps at speeds above max breakpoint."""
    result = get_max_accel(100.0)
    self.assertAlmostEqual(result, A_CRUISE_MAX_VALS[-1])

  def test_negative_speed(self):
    """Test accel at negative speed (edge case)."""
    result = get_max_accel(-5.0)
    self.assertAlmostEqual(result, A_CRUISE_MAX_VALS[0])


class TestGetCoastAccel(unittest.TestCase):
  """Test get_coast_accel function."""

  def test_zero_pitch(self):
    """Test coast accel at zero pitch (flat road)."""
    result = get_coast_accel(0.0)
    self.assertAlmostEqual(result, -0.3, places=5)

  def test_uphill_pitch(self):
    """Test coast accel going uphill (positive pitch)."""
    result = get_coast_accel(0.1)  # ~5.7 degrees uphill
    # Should be more negative (decelerating)
    self.assertLess(result, -0.3)

  def test_downhill_pitch(self):
    """Test coast accel going downhill (negative pitch)."""
    result = get_coast_accel(-0.1)  # ~5.7 degrees downhill
    # Should be less negative (maybe even positive)
    self.assertGreater(result, -0.3)

  def test_steep_uphill(self):
    """Test coast accel on steep uphill."""
    result = get_coast_accel(0.2)  # ~11.5 degrees
    self.assertLess(result, -1.0)


class TestLimitAccelInTurns(unittest.TestCase):
  """Test limit_accel_in_turns function."""

  def test_straight_no_limit(self):
    """Test no limiting when driving straight."""
    CP = create_mock_cp()
    a_target = [ACCEL_MIN, 1.5]

    result = limit_accel_in_turns(20.0, 0.0, a_target, CP)

    self.assertEqual(result[0], ACCEL_MIN)
    self.assertAlmostEqual(result[1], 1.5, places=3)

  def test_turn_limits_accel(self):
    """Test accel is limited during turns."""
    CP = create_mock_cp()
    a_target = [ACCEL_MIN, 2.0]

    # Large steering angle should limit accel
    result = limit_accel_in_turns(25.0, 30.0, a_target, CP)

    self.assertEqual(result[0], ACCEL_MIN)
    self.assertLess(result[1], 2.0)

  def test_min_accel_unchanged(self):
    """Test minimum accel is never changed."""
    CP = create_mock_cp()
    a_target = [-2.0, 1.5]

    result = limit_accel_in_turns(20.0, 45.0, a_target, CP)

    self.assertEqual(result[0], -2.0)

  def test_low_speed_less_limiting(self):
    """Test less limiting at low speeds."""
    CP = create_mock_cp()
    a_target = [ACCEL_MIN, 1.5]

    result_low = limit_accel_in_turns(5.0, 30.0, a_target, CP)
    result_high = limit_accel_in_turns(30.0, 30.0, a_target, CP)

    # Low speed should allow more accel
    self.assertGreater(result_low[1], result_high[1])

  def test_negative_angle(self):
    """Test works with negative steering angles."""
    CP = create_mock_cp()
    a_target = [ACCEL_MIN, 1.5]

    result_pos = limit_accel_in_turns(20.0, 30.0, a_target, CP)
    result_neg = limit_accel_in_turns(20.0, -30.0, a_target, CP)

    # Should be symmetric
    self.assertAlmostEqual(result_pos[1], result_neg[1], places=5)


class TestLongitudinalPlannerInit(unittest.TestCase):
  """Test LongitudinalPlanner initialization."""

  @patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc')
  def test_init_default(self, mock_mpc_class):
    """Test default initialization."""
    CP = create_mock_cp()
    mock_mpc = MagicMock()
    mock_mpc_class.return_value = mock_mpc

    planner = LongitudinalPlanner(CP)

    self.assertEqual(planner.CP, CP)
    self.assertFalse(planner.fcw)
    self.assertTrue(planner.allow_throttle)
    self.assertEqual(planner.a_desired, 0.0)
    self.assertEqual(planner.output_a_target, 0.0)
    self.assertFalse(planner.output_should_stop)

  @patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc')
  def test_init_with_initial_values(self, mock_mpc_class):
    """Test initialization with initial v and a."""
    CP = create_mock_cp()
    mock_mpc = MagicMock()
    mock_mpc_class.return_value = mock_mpc

    planner = LongitudinalPlanner(CP, init_v=10.0, init_a=1.0)

    self.assertEqual(planner.a_desired, 1.0)
    self.assertEqual(planner.v_desired_filter.x, 10.0)

  @patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc')
  def test_init_trajectories_shape(self, mock_mpc_class):
    """Test trajectory arrays have correct shape."""
    CP = create_mock_cp()
    mock_mpc = MagicMock()
    mock_mpc_class.return_value = mock_mpc

    planner = LongitudinalPlanner(CP)

    self.assertEqual(len(planner.v_desired_trajectory), CONTROL_N)
    self.assertEqual(len(planner.a_desired_trajectory), CONTROL_N)
    self.assertEqual(len(planner.j_desired_trajectory), CONTROL_N)

  @patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc')
  def test_init_prev_accel_clip(self, mock_mpc_class):
    """Test initial accel clip values."""
    CP = create_mock_cp()
    mock_mpc = MagicMock()
    mock_mpc_class.return_value = mock_mpc

    planner = LongitudinalPlanner(CP)

    self.assertEqual(planner.prev_accel_clip[0], ACCEL_MIN)
    self.assertEqual(planner.prev_accel_clip[1], ACCEL_MAX)


class TestLongitudinalPlannerParseModel(unittest.TestCase):
  """Test LongitudinalPlanner.parse_model static method."""

  def test_valid_model(self):
    """Test parsing valid model message."""
    model = create_mock_model_msg(valid=True, throttle_prob=0.8)

    x, v, a, j, throttle_prob = LongitudinalPlanner.parse_model(model)

    self.assertEqual(len(x), len(v))
    self.assertEqual(len(v), len(a))
    self.assertAlmostEqual(throttle_prob, 0.8)
    # Should have valid values, not zeros
    self.assertGreater(np.max(x), 0)

  def test_invalid_model_zeros(self):
    """Test parsing invalid model returns zeros."""
    model = create_mock_model_msg(valid=False)

    x, v, a, j, throttle_prob = LongitudinalPlanner.parse_model(model)

    np.testing.assert_array_equal(x, np.zeros_like(x))
    np.testing.assert_array_equal(v, np.zeros_like(v))
    np.testing.assert_array_equal(a, np.zeros_like(a))

  def test_missing_throttle_prob(self):
    """Test default throttle prob when missing."""
    model = MagicMock()
    model.position.x = list(np.zeros(ModelConstants.IDX_N))
    model.velocity.x = list(np.zeros(ModelConstants.IDX_N))
    model.acceleration.x = list(np.zeros(ModelConstants.IDX_N))
    model.meta.disengagePredictions.gasPressProbs = []

    _, _, _, _, throttle_prob = LongitudinalPlanner.parse_model(model)

    self.assertEqual(throttle_prob, 1.0)

  def test_single_throttle_prob(self):
    """Test single element throttle prob defaults to 1.0."""
    model = MagicMock()
    model.position.x = list(np.zeros(ModelConstants.IDX_N))
    model.velocity.x = list(np.zeros(ModelConstants.IDX_N))
    model.acceleration.x = list(np.zeros(ModelConstants.IDX_N))
    model.meta.disengagePredictions.gasPressProbs = [0.5]  # Only one element

    _, _, _, _, throttle_prob = LongitudinalPlanner.parse_model(model)

    self.assertEqual(throttle_prob, 1.0)


class TestLongitudinalPlannerUpdate(unittest.TestCase):
  """Test LongitudinalPlanner.update method."""

  @patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc')
  def test_update_resets_when_off(self, mock_mpc_class):
    """Test state resets when longitudinal control is off."""
    CP = create_mock_cp()
    mock_mpc = MagicMock()
    mock_mpc.v_solution = np.zeros(13)  # T_IDXS_MPC has 13 elements
    mock_mpc.a_solution = np.zeros(13)
    mock_mpc.j_solution = np.zeros(12)  # One less than v/a solutions
    mock_mpc.crash_cnt = 0
    mock_mpc_class.return_value = mock_mpc

    planner = LongitudinalPlanner(CP)
    planner.a_desired = 2.0  # Set non-zero

    sm = create_mock_sm(v_ego=10.0, long_control_off=True)
    planner.update(sm)

    # a_desired should be clipped to cruise limits
    self.assertLessEqual(planner.a_desired, get_max_accel(10.0))

  @patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc')
  def test_update_cruise_not_initialized(self, mock_mpc_class):
    """Test reset when cruise not initialized."""
    CP = create_mock_cp()
    mock_mpc = MagicMock()
    mock_mpc.v_solution = np.zeros(13)
    mock_mpc.a_solution = np.zeros(13)
    mock_mpc.j_solution = np.zeros(12)
    mock_mpc.crash_cnt = 0
    mock_mpc_class.return_value = mock_mpc

    planner = LongitudinalPlanner(CP)
    sm = create_mock_sm(v_cruise=V_CRUISE_UNSET)
    planner.update(sm)

    # Should have reset state
    self.assertAlmostEqual(planner.v_desired_filter.x, sm['carState'].vEgo, places=1)

  @patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc')
  def test_update_fcw_triggered(self, mock_mpc_class):
    """Test FCW is triggered on crash prediction."""
    CP = create_mock_cp()
    mock_mpc = MagicMock()
    mock_mpc.v_solution = np.zeros(13)
    mock_mpc.a_solution = np.zeros(13)
    mock_mpc.j_solution = np.zeros(12)
    mock_mpc.crash_cnt = 5  # Above threshold
    mock_mpc_class.return_value = mock_mpc

    planner = LongitudinalPlanner(CP)
    sm = create_mock_sm(standstill=False)
    planner.update(sm)

    self.assertTrue(planner.fcw)

  @patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc')
  def test_update_no_fcw_at_standstill(self, mock_mpc_class):
    """Test FCW not triggered at standstill."""
    CP = create_mock_cp()
    mock_mpc = MagicMock()
    mock_mpc.v_solution = np.zeros(13)
    mock_mpc.a_solution = np.zeros(13)
    mock_mpc.j_solution = np.zeros(12)
    mock_mpc.crash_cnt = 5
    mock_mpc_class.return_value = mock_mpc

    planner = LongitudinalPlanner(CP)
    sm = create_mock_sm(standstill=True)
    planner.update(sm)

    self.assertFalse(planner.fcw)

  @patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc')
  def test_update_experimental_mode(self, mock_mpc_class):
    """Test blended mode in experimental mode."""
    CP = create_mock_cp()
    mock_mpc = MagicMock()
    mock_mpc.v_solution = np.zeros(13)
    mock_mpc.a_solution = np.zeros(13)
    mock_mpc.j_solution = np.zeros(12)
    mock_mpc.crash_cnt = 0
    mock_mpc_class.return_value = mock_mpc

    planner = LongitudinalPlanner(CP)
    sm = create_mock_sm(experimental_mode=True)
    planner.update(sm)

    # Check mode was set
    mock_mpc.set_weights.assert_called()

  @patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc')
  def test_update_force_decel(self, mock_mpc_class):
    """Test force decel sets v_cruise to zero."""
    CP = create_mock_cp()
    mock_mpc = MagicMock()
    mock_mpc.v_solution = np.zeros(13)
    mock_mpc.a_solution = np.zeros(13)
    mock_mpc.j_solution = np.zeros(12)
    mock_mpc.crash_cnt = 0
    mock_mpc_class.return_value = mock_mpc

    planner = LongitudinalPlanner(CP)
    sm = create_mock_sm(force_decel=True)
    planner.update(sm)

    # mpc.update should have been called with v_cruise=0
    args = mock_mpc.update.call_args
    self.assertEqual(args[0][1], 0.0)  # v_cruise argument

  @patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc')
  def test_update_allow_throttle_low_prob(self, mock_mpc_class):
    """Test allow_throttle is False when throttle_prob is low."""
    CP = create_mock_cp()
    mock_mpc = MagicMock()
    mock_mpc.v_solution = np.zeros(13)
    mock_mpc.a_solution = np.zeros(13)
    mock_mpc.j_solution = np.zeros(12)
    mock_mpc.crash_cnt = 0
    mock_mpc_class.return_value = mock_mpc

    planner = LongitudinalPlanner(CP)
    model = create_mock_model_msg(valid=True, throttle_prob=0.1)
    sm = create_mock_sm(v_ego=10.0, model_msg=model)  # Above MIN_ALLOW_THROTTLE_SPEED
    planner.update(sm)

    self.assertFalse(planner.allow_throttle)

  @patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc')
  def test_update_allow_throttle_low_speed(self, mock_mpc_class):
    """Test allow_throttle is True at low speed regardless of prob."""
    CP = create_mock_cp()
    mock_mpc = MagicMock()
    mock_mpc.v_solution = np.zeros(13)
    mock_mpc.a_solution = np.zeros(13)
    mock_mpc.j_solution = np.zeros(12)
    mock_mpc.crash_cnt = 0
    mock_mpc_class.return_value = mock_mpc

    planner = LongitudinalPlanner(CP)
    model = create_mock_model_msg(valid=True, throttle_prob=0.1)
    sm = create_mock_sm(v_ego=1.0, model_msg=model)  # Below MIN_ALLOW_THROTTLE_SPEED
    planner.update(sm)

    self.assertTrue(planner.allow_throttle)

  @patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc')
  def test_update_accel_clip_smoothed(self, mock_mpc_class):
    """Test accel clip values are smoothed frame to frame."""
    CP = create_mock_cp()
    mock_mpc = MagicMock()
    mock_mpc.v_solution = np.zeros(13)
    mock_mpc.a_solution = np.ones(13) * 1.0  # Constant accel
    mock_mpc.j_solution = np.zeros(12)
    mock_mpc.crash_cnt = 0
    mock_mpc_class.return_value = mock_mpc

    planner = LongitudinalPlanner(CP)
    # Set initial prev_accel_clip to extreme values
    planner.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]

    sm = create_mock_sm(v_ego=15.0)
    planner.update(sm)

    # Clip values should only change by max 0.05 per frame
    self.assertLessEqual(
      abs(planner.prev_accel_clip[0] - ACCEL_MIN), 0.05 + 0.001
    )


class TestLongitudinalPlannerPublish(unittest.TestCase):
  """Test LongitudinalPlanner.publish method."""

  @patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc')
  @patch('openpilot.selfdrive.controls.lib.longitudinal_planner.messaging')
  def test_publish_creates_message(self, mock_messaging, mock_mpc_class):
    """Test publish creates longitudinalPlan message."""
    CP = create_mock_cp()
    mock_mpc = MagicMock()
    mock_mpc.solve_time = 0.01
    mock_mpc.source = 'cruise'
    mock_mpc.crash_cnt = 0
    mock_mpc_class.return_value = mock_mpc

    planner = LongitudinalPlanner(CP)
    planner.output_a_target = 0.5
    planner.output_should_stop = False
    planner.allow_throttle = True
    planner.fcw = False
    planner.v_desired_trajectory = np.ones(CONTROL_N) * 10.0
    planner.a_desired_trajectory = np.ones(CONTROL_N) * 0.5
    planner.j_desired_trajectory = np.zeros(CONTROL_N)

    mock_plan = MagicMock()
    mock_plan.logMonoTime = 1000001
    mock_messaging.new_message.return_value = mock_plan

    sm = create_mock_sm()
    pm = MagicMock()

    planner.publish(sm, pm)

    mock_messaging.new_message.assert_called_once_with('longitudinalPlan')
    pm.send.assert_called_once_with('longitudinalPlan', mock_plan)

  @patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc')
  @patch('openpilot.selfdrive.controls.lib.longitudinal_planner.messaging')
  def test_publish_sets_values(self, mock_messaging, mock_mpc_class):
    """Test publish sets correct values in message."""
    CP = create_mock_cp()
    mock_mpc = MagicMock()
    mock_mpc.solve_time = 0.015
    mock_mpc.source = 'lead0'
    mock_mpc.crash_cnt = 0
    mock_mpc_class.return_value = mock_mpc

    planner = LongitudinalPlanner(CP)
    planner.output_a_target = -1.5
    planner.output_should_stop = True
    planner.allow_throttle = False
    planner.fcw = True
    planner.v_desired_trajectory = np.linspace(10, 0, CONTROL_N)
    planner.a_desired_trajectory = np.ones(CONTROL_N) * -1.5
    planner.j_desired_trajectory = np.zeros(CONTROL_N)

    mock_plan = MagicMock()
    mock_plan.logMonoTime = 1000001
    mock_plan.longitudinalPlan = MagicMock()
    mock_messaging.new_message.return_value = mock_plan

    sm = create_mock_sm()
    sm['radarState'].leadOne.status = True
    pm = MagicMock()

    planner.publish(sm, pm)

    lp = mock_plan.longitudinalPlan
    self.assertEqual(lp.aTarget, -1.5)
    self.assertTrue(lp.shouldStop)
    self.assertFalse(lp.allowThrottle)
    self.assertTrue(lp.fcw)
    self.assertTrue(lp.hasLead)
    self.assertEqual(lp.longitudinalPlanSource, 'lead0')


class TestConstants(unittest.TestCase):
  """Test module constants."""

  def test_lon_mpc_step_positive(self):
    """Test LON_MPC_STEP is positive."""
    self.assertGreater(LON_MPC_STEP, 0)

  def test_a_cruise_max_vals_decreasing(self):
    """Test A_CRUISE_MAX_VALS decreases with speed."""
    for i in range(len(A_CRUISE_MAX_VALS) - 1):
      self.assertGreaterEqual(A_CRUISE_MAX_VALS[i], A_CRUISE_MAX_VALS[i + 1])

  def test_a_cruise_max_bp_increasing(self):
    """Test A_CRUISE_MAX_BP is increasing."""
    for i in range(len(A_CRUISE_MAX_BP) - 1):
      self.assertLess(A_CRUISE_MAX_BP[i], A_CRUISE_MAX_BP[i + 1])

  def test_allow_throttle_threshold_valid(self):
    """Test ALLOW_THROTTLE_THRESHOLD is in valid range."""
    self.assertGreater(ALLOW_THROTTLE_THRESHOLD, 0)
    self.assertLess(ALLOW_THROTTLE_THRESHOLD, 1)

  def test_min_allow_throttle_speed_positive(self):
    """Test MIN_ALLOW_THROTTLE_SPEED is positive."""
    self.assertGreater(MIN_ALLOW_THROTTLE_SPEED, 0)


if __name__ == '__main__':
  unittest.main()
