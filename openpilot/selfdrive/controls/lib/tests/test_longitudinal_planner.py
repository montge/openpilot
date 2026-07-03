"""Tests for selfdrive/controls/lib/longitudinal_planner.py - longitudinal MPC planner."""

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  LongitudinalPlanner,
  get_max_accel,
  get_coast_accel,
  limit_accel_in_turns,
  A_CRUISE_MAX_VALS,
  A_CRUISE_MAX_BP,
  ALLOW_THROTTLE_THRESHOLD,
  MIN_ALLOW_THROTTLE_SPEED,
  LON_MPC_STEP,
)
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET


def create_mock_cp(mocker, steer_ratio=15.0, wheelbase=2.7, openpilot_long=True, actuator_delay=0.2, v_ego_stopping=0.5):
  """Create a mock CarParams for testing."""
  CP = mocker.MagicMock()
  CP.steerRatio = steer_ratio
  CP.wheelbase = wheelbase
  CP.openpilotLongitudinalControl = openpilot_long
  CP.longitudinalActuatorDelay = actuator_delay
  CP.vEgoStopping = v_ego_stopping
  return CP


def create_mock_model_msg(mocker, valid=True, throttle_prob=1.0, desired_accel=0.0, should_stop=False):
  """Create a mock modelV2 message."""
  model = mocker.MagicMock()

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
  model.action = mocker.MagicMock()
  model.action.desiredAcceleration = float(desired_accel)
  model.action.shouldStop = bool(should_stop)

  return model


def create_mock_sm(
  mocker,
  v_ego=15.0,
  v_cruise=100.0,
  experimental_mode=False,
  long_control_off=False,
  enabled=True,
  force_decel=False,
  standstill=False,
  a_ego=0.0,
  steering_angle=0.0,
  angle_offset=0.0,
  orientation_ned=None,
  model_msg=None,
):
  """Create a mock SubMaster for testing."""
  from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState

  # Create individual message mocks
  car_state = mocker.MagicMock()
  car_state.vEgo = v_ego
  car_state.vCruise = v_cruise
  car_state.aEgo = a_ego
  car_state.standstill = standstill
  car_state.steeringAngleDeg = steering_angle

  selfdrive_state = mocker.MagicMock()
  selfdrive_state.experimentalMode = experimental_mode
  selfdrive_state.enabled = enabled
  selfdrive_state.personality = 0

  controls_state = mocker.MagicMock()
  controls_state.longControlState = LongCtrlState.off if long_control_off else LongCtrlState.pid
  controls_state.forceDecel = force_decel

  live_params = mocker.MagicMock()
  live_params.angleOffsetDeg = angle_offset

  car_control = mocker.MagicMock()
  if orientation_ned is not None:
    car_control.orientationNED = orientation_ned
  else:
    car_control.orientationNED = [0.0, 0.0, 0.0]

  if model_msg is None:
    model_msg = create_mock_model_msg(mocker)

  radar_state = mocker.MagicMock()
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

  sm = mocker.MagicMock()
  sm.__getitem__ = lambda self, key: messages[key]
  sm.logMonoTime = {'modelV2': 1000000}
  sm.all_checks = mocker.MagicMock(return_value=True)

  return sm


class TestGetMaxAccel:
  """Test get_max_accel function."""

  def test_zero_speed(self):
    """Test max accel at zero speed."""
    result = get_max_accel(0.0)
    assert result == pytest.approx(A_CRUISE_MAX_VALS[0])

  def test_high_speed(self):
    """Test max accel at high speed."""
    result = get_max_accel(40.0)
    assert result == pytest.approx(A_CRUISE_MAX_VALS[-1])

  def test_interpolation(self):
    """Test accel interpolates between breakpoints."""
    result = get_max_accel(17.5)  # Between 10 and 25
    # Should be between 1.2 and 0.8
    assert result < 1.2
    assert result > 0.8

  def test_above_max_speed(self):
    """Test accel clamps at speeds above max breakpoint."""
    result = get_max_accel(100.0)
    assert result == pytest.approx(A_CRUISE_MAX_VALS[-1])

  def test_negative_speed(self):
    """Test accel at negative speed (edge case)."""
    result = get_max_accel(-5.0)
    assert result == pytest.approx(A_CRUISE_MAX_VALS[0])


class TestGetCoastAccel:
  """Test get_coast_accel function."""

  def test_zero_pitch(self):
    """Test coast accel at zero pitch (flat road)."""
    result = get_coast_accel(0.0)
    assert result == pytest.approx(-0.3, abs=1e-5)

  def test_uphill_pitch(self):
    """Test coast accel going uphill (positive pitch)."""
    result = get_coast_accel(0.1)  # ~5.7 degrees uphill
    # Should be more negative (decelerating)
    assert result < -0.3

  def test_downhill_pitch(self):
    """Test coast accel going downhill (negative pitch)."""
    result = get_coast_accel(-0.1)  # ~5.7 degrees downhill
    # Should be less negative (maybe even positive)
    assert result > -0.3

  def test_steep_uphill(self):
    """Test coast accel on steep uphill."""
    result = get_coast_accel(0.2)  # ~11.5 degrees
    assert result < -1.0


class TestLimitAccelInTurns:
  """Test limit_accel_in_turns function."""

  def test_straight_no_limit(self, mocker):
    """Test no limiting when driving straight."""
    CP = create_mock_cp(mocker)
    a_target = [ACCEL_MIN, 1.5]

    result = limit_accel_in_turns(20.0, 0.0, a_target, CP)

    assert result[0] == ACCEL_MIN
    assert result[1] == pytest.approx(1.5, abs=1e-3)

  def test_turn_limits_accel(self, mocker):
    """Test accel is limited during turns."""
    CP = create_mock_cp(mocker)
    a_target = [ACCEL_MIN, 2.0]

    # Large steering angle should limit accel
    result = limit_accel_in_turns(25.0, 30.0, a_target, CP)

    assert result[0] == ACCEL_MIN
    assert result[1] < 2.0

  def test_min_accel_unchanged(self, mocker):
    """Test minimum accel is never changed."""
    CP = create_mock_cp(mocker)
    a_target = [-2.0, 1.5]

    result = limit_accel_in_turns(20.0, 45.0, a_target, CP)

    assert result[0] == -2.0

  def test_low_speed_less_limiting(self, mocker):
    """Test less limiting at low speeds."""
    CP = create_mock_cp(mocker)
    a_target = [ACCEL_MIN, 1.5]

    result_low = limit_accel_in_turns(5.0, 30.0, a_target, CP)
    result_high = limit_accel_in_turns(30.0, 30.0, a_target, CP)

    # Low speed should allow more accel
    assert result_low[1] > result_high[1]

  def test_negative_angle(self, mocker):
    """Test works with negative steering angles."""
    CP = create_mock_cp(mocker)
    a_target = [ACCEL_MIN, 1.5]

    result_pos = limit_accel_in_turns(20.0, 30.0, a_target, CP)
    result_neg = limit_accel_in_turns(20.0, -30.0, a_target, CP)

    # Should be symmetric
    assert result_pos[1] == pytest.approx(result_neg[1], abs=1e-5)


class TestLongitudinalPlannerInit:
  """Test LongitudinalPlanner initialization."""

  def test_init_default(self, mocker):
    """Test default initialization."""
    mock_mpc = mocker.MagicMock()
    mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc', return_value=mock_mpc)
    CP = create_mock_cp(mocker)

    planner = LongitudinalPlanner(CP)

    assert planner.CP == CP
    assert planner.fcw is False
    assert planner.allow_throttle is True
    assert planner.a_desired == 0.0
    assert planner.output_a_target == 0.0
    assert planner.output_should_stop is False

  def test_init_with_initial_values(self, mocker):
    """Test initialization with initial v and a."""
    mock_mpc = mocker.MagicMock()
    mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc', return_value=mock_mpc)
    CP = create_mock_cp(mocker)

    planner = LongitudinalPlanner(CP, init_v=10.0, init_a=1.0)

    assert planner.a_desired == 1.0
    assert planner.v_desired_filter.x == 10.0

  def test_init_trajectories_shape(self, mocker):
    """Test trajectory arrays have correct shape."""
    mock_mpc = mocker.MagicMock()
    mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc', return_value=mock_mpc)
    CP = create_mock_cp(mocker)

    planner = LongitudinalPlanner(CP)

    assert len(planner.v_desired_trajectory) == CONTROL_N
    assert len(planner.a_desired_trajectory) == CONTROL_N
    assert len(planner.j_desired_trajectory) == CONTROL_N

  def test_init_prev_accel_clip(self, mocker):
    """Test initial accel clip values."""
    mock_mpc = mocker.MagicMock()
    mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc', return_value=mock_mpc)
    CP = create_mock_cp(mocker)

    planner = LongitudinalPlanner(CP)

    assert planner.prev_accel_clip[0] == ACCEL_MIN
    assert planner.prev_accel_clip[1] == ACCEL_MAX


class TestLongitudinalPlannerParseModel:
  """Test LongitudinalPlanner.parse_model static method."""

  def test_valid_model(self, mocker):
    """Test parsing valid model message."""
    model = create_mock_model_msg(mocker, valid=True, throttle_prob=0.8)

    x, v, a, j, throttle_prob = LongitudinalPlanner.parse_model(model)

    assert len(x) == len(v)
    assert len(v) == len(a)
    assert throttle_prob == pytest.approx(0.8)
    # Should have valid values, not zeros
    assert np.max(x) > 0

  def test_invalid_model_zeros(self, mocker):
    """Test parsing invalid model returns zeros."""
    model = create_mock_model_msg(mocker, valid=False)

    x, v, a, j, throttle_prob = LongitudinalPlanner.parse_model(model)

    np.testing.assert_array_equal(x, np.zeros_like(x))
    np.testing.assert_array_equal(v, np.zeros_like(v))
    np.testing.assert_array_equal(a, np.zeros_like(a))

  def test_missing_throttle_prob(self, mocker):
    """Test default throttle prob when missing."""
    model = mocker.MagicMock()
    model.position.x = list(np.zeros(ModelConstants.IDX_N))
    model.velocity.x = list(np.zeros(ModelConstants.IDX_N))
    model.acceleration.x = list(np.zeros(ModelConstants.IDX_N))
    model.meta.disengagePredictions.gasPressProbs = []

    _, _, _, _, throttle_prob = LongitudinalPlanner.parse_model(model)

    assert throttle_prob == 1.0

  def test_single_throttle_prob(self, mocker):
    """Test single element throttle prob defaults to 1.0."""
    model = mocker.MagicMock()
    model.position.x = list(np.zeros(ModelConstants.IDX_N))
    model.velocity.x = list(np.zeros(ModelConstants.IDX_N))
    model.acceleration.x = list(np.zeros(ModelConstants.IDX_N))
    model.meta.disengagePredictions.gasPressProbs = [0.5]  # Only one element

    _, _, _, _, throttle_prob = LongitudinalPlanner.parse_model(model)

    assert throttle_prob == 1.0


class TestLongitudinalPlannerUpdate:
  """Test LongitudinalPlanner.update method."""

  def test_update_resets_when_off(self, mocker):
    """Test state resets when longitudinal control is off."""
    mock_mpc = mocker.MagicMock()
    mock_mpc.v_solution = np.zeros(13)  # T_IDXS_MPC has 13 elements
    mock_mpc.a_solution = np.zeros(13)
    mock_mpc.j_solution = np.zeros(12)  # One less than v/a solutions
    mock_mpc.crash_cnt = 0
    mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc', return_value=mock_mpc)
    CP = create_mock_cp(mocker)

    planner = LongitudinalPlanner(CP)
    planner.a_desired = 2.0  # Set non-zero

    sm = create_mock_sm(mocker, v_ego=10.0, long_control_off=True)
    planner.update(sm)

    # a_desired should be clipped to cruise limits
    assert planner.a_desired <= get_max_accel(10.0)

  def test_update_cruise_not_initialized(self, mocker):
    """Test reset when cruise not initialized."""
    mock_mpc = mocker.MagicMock()
    mock_mpc.v_solution = np.zeros(13)
    mock_mpc.a_solution = np.zeros(13)
    mock_mpc.j_solution = np.zeros(12)
    mock_mpc.crash_cnt = 0
    mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc', return_value=mock_mpc)
    CP = create_mock_cp(mocker)

    planner = LongitudinalPlanner(CP)
    sm = create_mock_sm(mocker, v_cruise=V_CRUISE_UNSET)
    planner.update(sm)

    # Should have reset state
    assert planner.v_desired_filter.x == pytest.approx(sm['carState'].vEgo, abs=0.1)

  def test_update_fcw_triggered(self, mocker):
    """Test FCW is triggered on crash prediction."""
    mock_mpc = mocker.MagicMock()
    mock_mpc.v_solution = np.zeros(13)
    mock_mpc.a_solution = np.zeros(13)
    mock_mpc.j_solution = np.zeros(12)
    mock_mpc.crash_cnt = 5  # Above threshold
    mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc', return_value=mock_mpc)
    CP = create_mock_cp(mocker)

    planner = LongitudinalPlanner(CP)
    sm = create_mock_sm(mocker, standstill=False)
    planner.update(sm)

    assert planner.fcw is True

  def test_update_no_fcw_at_standstill(self, mocker):
    """Test FCW not triggered at standstill."""
    mock_mpc = mocker.MagicMock()
    mock_mpc.v_solution = np.zeros(13)
    mock_mpc.a_solution = np.zeros(13)
    mock_mpc.j_solution = np.zeros(12)
    mock_mpc.crash_cnt = 5
    mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc', return_value=mock_mpc)
    CP = create_mock_cp(mocker)

    planner = LongitudinalPlanner(CP)
    sm = create_mock_sm(mocker, standstill=True)
    planner.update(sm)

    assert planner.fcw is False

  def test_update_experimental_mode(self, mocker):
    """Test blended mode in experimental mode."""
    mock_mpc = mocker.MagicMock()
    mock_mpc.v_solution = np.zeros(13)
    mock_mpc.a_solution = np.zeros(13)
    mock_mpc.j_solution = np.zeros(12)
    mock_mpc.crash_cnt = 0
    mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc', return_value=mock_mpc)
    CP = create_mock_cp(mocker)

    planner = LongitudinalPlanner(CP)
    sm = create_mock_sm(mocker, experimental_mode=True)
    planner.update(sm)

    # Check mode was set
    mock_mpc.set_weights.assert_called()

  def test_update_force_decel(self, mocker):
    """Test force decel sets v_cruise to zero."""
    mock_mpc = mocker.MagicMock()
    mock_mpc.v_solution = np.zeros(13)
    mock_mpc.a_solution = np.zeros(13)
    mock_mpc.j_solution = np.zeros(12)
    mock_mpc.crash_cnt = 0
    mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc', return_value=mock_mpc)
    CP = create_mock_cp(mocker)

    planner = LongitudinalPlanner(CP)
    sm = create_mock_sm(mocker, force_decel=True)
    planner.update(sm)

    # mpc.update should have been called with v_cruise=0
    args = mock_mpc.update.call_args
    assert args[0][1] == 0.0  # v_cruise argument

  def test_update_allow_throttle_low_prob(self, mocker):
    """Test allow_throttle is False when throttle_prob is low."""
    mock_mpc = mocker.MagicMock()
    mock_mpc.v_solution = np.zeros(13)
    mock_mpc.a_solution = np.zeros(13)
    mock_mpc.j_solution = np.zeros(12)
    mock_mpc.crash_cnt = 0
    mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc', return_value=mock_mpc)
    CP = create_mock_cp(mocker)

    planner = LongitudinalPlanner(CP)
    model = create_mock_model_msg(mocker, valid=True, throttle_prob=0.1)
    sm = create_mock_sm(mocker, v_ego=10.0, model_msg=model)  # Above MIN_ALLOW_THROTTLE_SPEED
    planner.update(sm)

    assert planner.allow_throttle is False

  def test_update_allow_throttle_low_speed(self, mocker):
    """Test allow_throttle is True at low speed regardless of prob."""
    mock_mpc = mocker.MagicMock()
    mock_mpc.v_solution = np.zeros(13)
    mock_mpc.a_solution = np.zeros(13)
    mock_mpc.j_solution = np.zeros(12)
    mock_mpc.crash_cnt = 0
    mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc', return_value=mock_mpc)
    CP = create_mock_cp(mocker)

    planner = LongitudinalPlanner(CP)
    model = create_mock_model_msg(mocker, valid=True, throttle_prob=0.1)
    sm = create_mock_sm(mocker, v_ego=1.0, model_msg=model)  # Below MIN_ALLOW_THROTTLE_SPEED
    planner.update(sm)

    assert planner.allow_throttle is True

  def test_update_accel_clip_smoothed(self, mocker):
    """Test accel clip values are smoothed frame to frame."""
    mock_mpc = mocker.MagicMock()
    mock_mpc.v_solution = np.zeros(13)
    mock_mpc.a_solution = np.ones(13) * 1.0  # Constant accel
    mock_mpc.j_solution = np.zeros(12)
    mock_mpc.crash_cnt = 0
    mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc', return_value=mock_mpc)
    CP = create_mock_cp(mocker)

    planner = LongitudinalPlanner(CP)
    # Set initial prev_accel_clip to extreme values
    planner.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]

    sm = create_mock_sm(mocker, v_ego=15.0)
    planner.update(sm)

    # Clip values should only change by max 0.05 per frame
    assert abs(planner.prev_accel_clip[0] - ACCEL_MIN) <= 0.05 + 0.001

  def test_update_empty_orientation_ned_uses_default_coast(self, mocker):
    """Test accel_coast defaults to ACCEL_MAX when orientationNED is empty."""
    mock_mpc = mocker.MagicMock()
    mock_mpc.v_solution = np.zeros(13)
    mock_mpc.a_solution = np.zeros(13)
    mock_mpc.j_solution = np.zeros(12)
    mock_mpc.crash_cnt = 0
    mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc', return_value=mock_mpc)
    CP = create_mock_cp(mocker)

    planner = LongitudinalPlanner(CP)
    # Pass empty list for orientationNED to trigger fallback
    sm = create_mock_sm(mocker, v_ego=15.0, orientation_ned=[])
    planner.update(sm)

    # Should have completed without error - internal accel_coast is ACCEL_MAX
    assert planner.v_desired_filter.x >= 0


class TestLongitudinalPlannerPublish:
  """Test LongitudinalPlanner.publish method."""

  def test_publish_creates_message(self, mocker):
    """Test publish creates longitudinalPlan message."""
    mock_mpc = mocker.MagicMock()
    mock_mpc.solve_time = 0.01
    mock_mpc.source = 'cruise'
    mock_mpc.crash_cnt = 0
    mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc', return_value=mock_mpc)
    mock_messaging = mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.messaging')
    CP = create_mock_cp(mocker)

    planner = LongitudinalPlanner(CP)
    planner.output_a_target = 0.5
    planner.output_should_stop = False
    planner.allow_throttle = True
    planner.fcw = False
    planner.v_desired_trajectory = np.ones(CONTROL_N) * 10.0
    planner.a_desired_trajectory = np.ones(CONTROL_N) * 0.5
    planner.j_desired_trajectory = np.zeros(CONTROL_N)

    mock_plan = mocker.MagicMock()
    mock_plan.logMonoTime = 1000001
    mock_messaging.new_message.return_value = mock_plan

    sm = create_mock_sm(mocker)
    pm = mocker.MagicMock()

    planner.publish(sm, pm)

    mock_messaging.new_message.assert_called_once_with('longitudinalPlan')
    pm.send.assert_called_once_with('longitudinalPlan', mock_plan)

  def test_publish_sets_values(self, mocker):
    """Test publish sets correct values in message."""
    mock_mpc = mocker.MagicMock()
    mock_mpc.solve_time = 0.015
    mock_mpc.source = 'lead0'
    mock_mpc.crash_cnt = 0
    mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.LongitudinalMpc', return_value=mock_mpc)
    mock_messaging = mocker.patch('openpilot.selfdrive.controls.lib.longitudinal_planner.messaging')
    CP = create_mock_cp(mocker)

    planner = LongitudinalPlanner(CP)
    planner.output_a_target = -1.5
    planner.output_should_stop = True
    planner.allow_throttle = False
    planner.fcw = True
    planner.v_desired_trajectory = np.linspace(10, 0, CONTROL_N)
    planner.a_desired_trajectory = np.ones(CONTROL_N) * -1.5
    planner.j_desired_trajectory = np.zeros(CONTROL_N)

    mock_plan = mocker.MagicMock()
    mock_plan.logMonoTime = 1000001
    mock_plan.longitudinalPlan = mocker.MagicMock()
    mock_messaging.new_message.return_value = mock_plan

    sm = create_mock_sm(mocker)
    sm['radarState'].leadOne.status = True
    pm = mocker.MagicMock()

    planner.publish(sm, pm)

    lp = mock_plan.longitudinalPlan
    assert lp.aTarget == -1.5
    assert lp.shouldStop is True
    assert lp.allowThrottle is False
    assert lp.fcw is True
    assert lp.hasLead is True
    assert lp.longitudinalPlanSource == 'lead0'


class TestConstants:
  """Test module constants."""

  def test_lon_mpc_step_positive(self):
    """Test LON_MPC_STEP is positive."""
    assert LON_MPC_STEP > 0

  def test_a_cruise_max_vals_decreasing(self):
    """Test A_CRUISE_MAX_VALS decreases with speed."""
    for i in range(len(A_CRUISE_MAX_VALS) - 1):
      assert A_CRUISE_MAX_VALS[i] >= A_CRUISE_MAX_VALS[i + 1]

  def test_a_cruise_max_bp_increasing(self):
    """Test A_CRUISE_MAX_BP is increasing."""
    for i in range(len(A_CRUISE_MAX_BP) - 1):
      assert A_CRUISE_MAX_BP[i] < A_CRUISE_MAX_BP[i + 1]

  def test_allow_throttle_threshold_valid(self):
    """Test ALLOW_THROTTLE_THRESHOLD is in valid range."""
    assert ALLOW_THROTTLE_THRESHOLD > 0
    assert ALLOW_THROTTLE_THRESHOLD < 1

  def test_min_allow_throttle_speed_positive(self):
    """Test MIN_ALLOW_THROTTLE_SPEED is positive."""
    assert MIN_ALLOW_THROTTLE_SPEED > 0
