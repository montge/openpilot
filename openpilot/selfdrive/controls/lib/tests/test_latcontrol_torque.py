"""Tests for selfdrive/controls/lib/latcontrol_torque.py - torque-based lateral control."""

import pytest

from openpilot.selfdrive.controls.lib.latcontrol_torque import (
  LatControlTorque,
  KP,
  KI,
  INTERP_SPEEDS,
  KP_INTERP,
  JERK_LOOKAHEAD_SECONDS,
  JERK_GAIN,
  LAT_ACCEL_REQUEST_BUFFER_SECONDS,
  VERSION,
)


def create_mock_torque_params(mocker):
  """Create mock torque params."""
  params = mocker.MagicMock()
  params.latAccelFactor = 1.0
  params.latAccelOffset = 0.0
  params.friction = 0.1
  params.steeringAngleDeadzoneDeg = 0.0
  return params


def create_mock_cp(mocker):
  """Create a mock CarParams for LatControlTorque."""
  CP = mocker.MagicMock()
  CP.steerLimitTimer = 4.0
  torque_params = create_mock_torque_params(mocker)
  CP.lateralTuning.torque.as_builder.return_value = torque_params
  CP.lateralTuning.torque.steeringAngleDeadzoneDeg = 0.0
  return CP


def _default_torque_from_accel(accel, params):
  return accel


def _default_accel_from_torque(torque, params):
  return torque


def create_mock_ci(mocker, torque_from_accel=None, accel_from_torque=None):
  """Create a mock CarInterface."""
  CI = mocker.MagicMock()

  # Default: identity functions
  if torque_from_accel is None:
    torque_from_accel = _default_torque_from_accel
  if accel_from_torque is None:
    accel_from_torque = _default_accel_from_torque

  CI.torque_from_lateral_accel.return_value = torque_from_accel
  CI.lateral_accel_from_torque.return_value = accel_from_torque
  return CI


def create_mock_cs(mocker, v_ego=15.0, steering_angle_deg=0.0, steering_pressed=False):
  """Create a mock CarState."""
  CS = mocker.MagicMock()
  CS.vEgo = v_ego
  CS.steeringAngleDeg = steering_angle_deg
  CS.steeringPressed = steering_pressed
  return CS


def create_mock_vm(mocker, calc_curvature_return=0.0):
  """Create a mock VehicleModel."""
  VM = mocker.MagicMock()
  VM.calc_curvature.return_value = calc_curvature_return
  return VM


def create_mock_params(mocker, angle_offset_deg=0.0, roll=0.0):
  """Create mock calibration params."""
  params = mocker.MagicMock()
  params.angleOffsetDeg = angle_offset_deg
  params.roll = roll
  return params


class TestLatControlTorqueInit:
  """Test LatControlTorque initialization."""

  def test_init_gets_torque_params(self, mocker):
    """Test initialization gets torque params from CP."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)

    controller = LatControlTorque(CP, CI, 0.01)

    CP.lateralTuning.torque.as_builder.assert_called_once()
    assert controller.torque_params is not None

  def test_init_gets_conversion_functions(self, mocker):
    """Test initialization gets torque conversion functions."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)

    controller = LatControlTorque(CP, CI, 0.01)

    CI.torque_from_lateral_accel.assert_called_once()
    CI.lateral_accel_from_torque.assert_called_once()
    assert controller.torque_from_lateral_accel is not None
    assert controller.lateral_accel_from_torque is not None

  def test_init_creates_pid_controller(self, mocker):
    """Test initialization creates PID controller."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)

    controller = LatControlTorque(CP, CI, 0.01)

    assert controller.pid is not None

  def test_init_creates_buffer(self, mocker):
    """Test initialization creates lateral accel request buffer."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    dt = 0.01

    controller = LatControlTorque(CP, CI, dt)

    expected_len = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / dt)
    assert len(controller.lat_accel_request_buffer) == expected_len
    assert controller.lat_accel_request_buffer_len == expected_len

  def test_init_creates_jerk_filter(self, mocker):
    """Test initialization creates jerk filter."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)

    controller = LatControlTorque(CP, CI, 0.01)

    assert controller.jerk_filter is not None

  def test_init_calculates_lookahead_frames(self, mocker):
    """Test initialization calculates lookahead frames."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    dt = 0.01

    controller = LatControlTorque(CP, CI, dt)

    expected_frames = int(JERK_LOOKAHEAD_SECONDS / dt)
    assert controller.lookahead_frames == expected_frames

  def test_init_inherits_from_base(self, mocker):
    """Test that base class attributes are initialized."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)

    controller = LatControlTorque(CP, CI, 0.01)

    assert controller.dt == 0.01
    assert controller.sat_limit == 4.0
    assert controller.steer_max == 1.0


class TestLatControlTorqueUpdateLiveParams:
  """Test update_live_torque_params method."""

  def test_updates_lat_accel_factor(self, mocker):
    """Test updating latAccelFactor."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)

    controller.update_live_torque_params(2.0, 0.1, 0.2)

    assert controller.torque_params.latAccelFactor == 2.0

  def test_updates_lat_accel_offset(self, mocker):
    """Test updating latAccelOffset."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)

    controller.update_live_torque_params(1.0, 0.5, 0.1)

    assert controller.torque_params.latAccelOffset == 0.5

  def test_updates_friction(self, mocker):
    """Test updating friction."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)

    controller.update_live_torque_params(1.0, 0.0, 0.3)

    assert controller.torque_params.friction == 0.3

  def test_calls_update_limits(self, mocker):
    """Test that update_live_torque_params calls update_limits."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)
    mocker.patch.object(controller, 'update_limits')

    controller.update_live_torque_params(1.0, 0.0, 0.1)

    controller.update_limits.assert_called_once()


class TestLatControlTorqueUpdateLimits:
  """Test update_limits method."""

  def test_sets_pid_limits(self, mocker):
    """Test that update_limits sets PID controller limits."""
    CP = create_mock_cp(mocker)
    # Return specific values for limit calculation
    CI = create_mock_ci(mocker, accel_from_torque=lambda t, p: t * 2.0)
    controller = LatControlTorque(CP, CI, 0.01)

    # Limits should be set based on steer_max (1.0)
    assert controller.pid.pos_limit == 2.0  # 1.0 * 2.0
    assert controller.pid.neg_limit == -2.0  # -1.0 * 2.0


class TestLatControlTorqueUpdateInactive:
  """Test LatControlTorque update when inactive."""

  def test_inactive_returns_zero_torque(self, mocker):
    """Test that inactive update returns zero torque."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)
    CS = create_mock_cs(mocker)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker)

    torque, angle, log = controller.update(
      active=False, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    assert torque == 0.0
    assert angle == 0.0
    assert log.active is False

  def test_inactive_log_has_version(self, mocker):
    """Test that inactive log contains version."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)
    CS = create_mock_cs(mocker)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker)

    _, _, log = controller.update(
      active=False, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.0, curvature_limited=False, lat_delay=0.1
    )

    assert log.version == VERSION


class TestLatControlTorqueUpdateActive:
  """Test LatControlTorque update when active."""

  def test_active_returns_torque(self, mocker):
    """Test that active update returns torque output."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)
    CS = create_mock_cs(mocker, v_ego=20.0, steering_angle_deg=0.0)
    VM = create_mock_vm(mocker, calc_curvature_return=0.0)
    params = create_mock_params(mocker)

    torque, angle, log = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    assert log.active is True
    assert log.version == VERSION

  def test_active_calculates_lateral_accel(self, mocker):
    """Test that active update calculates lateral acceleration."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)
    CS = create_mock_cs(mocker, v_ego=20.0, steering_angle_deg=5.0)
    VM = create_mock_vm(mocker, calc_curvature_return=0.01)
    params = create_mock_params(mocker)

    _, _, log = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    # measurement = -curvature * v_ego^2 = -0.01 * 400 = -4.0
    assert log.actualLateralAccel == pytest.approx(-0.01 * 20.0**2, abs=0.1)

  def test_active_updates_buffer(self, mocker):
    """Test that active update appends to buffer."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)
    CS = create_mock_cs(mocker, v_ego=20.0)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker)

    controller.update(active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1)

    # Buffer should have new value: desired_curvature * v_ego^2 = 0.01 * 400 = 4.0
    assert controller.lat_accel_request_buffer[-1] == pytest.approx(0.01 * 20.0**2, abs=0.01)

  def test_active_log_contains_pid_values(self, mocker):
    """Test that active log contains PID component values."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)
    CS = create_mock_cs(mocker, v_ego=20.0)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker)

    _, _, log = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    assert hasattr(log, 'p')
    assert hasattr(log, 'i')
    assert hasattr(log, 'd')
    assert hasattr(log, 'f')
    assert hasattr(log, 'output')

  def test_active_log_contains_jerk(self, mocker):
    """Test that active log contains desired lateral jerk."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)
    CS = create_mock_cs(mocker, v_ego=20.0)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker)

    _, _, log = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    assert hasattr(log, 'desiredLateralJerk')

  def test_active_negates_output_torque(self, mocker):
    """Test that output torque is negated (left positive convention)."""
    CP = create_mock_cp(mocker)
    # Make torque_from_lateral_accel return positive value
    CI = create_mock_ci(mocker, torque_from_accel=lambda a, p: 0.5)
    controller = LatControlTorque(CP, CI, 0.01)
    CS = create_mock_cs(mocker, v_ego=20.0)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker)

    torque, _, log = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    # Torque should be negated
    assert torque == -0.5
    # Log output is -output_torque which is 0.5
    assert log.output == -0.5


class TestLatControlTorqueRollCompensation:
  """Test roll compensation in LatControlTorque."""

  def test_roll_affects_feedforward(self, mocker):
    """Test that roll affects feedforward calculation."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)
    CS = create_mock_cs(mocker, v_ego=20.0)
    VM = create_mock_vm(mocker)

    # First run with no roll
    params_no_roll = create_mock_params(mocker, roll=0.0)
    controller.update(
      active=True, CS=CS, VM=VM, params=params_no_roll, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )
    ff_no_roll = controller.pid.f

    # Reset and run with roll
    controller = LatControlTorque(CP, CI, 0.01)
    params_with_roll = create_mock_params(mocker, roll=0.1)
    controller.update(
      active=True, CS=CS, VM=VM, params=params_with_roll, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )
    ff_with_roll = controller.pid.f

    # Feedforward should differ due to roll compensation
    assert ff_no_roll != ff_with_roll


class TestLatControlTorqueIntegratorFreeze:
  """Test integrator freeze conditions."""

  def test_freeze_when_steer_limited_by_safety(self, mocker):
    """Test integrator freezes when steer limited by safety."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)
    CS = create_mock_cs(mocker, v_ego=20.0)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker)

    # First update to accumulate integrator
    controller.update(active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1)
    i_before = controller.pid.i

    # Update with safety limit
    controller.update(active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=True, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1)
    i_after = controller.pid.i

    assert abs(i_after - i_before) < 0.01

  def test_freeze_when_steering_pressed(self, mocker):
    """Test integrator freezes when driver is steering."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)
    CS_normal = create_mock_cs(mocker, v_ego=20.0, steering_pressed=False)
    CS_pressed = create_mock_cs(mocker, v_ego=20.0, steering_pressed=True)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker)

    # First update
    controller.update(
      active=True, CS=CS_normal, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )
    i_before = controller.pid.i

    # Update with steering pressed
    controller.update(
      active=True, CS=CS_pressed, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )
    i_after = controller.pid.i

    assert abs(i_after - i_before) < 0.01

  def test_freeze_when_low_speed(self, mocker):
    """Test integrator freezes at low speed."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)
    CS_fast = create_mock_cs(mocker, v_ego=20.0)
    CS_slow = create_mock_cs(mocker, v_ego=3.0)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker)

    # First update at normal speed
    controller.update(
      active=True, CS=CS_fast, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )
    i_before = controller.pid.i

    # Update at low speed
    controller.update(
      active=True, CS=CS_slow, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )
    i_after = controller.pid.i

    assert abs(i_after - i_before) < 0.01


class TestLatControlTorqueSaturation:
  """Test saturation detection."""

  def test_saturation_when_output_at_limit(self, mocker):
    """Test saturation detected when output at steer_max."""
    CP = create_mock_cp(mocker)
    # Return high torque to force saturation
    CI = create_mock_ci(mocker, torque_from_accel=lambda a, p: 1.0)
    controller = LatControlTorque(CP, CI, 0.01)
    CS = create_mock_cs(mocker, v_ego=15.0)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker)

    controller.update(active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.1, curvature_limited=False, lat_delay=0.1)

    assert controller.sat_time > 0

  def test_curvature_limited_causes_saturation(self, mocker):
    """Test curvature_limited flag causes saturation."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)
    CS = create_mock_cs(mocker, v_ego=15.0)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker)

    controller.update(active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.0, curvature_limited=True, lat_delay=0.1)

    assert controller.sat_time > 0

  def test_saturation_log_reflects_state(self, mocker):
    """Test log.saturated reflects saturation state."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker, torque_from_accel=lambda a, p: 1.0)
    controller = LatControlTorque(CP, CI, 0.01)
    controller.sat_time = controller.sat_limit - 0.001
    CS = create_mock_cs(mocker, v_ego=15.0)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker)

    _, _, log = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.1, curvature_limited=False, lat_delay=0.1
    )

    assert log.saturated is True


class TestLatControlTorqueConstants:
  """Test module constants."""

  def test_kp_value(self):
    """Test KP constant value."""
    assert KP == 0.8

  def test_ki_value(self):
    """Test KI constant value."""
    assert KI == 0.15

  def test_interp_speeds_length(self):
    """Test INTERP_SPEEDS has correct length."""
    assert len(INTERP_SPEEDS) == 9
    assert len(KP_INTERP) == 9

  def test_version_value(self):
    """Test VERSION constant value."""
    assert VERSION == 1

  def test_jerk_gain_value(self):
    """Test JERK_GAIN constant value."""
    assert JERK_GAIN == 0.3
