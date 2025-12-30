"""Tests for selfdrive/controls/lib/latcontrol_pid.py - PID-based lateral control."""

import math

from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID


def create_mock_cp(mocker):
  """Create a mock CarParams for LatControlPID."""
  CP = mocker.MagicMock()
  CP.steerLimitTimer = 4.0
  CP.lateralTuning.pid.kpBP = [0.0]
  CP.lateralTuning.pid.kpV = [0.5]
  CP.lateralTuning.pid.kiBP = [0.0]
  CP.lateralTuning.pid.kiV = [0.1]
  CP.lateralTuning.pid.kf = 1.0
  return CP


def create_mock_ci(mocker, feedforward_value=0.0):
  """Create a mock CarInterface."""
  CI = mocker.MagicMock()
  feedforward_fn = mocker.MagicMock(return_value=feedforward_value)
  CI.get_steer_feedforward_function.return_value = feedforward_fn
  return CI, feedforward_fn


def create_mock_cs(mocker, v_ego=15.0, steering_angle_deg=0.0, steering_rate_deg=0.0, steering_pressed=False):
  """Create a mock CarState."""
  CS = mocker.MagicMock()
  CS.vEgo = v_ego
  CS.steeringAngleDeg = steering_angle_deg
  CS.steeringRateDeg = steering_rate_deg
  CS.steeringPressed = steering_pressed
  return CS


def create_mock_vm(mocker, steer_from_curvature=0.0):
  """Create a mock VehicleModel."""
  VM = mocker.MagicMock()
  VM.get_steer_from_curvature.return_value = steer_from_curvature
  return VM


def create_mock_params(mocker, angle_offset_deg=0.0, roll=0.0):
  """Create mock calibration params."""
  params = mocker.MagicMock()
  params.angleOffsetDeg = angle_offset_deg
  params.roll = roll
  return params


class TestLatControlPIDInit:
  """Test LatControlPID initialization."""

  def test_init_creates_pid_controller(self, mocker):
    """Test initialization creates PID controller with correct params."""
    CP = create_mock_cp(mocker)
    CI, _ = create_mock_ci(mocker)

    controller = LatControlPID(CP, CI, 0.01)

    assert controller.pid is not None
    assert controller.pid.pos_limit == 1.0
    assert controller.pid.neg_limit == -1.0

  def test_init_stores_ff_factor(self, mocker):
    """Test initialization stores feedforward factor."""
    CP = create_mock_cp(mocker)
    CP.lateralTuning.pid.kf = 2.5
    CI, _ = create_mock_ci(mocker)

    controller = LatControlPID(CP, CI, 0.01)

    assert controller.ff_factor == 2.5

  def test_init_gets_feedforward_function(self, mocker):
    """Test initialization gets feedforward function from CI."""
    CP = create_mock_cp(mocker)
    CI, ff_fn = create_mock_ci(mocker)

    controller = LatControlPID(CP, CI, 0.01)

    CI.get_steer_feedforward_function.assert_called_once()
    assert controller.get_steer_feedforward == ff_fn

  def test_init_inherits_from_base(self, mocker):
    """Test that base class attributes are initialized."""
    CP = create_mock_cp(mocker)
    CI, _ = create_mock_ci(mocker)

    controller = LatControlPID(CP, CI, 0.01)

    assert controller.dt == 0.01
    assert controller.sat_limit == 4.0
    assert controller.steer_max == 1.0


class TestLatControlPIDUpdateInactive:
  """Test LatControlPID update when inactive."""

  def test_inactive_returns_zero_torque(self, mocker):
    """Test that inactive update returns zero torque."""
    CP = create_mock_cp(mocker)
    CI, _ = create_mock_ci(mocker)
    controller = LatControlPID(CP, CI, 0.01)
    CS = create_mock_cs(mocker, steering_angle_deg=10.0)
    VM = create_mock_vm(mocker, steer_from_curvature=0.1)
    params = create_mock_params(mocker)

    torque, angle, log = controller.update(
      active=False, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    assert torque == 0.0
    assert log.active is False

  def test_inactive_still_calculates_desired_angle(self, mocker):
    """Test that inactive update still calculates desired angle for logging."""
    CP = create_mock_cp(mocker)
    CI, _ = create_mock_ci(mocker)
    controller = LatControlPID(CP, CI, 0.01)
    CS = create_mock_cs(mocker, steering_angle_deg=5.0)
    VM = create_mock_vm(mocker, steer_from_curvature=0.2)
    params = create_mock_params(mocker, angle_offset_deg=1.0)

    _, angle, log = controller.update(
      active=False, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    expected_angle = math.degrees(0.2) + 1.0
    assert abs(angle - expected_angle) < 0.001
    assert abs(log.steeringAngleDesiredDeg - expected_angle) < 0.001

  def test_inactive_log_contains_current_state(self, mocker):
    """Test that inactive log contains current steering state."""
    CP = create_mock_cp(mocker)
    CI, _ = create_mock_ci(mocker)
    controller = LatControlPID(CP, CI, 0.01)
    CS = create_mock_cs(mocker, steering_angle_deg=15.0, steering_rate_deg=2.5)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker)

    _, _, log = controller.update(
      active=False, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.0, curvature_limited=False, lat_delay=0.1
    )

    assert log.steeringAngleDeg == 15.0
    assert log.steeringRateDeg == 2.5

  def test_inactive_log_contains_error(self, mocker):
    """Test that inactive log contains angle error."""
    CP = create_mock_cp(mocker)
    CI, _ = create_mock_ci(mocker)
    controller = LatControlPID(CP, CI, 0.01)
    CS = create_mock_cs(mocker, steering_angle_deg=10.0)
    VM = create_mock_vm(mocker, steer_from_curvature=0.0)
    params = create_mock_params(mocker, angle_offset_deg=5.0)

    _, _, log = controller.update(
      active=False, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.0, curvature_limited=False, lat_delay=0.1
    )

    # desired = 0 + 5 = 5, current = 10, error = 5 - 10 = -5
    assert log.angleError == -5.0


class TestLatControlPIDUpdateActive:
  """Test LatControlPID update when active."""

  def test_active_returns_pid_output(self, mocker):
    """Test that active update returns PID controller output."""
    CP = create_mock_cp(mocker)
    CI, _ = create_mock_ci(mocker, feedforward_value=0.0)
    controller = LatControlPID(CP, CI, 0.01)
    CS = create_mock_cs(mocker, v_ego=20.0, steering_angle_deg=0.0)
    VM = create_mock_vm(mocker, steer_from_curvature=0.1)
    params = create_mock_params(mocker)

    torque, _, log = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    assert log.active is True
    # With error and P gain, should have some output
    assert torque != 0.0 or controller.pid.p != 0.0

  def test_active_applies_feedforward(self, mocker):
    """Test that feedforward is applied in active mode."""
    CP = create_mock_cp(mocker)
    CP.lateralTuning.pid.kf = 2.0
    CI, ff_fn = create_mock_ci(mocker, feedforward_value=0.5)
    controller = LatControlPID(CP, CI, 0.01)
    CS = create_mock_cs(mocker, v_ego=20.0, steering_angle_deg=0.0)
    VM = create_mock_vm(mocker, steer_from_curvature=0.1)
    params = create_mock_params(mocker)

    controller.update(active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1)

    # Feedforward function should be called with angle (no offset) and speed
    ff_fn.assert_called_once()
    call_args = ff_fn.call_args[0]
    assert abs(call_args[0] - math.degrees(0.1)) < 0.001  # angle_steers_des_no_offset
    assert call_args[1] == 20.0  # v_ego

  def test_active_log_contains_pid_values(self, mocker):
    """Test that active log contains PID component values."""
    CP = create_mock_cp(mocker)
    CI, _ = create_mock_ci(mocker)
    controller = LatControlPID(CP, CI, 0.01)
    CS = create_mock_cs(mocker, v_ego=20.0, steering_angle_deg=0.0)
    VM = create_mock_vm(mocker, steer_from_curvature=0.1)
    params = create_mock_params(mocker)

    _, _, log = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    assert hasattr(log, 'p')
    assert hasattr(log, 'i')
    assert hasattr(log, 'f')
    assert hasattr(log, 'output')


class TestLatControlPIDIntegratorFreeze:
  """Test LatControlPID integrator freeze conditions."""

  def test_freeze_when_steer_limited_by_safety(self, mocker):
    """Test integrator freezes when steer limited by safety."""
    CP = create_mock_cp(mocker)
    CI, _ = create_mock_ci(mocker)
    controller = LatControlPID(CP, CI, 0.01)
    CS = create_mock_cs(mocker, v_ego=20.0, steering_angle_deg=0.0)
    VM = create_mock_vm(mocker, steer_from_curvature=0.1)
    params = create_mock_params(mocker)

    # First update to accumulate some integrator
    controller.update(active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1)
    i_before = controller.pid.i

    # Update with safety limit - integrator should freeze
    controller.update(active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=True, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1)
    i_after = controller.pid.i

    # When frozen, integrator should not change significantly
    assert abs(i_after - i_before) < 0.01

  def test_freeze_when_steering_pressed(self, mocker):
    """Test integrator freezes when driver is steering."""
    CP = create_mock_cp(mocker)
    CI, _ = create_mock_ci(mocker)
    controller = LatControlPID(CP, CI, 0.01)
    CS_normal = create_mock_cs(mocker, v_ego=20.0, steering_angle_deg=0.0, steering_pressed=False)
    CS_pressed = create_mock_cs(mocker, v_ego=20.0, steering_angle_deg=0.0, steering_pressed=True)
    VM = create_mock_vm(mocker, steer_from_curvature=0.1)
    params = create_mock_params(mocker)

    # First update to accumulate some integrator
    controller.update(
      active=True, CS=CS_normal, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )
    i_before = controller.pid.i

    # Update with steering pressed - integrator should freeze
    controller.update(
      active=True, CS=CS_pressed, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )
    i_after = controller.pid.i

    assert abs(i_after - i_before) < 0.01

  def test_freeze_when_low_speed(self, mocker):
    """Test integrator freezes at low speed."""
    CP = create_mock_cp(mocker)
    CI, _ = create_mock_ci(mocker)
    controller = LatControlPID(CP, CI, 0.01)
    CS_fast = create_mock_cs(mocker, v_ego=20.0, steering_angle_deg=0.0)
    CS_slow = create_mock_cs(mocker, v_ego=3.0, steering_angle_deg=0.0)  # Below 5 m/s
    VM = create_mock_vm(mocker, steer_from_curvature=0.1)
    params = create_mock_params(mocker)

    # First update at normal speed
    controller.update(
      active=True, CS=CS_fast, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )
    i_before = controller.pid.i

    # Update at low speed - integrator should freeze
    controller.update(
      active=True, CS=CS_slow, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )
    i_after = controller.pid.i

    assert abs(i_after - i_before) < 0.01


class TestLatControlPIDSaturation:
  """Test LatControlPID saturation detection."""

  def test_saturation_when_output_at_limit(self, mocker):
    """Test saturation detected when output at steer_max."""
    CP = create_mock_cp(mocker)
    CI, _ = create_mock_ci(mocker)
    controller = LatControlPID(CP, CI, 0.01)
    CS = create_mock_cs(mocker, v_ego=15.0, steering_angle_deg=0.0)
    # Large curvature to force high output
    VM = create_mock_vm(mocker, steer_from_curvature=1.0)  # ~57 degrees
    params = create_mock_params(mocker)

    # Run several updates to build up output
    for _ in range(100):
      controller.update(active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.1, curvature_limited=False, lat_delay=0.1)

    # sat_time should have increased
    assert controller.sat_time > 0

  def test_no_saturation_when_output_low(self, mocker):
    """Test no saturation when output is well below limit."""
    CP = create_mock_cp(mocker)
    CI, _ = create_mock_ci(mocker)
    controller = LatControlPID(CP, CI, 0.01)
    CS = create_mock_cs(mocker, v_ego=15.0, steering_angle_deg=0.0)
    VM = create_mock_vm(mocker, steer_from_curvature=0.0)
    params = create_mock_params(mocker)

    controller.update(active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.0, curvature_limited=False, lat_delay=0.1)

    assert controller.sat_time == 0

  def test_curvature_limited_causes_saturation(self, mocker):
    """Test curvature_limited flag causes saturation."""
    CP = create_mock_cp(mocker)
    CI, _ = create_mock_ci(mocker)
    controller = LatControlPID(CP, CI, 0.01)
    CS = create_mock_cs(mocker, v_ego=15.0, steering_angle_deg=0.0)
    VM = create_mock_vm(mocker, steer_from_curvature=0.0)
    params = create_mock_params(mocker)

    controller.update(active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.0, curvature_limited=True, lat_delay=0.1)

    assert controller.sat_time > 0

  def test_saturation_log_reflects_state(self, mocker):
    """Test log.saturated reflects saturation state."""
    CP = create_mock_cp(mocker)
    CI, _ = create_mock_ci(mocker)
    controller = LatControlPID(CP, CI, 0.01)
    controller.sat_time = controller.sat_limit - 0.001
    CS = create_mock_cs(mocker, v_ego=15.0, steering_angle_deg=0.0)
    VM = create_mock_vm(mocker, steer_from_curvature=0.0)
    params = create_mock_params(mocker)

    _, _, log = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.0, curvature_limited=True, lat_delay=0.1
    )

    assert log.saturated is True
