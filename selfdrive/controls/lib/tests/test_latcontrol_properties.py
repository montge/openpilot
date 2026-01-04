"""Property-based tests for lateral control using Hypothesis.

These tests verify mathematical properties that should hold for all valid inputs,
complementing the unit tests with broader input coverage.
"""

import math

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID

# Suppress health check for function-scoped fixtures (mocker)
# This is safe because we create fresh mocks for each example
HYPOTHESIS_SETTINGS = settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])


def create_mock_cp(mocker, kp=0.5, ki=0.1, kf=1.0):
  """Create a mock CarParams for LatControlPID."""
  CP = mocker.MagicMock()
  CP.steerLimitTimer = 4.0
  CP.lateralTuning.pid.kpBP = [0.0]
  CP.lateralTuning.pid.kpV = [kp]
  CP.lateralTuning.pid.kiBP = [0.0]
  CP.lateralTuning.pid.kiV = [ki]
  CP.lateralTuning.pid.kf = kf
  return CP


def create_mock_ci(mocker, feedforward_value=0.0):
  """Create a mock CarInterface."""
  CI = mocker.MagicMock()
  feedforward_fn = mocker.MagicMock(return_value=feedforward_value)
  CI.get_steer_feedforward_function.return_value = feedforward_fn
  return CI


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


class TestLatControlPIDProperties:
  """Property-based tests for LatControlPID."""

  @given(
    v_ego=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    steering_angle=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
    desired_curvature=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_output_always_bounded(self, mocker, v_ego, steering_angle, desired_curvature):
    """Property: Output torque is always within [-1, 1]."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlPID(CP, CI, 0.01)

    CS = create_mock_cs(mocker, v_ego=v_ego, steering_angle_deg=steering_angle)
    VM = create_mock_vm(mocker, steer_from_curvature=desired_curvature * 10)
    params = create_mock_params(mocker)

    torque, _, _ = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=desired_curvature, curvature_limited=False, lat_delay=0.1
    )

    assert -1.0 <= torque <= 1.0, f"Torque {torque} out of bounds"

  @given(
    v_ego=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    steering_angle=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_inactive_always_zero_torque(self, mocker, v_ego, steering_angle):
    """Property: Inactive mode always returns zero torque."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlPID(CP, CI, 0.01)

    CS = create_mock_cs(mocker, v_ego=v_ego, steering_angle_deg=steering_angle)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker)

    torque, _, log = controller.update(
      active=False, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    assert torque == 0.0, f"Inactive mode returned non-zero torque: {torque}"
    assert log.active is False

  @given(
    kp=st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False),
    ki=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_output_bounded_with_varying_gains(self, mocker, kp, ki):
    """Property: Output remains bounded regardless of PID gains."""
    CP = create_mock_cp(mocker, kp=kp, ki=ki)
    CI = create_mock_ci(mocker)
    controller = LatControlPID(CP, CI, 0.01)

    CS = create_mock_cs(mocker, v_ego=20.0, steering_angle_deg=0.0)
    VM = create_mock_vm(mocker, steer_from_curvature=0.5)  # Large desired angle
    params = create_mock_params(mocker)

    # Run multiple updates to build up integrator
    for _ in range(10):
      torque, _, _ = controller.update(
        active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.05, curvature_limited=False, lat_delay=0.1
      )
      assert -1.0 <= torque <= 1.0, f"Torque {torque} out of bounds with kp={kp}, ki={ki}"

  @given(
    angle_offset=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_angle_offset_applied_correctly(self, mocker, angle_offset):
    """Property: Angle offset is consistently applied to desired angle."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlPID(CP, CI, 0.01)

    steer_from_curvature = 0.1
    CS = create_mock_cs(mocker, v_ego=20.0, steering_angle_deg=0.0)
    VM = create_mock_vm(mocker, steer_from_curvature=steer_from_curvature)
    params = create_mock_params(mocker, angle_offset_deg=angle_offset)

    _, angle, log = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    expected = math.degrees(steer_from_curvature) + angle_offset
    assert abs(angle - expected) < 0.01, f"Angle {angle} != expected {expected}"

  @given(
    v_ego=st.floats(min_value=1.0, max_value=40.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_log_values_finite(self, mocker, v_ego):
    """Property: All log values are finite numbers."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlPID(CP, CI, 0.01)

    CS = create_mock_cs(mocker, v_ego=v_ego, steering_angle_deg=5.0)
    VM = create_mock_vm(mocker, steer_from_curvature=0.1)
    params = create_mock_params(mocker)

    _, _, log = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    assert math.isfinite(log.steeringAngleDeg)
    assert math.isfinite(log.steeringAngleDesiredDeg)
    assert math.isfinite(log.angleError)
    assert math.isfinite(log.p)
    assert math.isfinite(log.i)
    assert math.isfinite(log.f)
    assert math.isfinite(log.output)
