"""Property-based tests for LatControlTorque using Hypothesis.

These tests verify mathematical properties that should hold for all valid inputs,
complementing the unit tests with broader input coverage.
"""

import math

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque

HYPOTHESIS_SETTINGS = settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])


def create_mock_cp(mocker):
  """Create a mock CarParams for LatControlTorque."""
  CP = mocker.MagicMock()
  CP.steerLimitTimer = 4.0

  # Torque tuning params
  torque_params = mocker.MagicMock()
  torque_params.latAccelFactor = 2.0
  torque_params.latAccelOffset = 0.0
  torque_params.friction = 0.1
  torque_params.steeringAngleDeadzoneDeg = 0.0
  torque_params.as_builder.return_value = torque_params

  CP.lateralTuning.torque = torque_params
  return CP


def create_mock_ci(mocker):
  """Create a mock CarInterface for LatControlTorque."""
  CI = mocker.MagicMock()

  # torque_from_lateral_accel: converts lateral accel to torque
  def torque_from_lat_accel(lat_accel, params):
    return max(-1.0, min(1.0, lat_accel / 3.0))

  # lateral_accel_from_torque: converts torque to lateral accel
  def lat_accel_from_torque(torque, params):
    return torque * 3.0

  CI.torque_from_lateral_accel.return_value = torque_from_lat_accel
  CI.lateral_accel_from_torque.return_value = lat_accel_from_torque
  return CI


def create_mock_cs(mocker, v_ego=15.0, steering_angle_deg=0.0, steering_pressed=False):
  """Create a mock CarState."""
  CS = mocker.MagicMock()
  CS.vEgo = v_ego
  CS.steeringAngleDeg = steering_angle_deg
  CS.steeringPressed = steering_pressed
  return CS


def create_mock_vm(mocker, calc_curvature_value=0.0):
  """Create a mock VehicleModel."""
  VM = mocker.MagicMock()
  VM.calc_curvature.return_value = calc_curvature_value
  return VM


def create_mock_params(mocker, angle_offset_deg=0.0, roll=0.0):
  """Create mock calibration params."""
  params = mocker.MagicMock()
  params.angleOffsetDeg = angle_offset_deg
  params.roll = roll
  return params


class TestLatControlTorqueProperties:
  """Property-based tests for LatControlTorque."""

  @given(
    v_ego=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    steering_angle=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
    desired_curvature=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_output_always_bounded(self, mocker, v_ego, steering_angle, desired_curvature):
    """Property: Output torque is always within [-1, 1]."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)

    CS = create_mock_cs(mocker, v_ego=v_ego, steering_angle_deg=steering_angle)
    VM = create_mock_vm(mocker, calc_curvature_value=desired_curvature)
    params = create_mock_params(mocker)

    torque, _, _ = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=desired_curvature, curvature_limited=False, lat_delay=0.1
    )

    assert -1.0 <= torque <= 1.0, f"Torque {torque} out of bounds"

  @given(
    v_ego=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    steering_angle=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_inactive_always_zero_torque(self, mocker, v_ego, steering_angle):
    """Property: Inactive mode always returns zero torque."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)

    CS = create_mock_cs(mocker, v_ego=v_ego, steering_angle_deg=steering_angle)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker)

    torque, _, log = controller.update(
      active=False, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    assert torque == 0.0, f"Inactive mode returned non-zero torque: {torque}"
    assert log.active is False

  @given(
    v_ego=st.floats(min_value=5.0, max_value=40.0, allow_nan=False, allow_infinity=False),
    roll=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_log_values_finite(self, mocker, v_ego, roll):
    """Property: All log values are finite numbers."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)

    CS = create_mock_cs(mocker, v_ego=v_ego, steering_angle_deg=5.0)
    VM = create_mock_vm(mocker, calc_curvature_value=0.01)
    params = create_mock_params(mocker, roll=roll)

    _, _, log = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    assert math.isfinite(log.error)
    assert math.isfinite(log.p)
    assert math.isfinite(log.i)
    assert math.isfinite(log.f)
    assert math.isfinite(log.output)
    assert math.isfinite(log.actualLateralAccel)
    assert math.isfinite(log.desiredLateralAccel)

  @given(
    lat_delay=st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_varying_delay_still_bounded(self, mocker, lat_delay):
    """Property: Output remains bounded with varying latency compensation."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlTorque(CP, CI, 0.01)

    CS = create_mock_cs(mocker, v_ego=20.0, steering_angle_deg=0.0)
    VM = create_mock_vm(mocker, calc_curvature_value=0.05)
    params = create_mock_params(mocker)

    # Run multiple updates to fill the buffer
    for _ in range(20):
      torque, _, _ = controller.update(
        active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.05, curvature_limited=False, lat_delay=lat_delay
      )
      assert -1.0 <= torque <= 1.0, f"Torque {torque} out of bounds with delay {lat_delay}"
