"""Property-based tests for LatControlAngle using Hypothesis.

These tests verify mathematical properties that should hold for all valid inputs,
complementing the unit tests with broader input coverage.
"""

import math

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle

HYPOTHESIS_SETTINGS = settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])


def create_mock_cp(mocker, brand="toyota"):
  """Create a mock CarParams for LatControlAngle."""
  CP = mocker.MagicMock()
  CP.steerLimitTimer = 4.0
  CP.brand = brand
  return CP


def create_mock_ci(mocker):
  """Create a mock CarInterface."""
  CI = mocker.MagicMock()
  return CI


def create_mock_cs(mocker, v_ego=15.0, steering_angle_deg=0.0, steering_pressed=False):
  """Create a mock CarState."""
  CS = mocker.MagicMock()
  CS.vEgo = v_ego
  CS.steeringAngleDeg = steering_angle_deg
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


class TestLatControlAngleProperties:
  """Property-based tests for LatControlAngle."""

  @given(
    v_ego=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
    steering_angle=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
    desired_curvature=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_torque_output_always_zero(self, mocker, v_ego, steering_angle, desired_curvature):
    """Property: LatControlAngle always returns 0 torque (angle-based control)."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlAngle(CP, CI, 0.01)

    CS = create_mock_cs(mocker, v_ego=v_ego, steering_angle_deg=steering_angle)
    VM = create_mock_vm(mocker, steer_from_curvature=desired_curvature * 10)
    params = create_mock_params(mocker)

    torque, angle, _ = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=desired_curvature, curvature_limited=False, lat_delay=0.1
    )

    assert torque == 0.0, f"Torque should always be 0, got {torque}"

  @given(
    v_ego=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
    steering_angle=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_inactive_returns_current_angle(self, mocker, v_ego, steering_angle):
    """Property: Inactive mode returns current steering angle."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlAngle(CP, CI, 0.01)

    CS = create_mock_cs(mocker, v_ego=v_ego, steering_angle_deg=steering_angle)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker)

    torque, angle, log = controller.update(
      active=False, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    assert torque == 0.0
    assert angle == steering_angle, f"Inactive should return current angle {steering_angle}, got {angle}"
    assert log.active is False

  @given(
    angle_offset=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    steer_from_curvature=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_angle_offset_applied(self, mocker, angle_offset, steer_from_curvature):
    """Property: Angle offset is consistently applied to output."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlAngle(CP, CI, 0.01)

    CS = create_mock_cs(mocker, v_ego=20.0, steering_angle_deg=0.0)
    VM = create_mock_vm(mocker, steer_from_curvature=steer_from_curvature)
    params = create_mock_params(mocker, angle_offset_deg=angle_offset)

    _, angle, _ = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    expected = math.degrees(steer_from_curvature) + angle_offset
    assert abs(angle - expected) < 0.01, f"Angle {angle} != expected {expected}"

  @given(
    v_ego=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_log_values_finite(self, mocker, v_ego):
    """Property: All log values are finite numbers."""
    CP = create_mock_cp(mocker)
    CI = create_mock_ci(mocker)
    controller = LatControlAngle(CP, CI, 0.01)

    CS = create_mock_cs(mocker, v_ego=v_ego, steering_angle_deg=5.0)
    VM = create_mock_vm(mocker, steer_from_curvature=0.1)
    params = create_mock_params(mocker)

    _, _, log = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    assert math.isfinite(log.steeringAngleDeg)
    assert math.isfinite(log.steeringAngleDesiredDeg)

  @given(
    brand=st.sampled_from(["toyota", "tesla", "honda", "hyundai"]),
  )
  @HYPOTHESIS_SETTINGS
  def test_works_with_different_brands(self, mocker, brand):
    """Property: Controller works correctly with different car brands."""
    CP = create_mock_cp(mocker, brand=brand)
    CI = create_mock_ci(mocker)
    controller = LatControlAngle(CP, CI, 0.01)

    CS = create_mock_cs(mocker, v_ego=20.0, steering_angle_deg=0.0)
    VM = create_mock_vm(mocker, steer_from_curvature=0.05)
    params = create_mock_params(mocker)

    torque, angle, log = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    assert torque == 0.0
    assert math.isfinite(angle)
    assert log.active is True
