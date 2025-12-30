"""Tests for selfdrive/controls/lib/latcontrol_angle.py - angle-based lateral control."""

import math

from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle, STEER_ANGLE_SATURATION_THRESHOLD


def create_mock_cp(mocker, brand="honda"):
  """Create a mock CarParams for LatControlAngle."""
  CP = mocker.MagicMock()
  CP.steerLimitTimer = 4.0
  CP.brand = brand
  return CP


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


class TestLatControlAngleInit:
  """Test LatControlAngle initialization."""

  def test_init_default_brand(self, mocker):
    """Test initialization with non-Tesla brand."""
    CP = create_mock_cp(mocker, brand="honda")
    CI = mocker.MagicMock()

    controller = LatControlAngle(CP, CI, 0.01)

    assert controller.sat_check_min_speed == 5.0
    assert controller.use_steer_limited_by_safety is False

  def test_init_tesla_brand(self, mocker):
    """Test initialization with Tesla brand."""
    CP = create_mock_cp(mocker, brand="tesla")
    CI = mocker.MagicMock()

    controller = LatControlAngle(CP, CI, 0.01)

    assert controller.sat_check_min_speed == 5.0
    assert controller.use_steer_limited_by_safety is True

  def test_init_inherits_from_base(self, mocker):
    """Test that base class attributes are initialized."""
    CP = create_mock_cp(mocker)
    CI = mocker.MagicMock()

    controller = LatControlAngle(CP, CI, 0.01)

    assert controller.dt == 0.01
    assert controller.sat_limit == 4.0
    assert controller.steer_max == 1.0


class TestLatControlAngleUpdateInactive:
  """Test LatControlAngle update when inactive."""

  def test_inactive_returns_current_angle(self, mocker):
    """Test that inactive update returns current steering angle."""
    CP = create_mock_cp(mocker)
    controller = LatControlAngle(CP, mocker.MagicMock(), 0.01)
    CS = create_mock_cs(mocker, steering_angle_deg=15.5)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker)

    torque, angle, log = controller.update(
      active=False, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    assert torque == 0
    assert angle == 15.5
    assert log.active is False

  def test_inactive_log_contains_steering_angle(self, mocker):
    """Test that inactive log contains current steering angle."""
    CP = create_mock_cp(mocker)
    controller = LatControlAngle(CP, mocker.MagicMock(), 0.01)
    CS = create_mock_cs(mocker, steering_angle_deg=10.0)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker)

    _, _, log = controller.update(
      active=False, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.0, curvature_limited=False, lat_delay=0.1
    )

    assert log.steeringAngleDeg == 10.0
    assert log.steeringAngleDesiredDeg == 10.0


class TestLatControlAngleUpdateActive:
  """Test LatControlAngle update when active."""

  def test_active_calculates_desired_angle(self, mocker):
    """Test that active update calculates desired angle from curvature."""
    CP = create_mock_cp(mocker)
    controller = LatControlAngle(CP, mocker.MagicMock(), 0.01)
    CS = create_mock_cs(mocker, v_ego=20.0, steering_angle_deg=0.0)
    # Return 0.1 radians from curvature calculation
    VM = create_mock_vm(mocker, steer_from_curvature=0.1)
    params = create_mock_params(mocker, angle_offset_deg=0.0)

    torque, angle, log = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    assert torque == 0
    # 0.1 radians = 5.729... degrees
    assert abs(angle - math.degrees(0.1)) < 0.001
    assert log.active is True

  def test_active_applies_angle_offset(self, mocker):
    """Test that angle offset is applied to desired angle."""
    CP = create_mock_cp(mocker)
    controller = LatControlAngle(CP, mocker.MagicMock(), 0.01)
    CS = create_mock_cs(mocker)
    VM = create_mock_vm(mocker, steer_from_curvature=0.0)
    params = create_mock_params(mocker, angle_offset_deg=2.5)

    _, angle, _ = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.0, curvature_limited=False, lat_delay=0.1
    )

    assert angle == 2.5

  def test_active_passes_roll_to_vm(self, mocker):
    """Test that roll is passed to vehicle model."""
    CP = create_mock_cp(mocker)
    controller = LatControlAngle(CP, mocker.MagicMock(), 0.01)
    CS = create_mock_cs(mocker, v_ego=15.0)
    VM = create_mock_vm(mocker)
    params = create_mock_params(mocker, roll=0.05)

    controller.update(active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1)

    # Verify VM was called with correct parameters
    VM.get_steer_from_curvature.assert_called_once()
    call_args = VM.get_steer_from_curvature.call_args[0]
    assert call_args[0] == -0.01  # -desired_curvature
    assert call_args[1] == 15.0  # v_ego
    assert call_args[2] == 0.05  # roll

  def test_active_log_contains_angles(self, mocker):
    """Test that active log contains steering angles."""
    CP = create_mock_cp(mocker)
    controller = LatControlAngle(CP, mocker.MagicMock(), 0.01)
    CS = create_mock_cs(mocker, steering_angle_deg=5.0)
    VM = create_mock_vm(mocker, steer_from_curvature=0.2)
    params = create_mock_params(mocker, angle_offset_deg=1.0)

    _, _, log = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    assert log.steeringAngleDeg == 5.0
    expected_desired = math.degrees(0.2) + 1.0
    assert abs(log.steeringAngleDesiredDeg - expected_desired) < 0.001


class TestLatControlAngleSaturation:
  """Test LatControlAngle saturation detection."""

  def test_tesla_uses_steer_limited_by_safety(self, mocker):
    """Test Tesla uses steer_limited_by_safety for saturation."""
    CP = create_mock_cp(mocker, brand="tesla")
    controller = LatControlAngle(CP, mocker.MagicMock(), 0.01)
    CS = create_mock_cs(mocker, v_ego=15.0, steering_angle_deg=0.0)
    VM = create_mock_vm(mocker, steer_from_curvature=0.0)
    params = create_mock_params(mocker)

    # With steer_limited_by_safety=True, should detect saturation
    controller.update(active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=True, desired_curvature=0.0, curvature_limited=False, lat_delay=0.1)

    # sat_time should increase
    assert controller.sat_time > 0

  def test_tesla_no_saturation_when_not_limited(self, mocker):
    """Test Tesla no saturation when not limited by safety."""
    CP = create_mock_cp(mocker, brand="tesla")
    controller = LatControlAngle(CP, mocker.MagicMock(), 0.01)
    CS = create_mock_cs(mocker, v_ego=15.0, steering_angle_deg=0.0)
    VM = create_mock_vm(mocker, steer_from_curvature=0.0)
    params = create_mock_params(mocker)

    controller.update(active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.0, curvature_limited=False, lat_delay=0.1)

    assert controller.sat_time == 0

  def test_non_tesla_uses_angle_threshold(self, mocker):
    """Test non-Tesla uses angle difference threshold for saturation."""
    CP = create_mock_cp(mocker, brand="honda")
    controller = LatControlAngle(CP, mocker.MagicMock(), 0.01)
    CS = create_mock_cs(mocker, v_ego=15.0, steering_angle_deg=0.0)
    # Return angle that exceeds threshold
    large_angle_rad = math.radians(STEER_ANGLE_SATURATION_THRESHOLD + 1.0)
    VM = create_mock_vm(mocker, steer_from_curvature=large_angle_rad)
    params = create_mock_params(mocker)

    controller.update(active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1)

    # sat_time should increase due to large angle difference
    assert controller.sat_time > 0

  def test_non_tesla_no_saturation_within_threshold(self, mocker):
    """Test non-Tesla no saturation when within angle threshold."""
    CP = create_mock_cp(mocker, brand="honda")
    controller = LatControlAngle(CP, mocker.MagicMock(), 0.01)
    CS = create_mock_cs(mocker, v_ego=15.0, steering_angle_deg=0.0)
    # Return small angle within threshold
    small_angle_rad = math.radians(STEER_ANGLE_SATURATION_THRESHOLD - 1.0)
    VM = create_mock_vm(mocker, steer_from_curvature=small_angle_rad)
    params = create_mock_params(mocker)

    controller.update(active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1)

    assert controller.sat_time == 0

  def test_curvature_limited_causes_saturation(self, mocker):
    """Test curvature_limited flag causes saturation."""
    CP = create_mock_cp(mocker, brand="honda")
    controller = LatControlAngle(CP, mocker.MagicMock(), 0.01)
    CS = create_mock_cs(mocker, v_ego=15.0, steering_angle_deg=0.0)
    VM = create_mock_vm(mocker, steer_from_curvature=0.0)
    params = create_mock_params(mocker)

    controller.update(active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.0, curvature_limited=True, lat_delay=0.1)

    assert controller.sat_time > 0

  def test_saturation_log_reflects_state(self, mocker):
    """Test log.saturated reflects saturation state."""
    CP = create_mock_cp(mocker, brand="honda")
    controller = LatControlAngle(CP, mocker.MagicMock(), 0.01)
    # Set sat_time near limit
    controller.sat_time = controller.sat_limit - 0.001
    CS = create_mock_cs(mocker, v_ego=15.0, steering_angle_deg=0.0)
    large_angle_rad = math.radians(STEER_ANGLE_SATURATION_THRESHOLD + 1.0)
    VM = create_mock_vm(mocker, steer_from_curvature=large_angle_rad)
    params = create_mock_params(mocker)

    _, _, log = controller.update(
      active=True, CS=CS, VM=VM, params=params, steer_limited_by_safety=False, desired_curvature=0.01, curvature_limited=False, lat_delay=0.1
    )

    assert log.saturated is True


class TestLatControlAngleConstants:
  """Test module constants."""

  def test_saturation_threshold_value(self):
    """Test STEER_ANGLE_SATURATION_THRESHOLD has expected value."""
    assert STEER_ANGLE_SATURATION_THRESHOLD == 2.5
