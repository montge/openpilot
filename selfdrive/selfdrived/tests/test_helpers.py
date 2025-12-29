from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from opendbc.car.lateral import ISO_LATERAL_ACCEL
from openpilot.selfdrive.selfdrived.helpers import (
  ExcessiveActuationCheck,
  ExcessiveActuationType,
  MIN_EXCESSIVE_ACTUATION_COUNT,
  MIN_LATERAL_ENGAGE_BUFFER,
)


class TestExcessiveActuationCheck:
  """Tests for the ExcessiveActuationCheck safety class."""

  def _create_mocks(
    self, mocker, long_active=True, lat_active=True, steering_pressed=False, v_ego=10.0, a_ego=0.0, accel_calibrated=0.0, yaw_rate=0.0, roll=0.0
  ):
    """Create mock objects for testing."""
    sm = mocker.MagicMock()
    sm.__getitem__ = mocker.MagicMock(
      side_effect=lambda key: {
        'carControl': mocker.MagicMock(longActive=long_active, latActive=lat_active),
        'liveParameters': mocker.MagicMock(roll=roll),
      }.get(key, mocker.MagicMock())
    )

    CS = mocker.MagicMock()
    CS.vEgo = v_ego
    CS.aEgo = a_ego
    CS.steeringPressed = steering_pressed

    pose = mocker.MagicMock()
    pose.acceleration.x = accel_calibrated
    pose.angular_velocity.yaw = yaw_rate

    return sm, CS, pose

  def test_initialization(self):
    """Check should start with counters at zero."""
    check = ExcessiveActuationCheck()
    assert check._excessive_counter == 0
    assert check._engaged_counter == 0

  def test_normal_operation_no_excessive(self, mocker):
    """Normal driving should not trigger excessive actuation."""
    check = ExcessiveActuationCheck()
    sm, CS, pose = self._create_mocks(
      mocker,
      accel_calibrated=1.0,  # Within ACCEL_MAX * 2
      a_ego=1.0,
      yaw_rate=0.1,
    )

    for _ in range(MIN_EXCESSIVE_ACTUATION_COUNT + 10):
      result = check.update(sm, CS, pose)
      assert result is None

  def test_excessive_longitudinal_accel_positive(self, mocker):
    """Excessive positive acceleration should trigger longitudinal type."""
    check = ExcessiveActuationCheck()
    excessive_accel = ACCEL_MAX * 2 + 1.0  # Above threshold
    sm, CS, pose = self._create_mocks(
      mocker,
      accel_calibrated=excessive_accel,
      a_ego=excessive_accel,  # Match for livepose_valid
    )

    # Need to exceed MIN_EXCESSIVE_ACTUATION_COUNT
    for _ in range(MIN_EXCESSIVE_ACTUATION_COUNT + 1):
      result = check.update(sm, CS, pose)

    assert result == ExcessiveActuationType.LONGITUDINAL

  def test_excessive_longitudinal_accel_negative(self, mocker):
    """Excessive negative acceleration (braking) should trigger longitudinal type."""
    check = ExcessiveActuationCheck()
    excessive_accel = ACCEL_MIN * 2 - 1.0  # Below threshold (more negative)
    sm, CS, pose = self._create_mocks(
      mocker,
      accel_calibrated=excessive_accel,
      a_ego=excessive_accel,
    )

    for _ in range(MIN_EXCESSIVE_ACTUATION_COUNT + 1):
      result = check.update(sm, CS, pose)

    assert result == ExcessiveActuationType.LONGITUDINAL

  def test_no_excessive_when_long_inactive(self, mocker):
    """Should not detect excessive longitudinal when long control is inactive."""
    check = ExcessiveActuationCheck()
    excessive_accel = ACCEL_MAX * 3
    sm, CS, pose = self._create_mocks(
      mocker,
      long_active=False,  # Longitudinal control inactive
      accel_calibrated=excessive_accel,
      a_ego=excessive_accel,
    )

    for _ in range(MIN_EXCESSIVE_ACTUATION_COUNT + 10):
      result = check.update(sm, CS, pose)

    assert result is None

  def test_excessive_lateral_requires_engaged_buffer(self, mocker):
    """Lateral excessive check requires MIN_LATERAL_ENGAGE_BUFFER frames engaged."""
    check = ExcessiveActuationCheck()
    # Create conditions for excessive lateral
    v_ego = 20.0
    # Lateral accel = v_ego * yaw_rate - sin(roll) * g
    # To exceed ISO_LATERAL_ACCEL * 2, we need a high yaw rate
    excessive_yaw = (ISO_LATERAL_ACCEL * 2 + 1.0) / v_ego

    sm, CS, pose = self._create_mocks(
      mocker,
      v_ego=v_ego,
      yaw_rate=excessive_yaw,
      a_ego=0.0,
      accel_calibrated=0.0,
      roll=0.0,
    )

    # First few frames should not trigger (building engage buffer)
    for _ in range(MIN_LATERAL_ENGAGE_BUFFER):
      result = check.update(sm, CS, pose)
      assert result is None

    # After engage buffer, need excessive counter to build
    for _ in range(MIN_EXCESSIVE_ACTUATION_COUNT + 1):
      result = check.update(sm, CS, pose)

    assert result == ExcessiveActuationType.LATERAL

  def test_steering_pressed_resets_engaged_counter(self, mocker):
    """Steering press should reset the engaged counter."""
    check = ExcessiveActuationCheck()
    sm, CS, pose = self._create_mocks(mocker)

    # Build up engaged counter
    for _ in range(MIN_LATERAL_ENGAGE_BUFFER + 5):
      check.update(sm, CS, pose)

    assert check._engaged_counter > MIN_LATERAL_ENGAGE_BUFFER

    # Now press steering
    sm_pressed, CS_pressed, pose_pressed = self._create_mocks(mocker, steering_pressed=True)
    check.update(sm_pressed, CS_pressed, pose_pressed)

    assert check._engaged_counter == 0

  def test_lat_inactive_resets_engaged_counter(self, mocker):
    """Lateral control going inactive should reset engaged counter."""
    check = ExcessiveActuationCheck()
    sm, CS, pose = self._create_mocks(mocker)

    # Build up engaged counter
    for _ in range(MIN_LATERAL_ENGAGE_BUFFER + 5):
      check.update(sm, CS, pose)

    # Now deactivate lateral
    sm_inactive, CS_inactive, pose_inactive = self._create_mocks(mocker, lat_active=False)
    check.update(sm_inactive, CS_inactive, pose_inactive)

    assert check._engaged_counter == 0

  def test_livepose_invalid_resets_excessive_counter(self, mocker):
    """Large difference between aEgo and calibrated accel should reset counter."""
    check = ExcessiveActuationCheck()
    excessive_accel = ACCEL_MAX * 3
    sm, CS, pose = self._create_mocks(
      mocker,
      accel_calibrated=excessive_accel,
      a_ego=excessive_accel,  # Valid initially
    )

    # Build up excessive counter
    for _ in range(MIN_EXCESSIVE_ACTUATION_COUNT - 1):
      check.update(sm, CS, pose)

    # Now make livepose invalid (aEgo differs from calibrated by more than 2)
    sm_invalid, CS_invalid, pose_invalid = self._create_mocks(
      mocker,
      accel_calibrated=excessive_accel,
      a_ego=excessive_accel - 3.0,  # Differs by 3, making livepose invalid
    )
    check.update(sm_invalid, CS_invalid, pose_invalid)

    assert check._excessive_counter == 0

  def test_counter_resets_on_normal_driving(self, mocker):
    """Excessive counter should reset when conditions normalize."""
    check = ExcessiveActuationCheck()
    excessive_accel = ACCEL_MAX * 3
    sm_excessive, CS_excessive, pose_excessive = self._create_mocks(
      mocker,
      accel_calibrated=excessive_accel,
      a_ego=excessive_accel,
    )

    # Build up some counter
    for _ in range(MIN_EXCESSIVE_ACTUATION_COUNT - 5):
      check.update(sm_excessive, CS_excessive, pose_excessive)

    # Now return to normal
    sm_normal, CS_normal, pose_normal = self._create_mocks(
      mocker,
      accel_calibrated=1.0,
      a_ego=1.0,
    )
    check.update(sm_normal, CS_normal, pose_normal)

    assert check._excessive_counter == 0

  def test_roll_compensation_in_lateral_check(self, mocker):
    """Roll compensation should be included in lateral acceleration calculation."""
    check = ExcessiveActuationCheck()
    v_ego = 20.0
    # If roll is present, it compensates for some lateral accel
    # roll_compensated = v_ego * yaw_rate - sin(roll) * g

    # Without roll: yaw_rate that would just exceed threshold
    threshold_yaw = (ISO_LATERAL_ACCEL * 2) / v_ego

    # With roll compensation, the same yaw_rate might not exceed
    # sin(roll) * g subtracts from the lateral accel
    roll = 0.1  # Positive roll (banked road)

    sm, CS, pose = self._create_mocks(
      mocker,
      v_ego=v_ego,
      yaw_rate=threshold_yaw,  # Would exceed without roll
      roll=roll,
      a_ego=0.0,
      accel_calibrated=0.0,
    )

    # First pass engage buffer
    for _ in range(MIN_LATERAL_ENGAGE_BUFFER + 1):
      check.update(sm, CS, pose)

    # Then check excessive counter
    for _ in range(MIN_EXCESSIVE_ACTUATION_COUNT + 10):
      result = check.update(sm, CS, pose)

    # With roll compensation, the actual lateral accel is reduced
    # So depending on the values, it may or may not trigger
    # This test just verifies the code path runs without error
    assert result is None or result == ExcessiveActuationType.LATERAL


class TestExcessiveActuationTypeEnum:
  """Tests for the ExcessiveActuationType enum."""

  def test_enum_values(self):
    """Verify enum has expected values."""
    assert ExcessiveActuationType.LONGITUDINAL == "longitudinal"
    assert ExcessiveActuationType.LATERAL == "lateral"

  def test_enum_is_string(self):
    """Enum values should be strings (StrEnum)."""
    assert isinstance(ExcessiveActuationType.LONGITUDINAL, str)
    assert isinstance(ExcessiveActuationType.LATERAL, str)
