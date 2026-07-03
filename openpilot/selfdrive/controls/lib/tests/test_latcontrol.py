"""Tests for selfdrive/controls/lib/latcontrol.py - lateral control base class."""

import pytest

from openpilot.selfdrive.controls.lib.latcontrol import LatControl


def create_mock_cp(mocker):
  """Create a mock CarParams for testing."""
  CP = mocker.MagicMock()
  CP.steerLimitTimer = 4.0
  return CP


def create_mock_cs(mocker, v_ego=15.0, steering_pressed=False):
  """Create a mock CarState for testing."""
  CS = mocker.MagicMock()
  CS.vEgo = v_ego
  CS.steeringPressed = steering_pressed
  return CS


class ConcreteLatControl(LatControl):
  """Concrete implementation for testing abstract base class."""

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay):
    return 0.0, 0.0, None


class TestLatControlInit:
  """Test LatControl initialization."""

  def test_init(self, mocker):
    """Test LatControl initializes correctly."""
    CP = create_mock_cp(mocker)
    CI = mocker.MagicMock()
    dt = 0.01

    controller = ConcreteLatControl(CP, CI, dt)

    assert controller.dt == 0.01
    assert controller.sat_limit == 4.0
    assert controller.sat_time == 0.0
    assert controller.sat_check_min_speed == 10.0
    assert controller.steer_max == 1.0

  def test_init_different_sat_limit(self, mocker):
    """Test initialization with different steerLimitTimer."""
    CP = create_mock_cp(mocker)
    CP.steerLimitTimer = 2.5
    CI = mocker.MagicMock()

    controller = ConcreteLatControl(CP, CI, 0.01)

    assert controller.sat_limit == 2.5


class TestLatControlReset:
  """Test LatControl reset method."""

  def test_reset_clears_sat_time(self, mocker):
    """Test reset sets sat_time to zero."""
    CP = create_mock_cp(mocker)
    controller = ConcreteLatControl(CP, mocker.MagicMock(), 0.01)
    controller.sat_time = 2.5

    controller.reset()

    assert controller.sat_time == 0.0

  def test_reset_from_zero_stays_zero(self, mocker):
    """Test reset when sat_time is already zero."""
    CP = create_mock_cp(mocker)
    controller = ConcreteLatControl(CP, mocker.MagicMock(), 0.01)

    controller.reset()

    assert controller.sat_time == 0.0


class TestLatControlCheckSaturation:
  """Test LatControl _check_saturation method."""

  @pytest.fixture
  def controller(self, mocker):
    """Set up test fixtures."""
    CP = create_mock_cp(mocker)
    return ConcreteLatControl(CP, mocker.MagicMock(), 0.01)

  def test_saturation_increments_time(self, mocker, controller):
    """Test saturation increments sat_time."""
    CS = create_mock_cs(mocker, v_ego=15.0)  # Above min speed
    controller.sat_time = 0.0

    controller._check_saturation(saturated=True, CS=CS, steer_limited_by_safety=False, curvature_limited=False)

    assert controller.sat_time > 0.0

  def test_no_saturation_decrements_time(self, mocker, controller):
    """Test no saturation decrements sat_time."""
    CS = create_mock_cs(mocker)
    controller.sat_time = 1.0

    controller._check_saturation(saturated=False, CS=CS, steer_limited_by_safety=False, curvature_limited=False)

    assert controller.sat_time < 1.0

  def test_saturation_clipped_to_limit(self, mocker, controller):
    """Test sat_time is clipped to sat_limit."""
    CS = create_mock_cs(mocker)
    controller.sat_time = 10.0  # Way above limit

    controller._check_saturation(saturated=True, CS=CS, steer_limited_by_safety=False, curvature_limited=False)

    assert controller.sat_time <= controller.sat_limit

  def test_saturation_clipped_to_zero(self, mocker, controller):
    """Test sat_time doesn't go below zero."""
    CS = create_mock_cs(mocker)
    controller.sat_time = 0.0

    controller._check_saturation(saturated=False, CS=CS, steer_limited_by_safety=False, curvature_limited=False)

    assert controller.sat_time == 0.0

  def test_no_increment_below_min_speed(self, mocker, controller):
    """Test no saturation increment below min speed."""
    CS = create_mock_cs(mocker, v_ego=5.0)  # Below sat_check_min_speed
    controller.sat_time = 0.0

    controller._check_saturation(saturated=True, CS=CS, steer_limited_by_safety=False, curvature_limited=False)

    # Should decrement, not increment
    assert controller.sat_time == 0.0

  def test_no_increment_when_safety_limited(self, mocker, controller):
    """Test no saturation increment when steer limited by safety."""
    CS = create_mock_cs(mocker)
    controller.sat_time = 0.0

    controller._check_saturation(saturated=True, CS=CS, steer_limited_by_safety=True, curvature_limited=False)

    # Should not increment
    assert controller.sat_time == 0.0

  def test_no_increment_when_steering_pressed(self, mocker, controller):
    """Test no saturation increment when driver is steering."""
    CS = create_mock_cs(mocker, steering_pressed=True)
    controller.sat_time = 0.0

    controller._check_saturation(saturated=True, CS=CS, steer_limited_by_safety=False, curvature_limited=False)

    # Should not increment
    assert controller.sat_time == 0.0

  def test_curvature_limited_increments(self, mocker, controller):
    """Test curvature_limited also causes increment."""
    CS = create_mock_cs(mocker)
    controller.sat_time = 0.0

    controller._check_saturation(saturated=False, CS=CS, steer_limited_by_safety=False, curvature_limited=True)

    assert controller.sat_time > 0.0

  def test_returns_true_at_limit(self, mocker, controller):
    """Test returns True when sat_time reaches limit."""
    CS = create_mock_cs(mocker)
    controller.sat_time = controller.sat_limit

    result = controller._check_saturation(saturated=True, CS=CS, steer_limited_by_safety=False, curvature_limited=False)

    assert result  # numpy bool compatible

  def test_returns_false_below_limit(self, mocker, controller):
    """Test returns False when sat_time below limit."""
    CS = create_mock_cs(mocker)
    controller.sat_time = 0.0

    result = controller._check_saturation(saturated=True, CS=CS, steer_limited_by_safety=False, curvature_limited=False)

    assert not result  # numpy bool compatible


class TestLatControlAbstract:
  """Test LatControl abstract nature."""

  def test_cannot_instantiate_directly(self, mocker):
    """Test LatControl cannot be instantiated directly."""
    CP = create_mock_cp(mocker)
    with pytest.raises(TypeError):
      LatControl(CP, mocker.MagicMock(), 0.01)

  def test_update_is_abstract(self):
    """Test update method is abstract."""
    assert hasattr(LatControl, 'update')
    # Check that it's declared as abstract in the class
    assert 'update' in LatControl.__abstractmethods__
