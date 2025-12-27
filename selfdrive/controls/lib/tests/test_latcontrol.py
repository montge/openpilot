"""Tests for selfdrive/controls/lib/latcontrol.py - lateral control base class."""
import unittest
from unittest.mock import MagicMock

import numpy as np

from openpilot.selfdrive.controls.lib.latcontrol import LatControl


def create_mock_cp():
  """Create a mock CarParams for testing."""
  CP = MagicMock()
  CP.steerLimitTimer = 4.0
  return CP


def create_mock_cs(v_ego=15.0, steering_pressed=False):
  """Create a mock CarState for testing."""
  CS = MagicMock()
  CS.vEgo = v_ego
  CS.steeringPressed = steering_pressed
  return CS


class ConcreteLatControl(LatControl):
  """Concrete implementation for testing abstract base class."""

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay):
    return 0.0, 0.0, None


class TestLatControlInit(unittest.TestCase):
  """Test LatControl initialization."""

  def test_init(self):
    """Test LatControl initializes correctly."""
    CP = create_mock_cp()
    CI = MagicMock()
    dt = 0.01

    controller = ConcreteLatControl(CP, CI, dt)

    self.assertEqual(controller.dt, 0.01)
    self.assertEqual(controller.sat_limit, 4.0)
    self.assertEqual(controller.sat_time, 0.0)
    self.assertEqual(controller.sat_check_min_speed, 10.0)
    self.assertEqual(controller.steer_max, 1.0)

  def test_init_different_sat_limit(self):
    """Test initialization with different steerLimitTimer."""
    CP = create_mock_cp()
    CP.steerLimitTimer = 2.5
    CI = MagicMock()

    controller = ConcreteLatControl(CP, CI, 0.01)

    self.assertEqual(controller.sat_limit, 2.5)


class TestLatControlReset(unittest.TestCase):
  """Test LatControl reset method."""

  def test_reset_clears_sat_time(self):
    """Test reset sets sat_time to zero."""
    CP = create_mock_cp()
    controller = ConcreteLatControl(CP, MagicMock(), 0.01)
    controller.sat_time = 2.5

    controller.reset()

    self.assertEqual(controller.sat_time, 0.0)

  def test_reset_from_zero_stays_zero(self):
    """Test reset when sat_time is already zero."""
    CP = create_mock_cp()
    controller = ConcreteLatControl(CP, MagicMock(), 0.01)

    controller.reset()

    self.assertEqual(controller.sat_time, 0.0)


class TestLatControlCheckSaturation(unittest.TestCase):
  """Test LatControl _check_saturation method."""

  def setUp(self):
    """Set up test fixtures."""
    CP = create_mock_cp()
    self.controller = ConcreteLatControl(CP, MagicMock(), 0.01)

  def test_saturation_increments_time(self):
    """Test saturation increments sat_time."""
    CS = create_mock_cs(v_ego=15.0)  # Above min speed
    self.controller.sat_time = 0.0

    self.controller._check_saturation(
      saturated=True, CS=CS,
      steer_limited_by_safety=False, curvature_limited=False
    )

    self.assertGreater(self.controller.sat_time, 0.0)

  def test_no_saturation_decrements_time(self):
    """Test no saturation decrements sat_time."""
    CS = create_mock_cs()
    self.controller.sat_time = 1.0

    self.controller._check_saturation(
      saturated=False, CS=CS,
      steer_limited_by_safety=False, curvature_limited=False
    )

    self.assertLess(self.controller.sat_time, 1.0)

  def test_saturation_clipped_to_limit(self):
    """Test sat_time is clipped to sat_limit."""
    CS = create_mock_cs()
    self.controller.sat_time = 10.0  # Way above limit

    self.controller._check_saturation(
      saturated=True, CS=CS,
      steer_limited_by_safety=False, curvature_limited=False
    )

    self.assertLessEqual(self.controller.sat_time, self.controller.sat_limit)

  def test_saturation_clipped_to_zero(self):
    """Test sat_time doesn't go below zero."""
    CS = create_mock_cs()
    self.controller.sat_time = 0.0

    self.controller._check_saturation(
      saturated=False, CS=CS,
      steer_limited_by_safety=False, curvature_limited=False
    )

    self.assertEqual(self.controller.sat_time, 0.0)

  def test_no_increment_below_min_speed(self):
    """Test no saturation increment below min speed."""
    CS = create_mock_cs(v_ego=5.0)  # Below sat_check_min_speed
    self.controller.sat_time = 0.0

    self.controller._check_saturation(
      saturated=True, CS=CS,
      steer_limited_by_safety=False, curvature_limited=False
    )

    # Should decrement, not increment
    self.assertEqual(self.controller.sat_time, 0.0)

  def test_no_increment_when_safety_limited(self):
    """Test no saturation increment when steer limited by safety."""
    CS = create_mock_cs()
    self.controller.sat_time = 0.0

    self.controller._check_saturation(
      saturated=True, CS=CS,
      steer_limited_by_safety=True, curvature_limited=False
    )

    # Should not increment
    self.assertEqual(self.controller.sat_time, 0.0)

  def test_no_increment_when_steering_pressed(self):
    """Test no saturation increment when driver is steering."""
    CS = create_mock_cs(steering_pressed=True)
    self.controller.sat_time = 0.0

    self.controller._check_saturation(
      saturated=True, CS=CS,
      steer_limited_by_safety=False, curvature_limited=False
    )

    # Should not increment
    self.assertEqual(self.controller.sat_time, 0.0)

  def test_curvature_limited_increments(self):
    """Test curvature_limited also causes increment."""
    CS = create_mock_cs()
    self.controller.sat_time = 0.0

    self.controller._check_saturation(
      saturated=False, CS=CS,
      steer_limited_by_safety=False, curvature_limited=True
    )

    self.assertGreater(self.controller.sat_time, 0.0)

  def test_returns_true_at_limit(self):
    """Test returns True when sat_time reaches limit."""
    CS = create_mock_cs()
    self.controller.sat_time = self.controller.sat_limit

    result = self.controller._check_saturation(
      saturated=True, CS=CS,
      steer_limited_by_safety=False, curvature_limited=False
    )

    self.assertTrue(result)

  def test_returns_false_below_limit(self):
    """Test returns False when sat_time below limit."""
    CS = create_mock_cs()
    self.controller.sat_time = 0.0

    result = self.controller._check_saturation(
      saturated=True, CS=CS,
      steer_limited_by_safety=False, curvature_limited=False
    )

    self.assertFalse(result)


class TestLatControlAbstract(unittest.TestCase):
  """Test LatControl abstract nature."""

  def test_cannot_instantiate_directly(self):
    """Test LatControl cannot be instantiated directly."""
    CP = create_mock_cp()
    with self.assertRaises(TypeError):
      LatControl(CP, MagicMock(), 0.01)

  def test_update_is_abstract(self):
    """Test update method is abstract."""
    self.assertTrue(hasattr(LatControl, 'update'))
    # Check that it's declared as abstract in the class
    self.assertIn('update', LatControl.__abstractmethods__)


if __name__ == '__main__':
  unittest.main()
