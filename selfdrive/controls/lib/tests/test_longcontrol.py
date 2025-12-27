"""Tests for selfdrive/controls/lib/longcontrol.py - longitudinal control."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from cereal import car

from openpilot.selfdrive.controls.lib.longcontrol import (
  long_control_state_trans, LongControl, LongCtrlState,
)


def create_mock_cp():
  """Create a mock CarParams for testing."""
  CP = MagicMock()
  CP.vEgoStarting = 0.3
  CP.startingState = True
  CP.stopAccel = -2.0
  CP.startAccel = 1.2
  CP.stoppingDecelRate = 0.8
  CP.longitudinalTuning.kpBP = [0.0, 5.0, 35.0]
  CP.longitudinalTuning.kpV = [0.0, 0.0, 0.0]
  CP.longitudinalTuning.kiBP = [0.0, 35.0]
  CP.longitudinalTuning.kiV = [0.0, 0.0]
  return CP


def create_mock_cs():
  """Create a mock CarState for testing."""
  CS = MagicMock()
  CS.vEgo = 10.0
  CS.aEgo = 0.0
  CS.brakePressed = False
  CS.cruiseState = MagicMock()
  CS.cruiseState.standstill = False
  return CS


class TestLongControlStateTrans(unittest.TestCase):
  """Test long_control_state_trans state machine."""

  def setUp(self):
    """Set up test fixtures."""
    self.CP = create_mock_cp()

  # OFF state transitions
  def test_off_stays_off_when_not_active(self):
    """Test OFF stays OFF when not active."""
    result = long_control_state_trans(
      self.CP, active=False, long_control_state=LongCtrlState.off,
      v_ego=10.0, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    self.assertEqual(result, LongCtrlState.off)

  def test_off_to_stopping_when_should_stop(self):
    """Test OFF -> STOPPING when should_stop is True."""
    result = long_control_state_trans(
      self.CP, active=True, long_control_state=LongCtrlState.off,
      v_ego=10.0, should_stop=True, brake_pressed=False, cruise_standstill=False
    )
    self.assertEqual(result, LongCtrlState.stopping)

  def test_off_to_stopping_when_brake_pressed(self):
    """Test OFF -> STOPPING when brake is pressed."""
    result = long_control_state_trans(
      self.CP, active=True, long_control_state=LongCtrlState.off,
      v_ego=10.0, should_stop=False, brake_pressed=True, cruise_standstill=False
    )
    self.assertEqual(result, LongCtrlState.stopping)

  def test_off_to_stopping_when_cruise_standstill(self):
    """Test OFF -> STOPPING when cruise standstill."""
    result = long_control_state_trans(
      self.CP, active=True, long_control_state=LongCtrlState.off,
      v_ego=10.0, should_stop=False, brake_pressed=False, cruise_standstill=True
    )
    self.assertEqual(result, LongCtrlState.stopping)

  def test_off_to_starting_with_starting_state(self):
    """Test OFF -> STARTING when conditions met and startingState True."""
    self.CP.startingState = True
    result = long_control_state_trans(
      self.CP, active=True, long_control_state=LongCtrlState.off,
      v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    self.assertEqual(result, LongCtrlState.starting)

  def test_off_to_pid_without_starting_state(self):
    """Test OFF -> PID when conditions met and startingState False."""
    self.CP.startingState = False
    result = long_control_state_trans(
      self.CP, active=True, long_control_state=LongCtrlState.off,
      v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    self.assertEqual(result, LongCtrlState.pid)

  # STOPPING state transitions
  def test_stopping_to_starting(self):
    """Test STOPPING -> STARTING when conditions met."""
    self.CP.startingState = True
    result = long_control_state_trans(
      self.CP, active=True, long_control_state=LongCtrlState.stopping,
      v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    self.assertEqual(result, LongCtrlState.starting)

  def test_stopping_to_pid(self):
    """Test STOPPING -> PID when starting without startingState."""
    self.CP.startingState = False
    result = long_control_state_trans(
      self.CP, active=True, long_control_state=LongCtrlState.stopping,
      v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    self.assertEqual(result, LongCtrlState.pid)

  def test_stopping_stays_stopping_when_should_stop(self):
    """Test STOPPING stays STOPPING when should_stop is True."""
    result = long_control_state_trans(
      self.CP, active=True, long_control_state=LongCtrlState.stopping,
      v_ego=0.1, should_stop=True, brake_pressed=False, cruise_standstill=False
    )
    self.assertEqual(result, LongCtrlState.stopping)

  def test_stopping_to_off_when_not_active(self):
    """Test STOPPING -> OFF when not active."""
    result = long_control_state_trans(
      self.CP, active=False, long_control_state=LongCtrlState.stopping,
      v_ego=10.0, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    self.assertEqual(result, LongCtrlState.off)

  # STARTING state transitions
  def test_starting_to_stopping(self):
    """Test STARTING -> STOPPING when should_stop."""
    result = long_control_state_trans(
      self.CP, active=True, long_control_state=LongCtrlState.starting,
      v_ego=0.1, should_stop=True, brake_pressed=False, cruise_standstill=False
    )
    self.assertEqual(result, LongCtrlState.stopping)

  def test_starting_to_pid_when_started(self):
    """Test STARTING -> PID when v_ego > vEgoStarting."""
    self.CP.vEgoStarting = 0.3
    result = long_control_state_trans(
      self.CP, active=True, long_control_state=LongCtrlState.starting,
      v_ego=0.5, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    self.assertEqual(result, LongCtrlState.pid)

  def test_starting_stays_starting_below_threshold(self):
    """Test STARTING stays STARTING when below vEgoStarting."""
    self.CP.vEgoStarting = 0.3
    result = long_control_state_trans(
      self.CP, active=True, long_control_state=LongCtrlState.starting,
      v_ego=0.2, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    self.assertEqual(result, LongCtrlState.starting)

  def test_starting_to_off_when_not_active(self):
    """Test STARTING -> OFF when not active."""
    result = long_control_state_trans(
      self.CP, active=False, long_control_state=LongCtrlState.starting,
      v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    self.assertEqual(result, LongCtrlState.off)

  # PID state transitions
  def test_pid_to_stopping(self):
    """Test PID -> STOPPING when should_stop."""
    result = long_control_state_trans(
      self.CP, active=True, long_control_state=LongCtrlState.pid,
      v_ego=10.0, should_stop=True, brake_pressed=False, cruise_standstill=False
    )
    self.assertEqual(result, LongCtrlState.stopping)

  def test_pid_stays_pid_when_driving(self):
    """Test PID stays PID when conditions normal."""
    self.CP.vEgoStarting = 0.3
    result = long_control_state_trans(
      self.CP, active=True, long_control_state=LongCtrlState.pid,
      v_ego=10.0, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    self.assertEqual(result, LongCtrlState.pid)

  def test_pid_to_off_when_not_active(self):
    """Test PID -> OFF when not active."""
    result = long_control_state_trans(
      self.CP, active=False, long_control_state=LongCtrlState.pid,
      v_ego=10.0, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    self.assertEqual(result, LongCtrlState.off)


class TestLongControl(unittest.TestCase):
  """Test LongControl class."""

  def setUp(self):
    """Set up test fixtures."""
    self.CP = create_mock_cp()

  def test_init(self):
    """Test LongControl initialization."""
    lc = LongControl(self.CP)
    self.assertEqual(lc.long_control_state, LongCtrlState.off)
    self.assertEqual(lc.last_output_accel, 0.0)
    self.assertIsNotNone(lc.pid)

  def test_reset(self):
    """Test reset clears PID state."""
    lc = LongControl(self.CP)
    lc.pid.i = 1.0  # Set some state
    lc.reset()
    self.assertEqual(lc.pid.i, 0.0)

  def test_update_off_state(self):
    """Test update in OFF state returns zero."""
    lc = LongControl(self.CP)
    CS = create_mock_cs()
    accel_limits = [-3.5, 2.0]

    output = lc.update(active=False, CS=CS, a_target=1.0,
                       should_stop=False, accel_limits=accel_limits)

    self.assertEqual(output, 0.0)
    self.assertEqual(lc.long_control_state, LongCtrlState.off)

  def test_update_stopping_state(self):
    """Test update in STOPPING state decreases output."""
    lc = LongControl(self.CP)
    lc.long_control_state = LongCtrlState.pid
    lc.last_output_accel = 0.0
    CS = create_mock_cs()
    CS.vEgo = 0.1
    accel_limits = [-3.5, 2.0]

    # First update should trigger stopping
    output = lc.update(active=True, CS=CS, a_target=0.0,
                       should_stop=True, accel_limits=accel_limits)

    self.assertEqual(lc.long_control_state, LongCtrlState.stopping)

  def test_update_starting_state(self):
    """Test update in STARTING state uses startAccel."""
    lc = LongControl(self.CP)
    lc.long_control_state = LongCtrlState.stopping
    CS = create_mock_cs()
    CS.vEgo = 0.1
    accel_limits = [-3.5, 2.0]

    # Transition to starting
    output = lc.update(active=True, CS=CS, a_target=0.5,
                       should_stop=False, accel_limits=accel_limits)

    # Should be in starting and using startAccel
    self.assertEqual(lc.long_control_state, LongCtrlState.starting)
    self.assertAlmostEqual(output, self.CP.startAccel, places=1)

  def test_update_pid_state(self):
    """Test update in PID state uses PID controller."""
    lc = LongControl(self.CP)
    self.CP.startingState = False
    lc.long_control_state = LongCtrlState.stopping
    CS = create_mock_cs()
    CS.vEgo = 5.0
    CS.aEgo = 0.0
    accel_limits = [-3.5, 2.0]

    # Transition to PID
    output = lc.update(active=True, CS=CS, a_target=1.0,
                       should_stop=False, accel_limits=accel_limits)

    self.assertEqual(lc.long_control_state, LongCtrlState.pid)
    self.assertIsInstance(output, (int, float, np.floating))

  def test_output_clamped_to_limits(self):
    """Test output is clamped to accel_limits."""
    lc = LongControl(self.CP)
    lc.long_control_state = LongCtrlState.starting
    CS = create_mock_cs()
    CS.vEgo = 0.1
    accel_limits = [-1.0, 1.0]  # Narrow limits

    # startAccel is 1.2, should be clamped to 1.0
    output = lc.update(active=True, CS=CS, a_target=0.5,
                       should_stop=False, accel_limits=accel_limits)

    self.assertLessEqual(output, accel_limits[1])
    self.assertGreaterEqual(output, accel_limits[0])

  def test_stopping_decelerates(self):
    """Test stopping state gradually decreases acceleration."""
    lc = LongControl(self.CP)
    lc.long_control_state = LongCtrlState.stopping
    lc.last_output_accel = 0.5  # Start with some accel
    CS = create_mock_cs()
    CS.vEgo = 1.0
    accel_limits = [-3.5, 2.0]

    output = lc.update(active=True, CS=CS, a_target=0.0,
                       should_stop=True, accel_limits=accel_limits)

    # Should decrease from last output
    self.assertLess(output, 0.5)

  def test_pid_limits_set(self):
    """Test PID limits are set from accel_limits."""
    lc = LongControl(self.CP)
    CS = create_mock_cs()
    accel_limits = [-2.5, 1.5]

    lc.update(active=False, CS=CS, a_target=0.0,
              should_stop=False, accel_limits=accel_limits)

    self.assertEqual(lc.pid.neg_limit, -2.5)
    self.assertEqual(lc.pid.pos_limit, 1.5)


class TestLongCtrlStateEnum(unittest.TestCase):
  """Test LongCtrlState enum values."""

  def test_enum_values_exist(self):
    """Test expected enum values exist."""
    self.assertIsNotNone(LongCtrlState.off)
    self.assertIsNotNone(LongCtrlState.stopping)
    self.assertIsNotNone(LongCtrlState.starting)
    self.assertIsNotNone(LongCtrlState.pid)

  def test_enum_values_different(self):
    """Test enum values are distinct."""
    values = {LongCtrlState.off, LongCtrlState.stopping,
              LongCtrlState.starting, LongCtrlState.pid}
    self.assertEqual(len(values), 4)


if __name__ == '__main__':
  unittest.main()
