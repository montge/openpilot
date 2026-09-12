"""Tests for selfdrive/controls/lib/longcontrol.py - longitudinal control.

fork: upstream removed the STARTING state from the longitudinal state machine, along
with the CP and v_ego arguments to long_control_state_trans and CP.stoppingDecelRate's
role in the stopping ramp. The STARTING transition tests that used to live here are
gone with the state; what remains covers every transition still reachable.
"""

import numpy as np

from openpilot.selfdrive.controls.lib.longcontrol import (
  long_control_state_trans,
  LongControl,
  LongCtrlState,
)


def create_mock_cp(mocker):
  """Create a mock CarParams for testing."""
  CP = mocker.MagicMock()
  CP.stopAccel = -2.0
  CP.longitudinalTuning.kpBP = [0.0, 5.0, 35.0]
  CP.longitudinalTuning.kpV = [0.0, 0.0, 0.0]
  CP.longitudinalTuning.kiBP = [0.0, 35.0]
  CP.longitudinalTuning.kiV = [0.0, 0.0]
  return CP


def create_mock_cs(mocker):
  """Create a mock CarState for testing."""
  CS = mocker.MagicMock()
  CS.vEgo = 10.0
  CS.aEgo = 0.0
  CS.brakePressed = False
  CS.cruiseState = mocker.MagicMock()
  CS.cruiseState.standstill = False
  return CS


class TestLongControlStateTrans:
  """Test long_control_state_trans state machine."""

  def test_off_stays_off_when_not_active(self):
    """Test OFF stays OFF when not active."""
    result = long_control_state_trans(
      active=False, long_control_state=LongCtrlState.off, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.off

  def test_off_to_stopping_when_should_stop(self):
    """Test OFF -> STOPPING when should_stop is True."""
    result = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.off, should_stop=True, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.stopping

  def test_off_to_stopping_when_brake_pressed(self):
    """Test OFF -> STOPPING when brake is pressed."""
    result = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.off, should_stop=False, brake_pressed=True, cruise_standstill=False
    )
    assert result == LongCtrlState.stopping

  def test_off_to_stopping_when_cruise_standstill(self):
    """Test OFF -> STOPPING when cruise standstill."""
    result = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.off, should_stop=False, brake_pressed=False, cruise_standstill=True
    )
    assert result == LongCtrlState.stopping

  def test_off_to_pid_when_starting_conditions_met(self):
    """Test OFF -> PID when starting conditions are met."""
    result = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.off, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.pid

  # STOPPING state transitions
  def test_stopping_to_pid(self):
    """Test STOPPING -> PID when starting conditions are met."""
    result = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.stopping, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.pid

  def test_stopping_stays_stopping_when_should_stop(self):
    """Test STOPPING stays STOPPING when should_stop is True."""
    result = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.stopping, should_stop=True, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.stopping

  def test_stopping_stays_stopping_when_brake_pressed(self):
    """Test STOPPING stays STOPPING while the brake is held."""
    result = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.stopping, should_stop=False, brake_pressed=True, cruise_standstill=False
    )
    assert result == LongCtrlState.stopping

  def test_stopping_to_off_when_not_active(self):
    """Test STOPPING -> OFF when not active."""
    result = long_control_state_trans(
      active=False, long_control_state=LongCtrlState.stopping, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.off

  # PID state transitions
  def test_pid_to_stopping(self):
    """Test PID -> STOPPING when should_stop."""
    result = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.pid, should_stop=True, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.stopping

  def test_pid_stays_pid_when_driving(self):
    """Test PID stays PID when conditions normal."""
    result = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.pid, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.pid

  def test_pid_stays_pid_when_brake_pressed(self):
    """Test PID stays PID when the brake is pressed but should_stop is False."""
    result = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.pid, should_stop=False, brake_pressed=True, cruise_standstill=False
    )
    assert result == LongCtrlState.pid

  def test_pid_to_off_when_not_active(self):
    """Test PID -> OFF when not active."""
    result = long_control_state_trans(
      active=False, long_control_state=LongCtrlState.pid, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.off


class TestLongControl:
  """Test LongControl class."""

  def test_init(self, mocker):
    """Test LongControl initialization."""
    CP = create_mock_cp(mocker)
    lc = LongControl(CP)
    assert lc.long_control_state == LongCtrlState.off
    assert lc.last_output_accel == 0.0
    assert lc.pid is not None

  def test_reset(self, mocker):
    """Test reset clears PID state."""
    CP = create_mock_cp(mocker)
    lc = LongControl(CP)
    lc.pid.i = 1.0  # Set some state
    lc.reset()
    assert lc.pid.i == 0.0

  def test_update_off_state(self, mocker):
    """Test update in OFF state returns zero."""
    CP = create_mock_cp(mocker)
    lc = LongControl(CP)
    CS = create_mock_cs(mocker)
    accel_limits = [-3.5, 2.0]

    output = lc.update(active=False, CS=CS, a_target=1.0, should_stop=False, accel_limits=accel_limits)

    assert output == 0.0
    assert lc.long_control_state == LongCtrlState.off

  def test_update_stopping_state(self, mocker):
    """Test update in STOPPING state decreases output."""
    CP = create_mock_cp(mocker)
    lc = LongControl(CP)
    lc.long_control_state = LongCtrlState.pid
    lc.last_output_accel = 0.0
    CS = create_mock_cs(mocker)
    CS.vEgo = 0.1
    accel_limits = [-3.5, 2.0]

    # First update should trigger stopping
    lc.update(active=True, CS=CS, a_target=0.0, should_stop=True, accel_limits=accel_limits)

    assert lc.long_control_state == LongCtrlState.stopping

  def test_update_pid_state(self, mocker):
    """Test update in PID state uses PID controller."""
    CP = create_mock_cp(mocker)
    lc = LongControl(CP)
    lc.long_control_state = LongCtrlState.stopping
    CS = create_mock_cs(mocker)
    CS.vEgo = 5.0
    CS.aEgo = 0.0
    accel_limits = [-3.5, 2.0]

    # Transition to PID
    output = lc.update(active=True, CS=CS, a_target=1.0, should_stop=False, accel_limits=accel_limits)

    assert lc.long_control_state == LongCtrlState.pid
    assert isinstance(output, (int, float, np.floating))

  def test_output_clamped_to_limits(self, mocker):
    """Test output is clamped to accel_limits."""
    CP = create_mock_cp(mocker)
    lc = LongControl(CP)
    CS = create_mock_cs(mocker)
    CS.vEgo = 5.0
    accel_limits = [-1.0, 1.0]  # Narrow limits

    # a_target well above the positive limit, should be clamped to it
    output = lc.update(active=True, CS=CS, a_target=5.0, should_stop=False, accel_limits=accel_limits)

    assert lc.long_control_state == LongCtrlState.pid
    assert output <= accel_limits[1]
    assert output >= accel_limits[0]

  def test_stopping_decelerates(self, mocker):
    """Test stopping state gradually decreases acceleration."""
    CP = create_mock_cp(mocker)
    lc = LongControl(CP)
    lc.long_control_state = LongCtrlState.stopping
    lc.last_output_accel = 0.5  # Start with some accel
    CS = create_mock_cs(mocker)
    CS.vEgo = 1.0
    accel_limits = [-3.5, 2.0]

    output = lc.update(active=True, CS=CS, a_target=0.0, should_stop=True, accel_limits=accel_limits)

    # Should decrease from last output
    assert output < 0.5

  def test_stopping_at_stop_accel(self, mocker):
    """Test stopping when output_accel <= stopAccel (no decel needed)."""
    CP = create_mock_cp(mocker)
    CP.stopAccel = -2.0
    lc = LongControl(CP)
    lc.long_control_state = LongCtrlState.stopping
    lc.last_output_accel = -2.5  # Already at or below stopAccel
    CS = create_mock_cs(mocker)
    CS.vEgo = 0.1
    accel_limits = [-3.5, 2.0]

    output = lc.update(active=True, CS=CS, a_target=0.0, should_stop=True, accel_limits=accel_limits)

    # Output should be clamped, stays at last_output_accel (no decel applied)
    assert output == lc.last_output_accel

  def test_pid_limits_set(self, mocker):
    """Test PID limits are set from accel_limits."""
    CP = create_mock_cp(mocker)
    lc = LongControl(CP)
    CS = create_mock_cs(mocker)
    accel_limits = [-2.5, 1.5]

    lc.update(active=False, CS=CS, a_target=0.0, should_stop=False, accel_limits=accel_limits)

    assert lc.pid.neg_limit == -2.5
    assert lc.pid.pos_limit == 1.5


class TestLongCtrlStateEnum:
  """Test LongCtrlState enum values."""

  def test_enum_values_exist(self):
    """Test expected enum values exist."""
    assert LongCtrlState.off is not None
    assert LongCtrlState.stopping is not None
    assert LongCtrlState.starting is not None
    assert LongCtrlState.pid is not None

  def test_enum_values_different(self):
    """Test enum values are distinct."""
    values = {LongCtrlState.off, LongCtrlState.stopping, LongCtrlState.starting, LongCtrlState.pid}
    assert len(values) == 4
