"""Tests for selfdrive/controls/lib/longcontrol.py - longitudinal control."""

import numpy as np

from openpilot.selfdrive.controls.lib.longcontrol import (
  long_control_state_trans,
  LongControl,
  LongCtrlState,
)


def create_mock_cp(mocker):
  """Create a mock CarParams for testing."""
  CP = mocker.MagicMock()
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

  def test_off_stays_off_when_not_active(self, mocker):
    """Test OFF stays OFF when not active."""
    CP = create_mock_cp(mocker)
    result = long_control_state_trans(
      CP, active=False, long_control_state=LongCtrlState.off, v_ego=10.0, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.off

  def test_off_to_stopping_when_should_stop(self, mocker):
    """Test OFF -> STOPPING when should_stop is True."""
    CP = create_mock_cp(mocker)
    result = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.off, v_ego=10.0, should_stop=True, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.stopping

  def test_off_to_stopping_when_brake_pressed(self, mocker):
    """Test OFF -> STOPPING when brake is pressed."""
    CP = create_mock_cp(mocker)
    result = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.off, v_ego=10.0, should_stop=False, brake_pressed=True, cruise_standstill=False
    )
    assert result == LongCtrlState.stopping

  def test_off_to_stopping_when_cruise_standstill(self, mocker):
    """Test OFF -> STOPPING when cruise standstill."""
    CP = create_mock_cp(mocker)
    result = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.off, v_ego=10.0, should_stop=False, brake_pressed=False, cruise_standstill=True
    )
    assert result == LongCtrlState.stopping

  def test_off_to_starting_with_starting_state(self, mocker):
    """Test OFF -> STARTING when conditions met and startingState True."""
    CP = create_mock_cp(mocker)
    CP.startingState = True
    result = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.off, v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.starting

  def test_off_to_pid_without_starting_state(self, mocker):
    """Test OFF -> PID when conditions met and startingState False."""
    CP = create_mock_cp(mocker)
    CP.startingState = False
    result = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.off, v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.pid

  # STOPPING state transitions
  def test_stopping_to_starting(self, mocker):
    """Test STOPPING -> STARTING when conditions met."""
    CP = create_mock_cp(mocker)
    CP.startingState = True
    result = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.stopping, v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.starting

  def test_stopping_to_pid(self, mocker):
    """Test STOPPING -> PID when starting without startingState."""
    CP = create_mock_cp(mocker)
    CP.startingState = False
    result = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.stopping, v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.pid

  def test_stopping_stays_stopping_when_should_stop(self, mocker):
    """Test STOPPING stays STOPPING when should_stop is True."""
    CP = create_mock_cp(mocker)
    result = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.stopping, v_ego=0.1, should_stop=True, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.stopping

  def test_stopping_to_off_when_not_active(self, mocker):
    """Test STOPPING -> OFF when not active."""
    CP = create_mock_cp(mocker)
    result = long_control_state_trans(
      CP, active=False, long_control_state=LongCtrlState.stopping, v_ego=10.0, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.off

  # STARTING state transitions
  def test_starting_to_stopping(self, mocker):
    """Test STARTING -> STOPPING when should_stop."""
    CP = create_mock_cp(mocker)
    result = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.starting, v_ego=0.1, should_stop=True, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.stopping

  def test_starting_to_pid_when_started(self, mocker):
    """Test STARTING -> PID when v_ego > vEgoStarting."""
    CP = create_mock_cp(mocker)
    CP.vEgoStarting = 0.3
    result = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.starting, v_ego=0.5, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.pid

  def test_starting_stays_starting_below_threshold(self, mocker):
    """Test STARTING stays STARTING when below vEgoStarting."""
    CP = create_mock_cp(mocker)
    CP.vEgoStarting = 0.3
    result = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.starting, v_ego=0.2, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.starting

  def test_starting_to_off_when_not_active(self, mocker):
    """Test STARTING -> OFF when not active."""
    CP = create_mock_cp(mocker)
    result = long_control_state_trans(
      CP, active=False, long_control_state=LongCtrlState.starting, v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.off

  # PID state transitions
  def test_pid_to_stopping(self, mocker):
    """Test PID -> STOPPING when should_stop."""
    CP = create_mock_cp(mocker)
    result = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.pid, v_ego=10.0, should_stop=True, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.stopping

  def test_pid_stays_pid_when_driving(self, mocker):
    """Test PID stays PID when conditions normal."""
    CP = create_mock_cp(mocker)
    CP.vEgoStarting = 0.3
    result = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.pid, v_ego=10.0, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert result == LongCtrlState.pid

  def test_pid_to_off_when_not_active(self, mocker):
    """Test PID -> OFF when not active."""
    CP = create_mock_cp(mocker)
    result = long_control_state_trans(
      CP, active=False, long_control_state=LongCtrlState.pid, v_ego=10.0, should_stop=False, brake_pressed=False, cruise_standstill=False
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

  def test_update_starting_state(self, mocker):
    """Test update in STARTING state uses startAccel."""
    CP = create_mock_cp(mocker)
    lc = LongControl(CP)
    lc.long_control_state = LongCtrlState.stopping
    CS = create_mock_cs(mocker)
    CS.vEgo = 0.1
    accel_limits = [-3.5, 2.0]

    # Transition to starting
    output = lc.update(active=True, CS=CS, a_target=0.5, should_stop=False, accel_limits=accel_limits)

    # Should be in starting and using startAccel
    assert lc.long_control_state == LongCtrlState.starting
    assert abs(output - CP.startAccel) < 0.2

  def test_update_pid_state(self, mocker):
    """Test update in PID state uses PID controller."""
    CP = create_mock_cp(mocker)
    CP.startingState = False
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
    lc.long_control_state = LongCtrlState.starting
    CS = create_mock_cs(mocker)
    CS.vEgo = 0.1
    accel_limits = [-1.0, 1.0]  # Narrow limits

    # startAccel is 1.2, should be clamped to 1.0
    output = lc.update(active=True, CS=CS, a_target=0.5, should_stop=False, accel_limits=accel_limits)

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
