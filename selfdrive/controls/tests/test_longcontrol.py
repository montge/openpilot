import pytest

from cereal import car
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState, long_control_state_trans, LongControl
from openpilot.common.realtime import DT_CTRL


class TestLongControlStateTransition:
  def test_stay_stopped(self):
    CP = car.CarParams.new_message()
    active = True
    current_state = LongCtrlState.stopping
    next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1, should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1, should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP, active, current_state, v_ego=1.0, should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid
    active = False
    next_state = long_control_state_trans(CP, active, current_state, v_ego=1.0, should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.off


def test_engage():
  CP = car.CarParams.new_message()
  active = True
  current_state = LongCtrlState.off
  next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1, should_stop=True, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.stopping
  next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1, should_stop=False, brake_pressed=True, cruise_standstill=False)
  assert next_state == LongCtrlState.stopping
  next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=True)
  assert next_state == LongCtrlState.stopping
  next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.pid


def test_starting():
  CP = car.CarParams.new_message(startingState=True, vEgoStarting=0.5)
  active = True
  current_state = LongCtrlState.starting
  next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.starting
  next_state = long_control_state_trans(CP, active, current_state, v_ego=1.0, should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.pid


class TestLongControlStateTransitionComplete:
  """Comprehensive tests for all state transitions in the longitudinal control state machine."""

  def test_off_to_starting_with_starting_state(self):
    """OFF → STARTING when startingState=True and starting conditions met."""
    CP = car.CarParams.new_message(startingState=True, vEgoStarting=0.5)
    next_state = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.off, v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert next_state == LongCtrlState.starting

  def test_off_to_pid_without_starting_state(self):
    """OFF → PID when startingState=False and starting conditions met."""
    CP = car.CarParams.new_message(startingState=False)
    next_state = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.off, v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert next_state == LongCtrlState.pid

  def test_off_to_stopping_when_should_stop(self):
    """OFF → STOPPING when should_stop is True."""
    CP = car.CarParams.new_message()
    next_state = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.off, v_ego=5.0, should_stop=True, brake_pressed=False, cruise_standstill=False
    )
    assert next_state == LongCtrlState.stopping

  def test_off_to_stopping_when_brake_pressed(self):
    """OFF → STOPPING when brake is pressed."""
    CP = car.CarParams.new_message()
    next_state = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.off, v_ego=5.0, should_stop=False, brake_pressed=True, cruise_standstill=False
    )
    assert next_state == LongCtrlState.stopping

  def test_off_to_stopping_when_cruise_standstill(self):
    """OFF → STOPPING when cruise is in standstill."""
    CP = car.CarParams.new_message()
    next_state = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.off, v_ego=0.0, should_stop=False, brake_pressed=False, cruise_standstill=True
    )
    assert next_state == LongCtrlState.stopping

  def test_stopping_to_starting_with_starting_state(self):
    """STOPPING → STARTING when startingState=True and starting conditions met."""
    CP = car.CarParams.new_message(startingState=True, vEgoStarting=0.5)
    next_state = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.stopping, v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert next_state == LongCtrlState.starting

  def test_stopping_to_pid_without_starting_state(self):
    """STOPPING → PID when startingState=False and starting conditions met."""
    CP = car.CarParams.new_message(startingState=False)
    next_state = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.stopping, v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert next_state == LongCtrlState.pid

  def test_stopping_remains_when_should_stop(self):
    """STOPPING remains in STOPPING when should_stop is True."""
    CP = car.CarParams.new_message()
    next_state = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.stopping, v_ego=0.0, should_stop=True, brake_pressed=False, cruise_standstill=False
    )
    assert next_state == LongCtrlState.stopping

  def test_starting_to_stopping_when_should_stop(self):
    """STARTING → STOPPING when stopping condition becomes True."""
    CP = car.CarParams.new_message(startingState=True, vEgoStarting=0.5)
    next_state = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.starting, v_ego=0.3, should_stop=True, brake_pressed=False, cruise_standstill=False
    )
    assert next_state == LongCtrlState.stopping

  def test_starting_to_pid_when_started(self):
    """STARTING → PID when v_ego exceeds vEgoStarting."""
    CP = car.CarParams.new_message(startingState=True, vEgoStarting=0.5)
    next_state = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.starting, v_ego=1.0, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert next_state == LongCtrlState.pid

  def test_starting_remains_below_vEgoStarting(self):
    """STARTING remains in STARTING when v_ego <= vEgoStarting."""
    CP = car.CarParams.new_message(startingState=True, vEgoStarting=0.5)
    next_state = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.starting, v_ego=0.3, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert next_state == LongCtrlState.starting

  def test_pid_to_stopping_when_should_stop(self):
    """PID → STOPPING when stopping condition becomes True."""
    CP = car.CarParams.new_message()
    next_state = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.pid, v_ego=5.0, should_stop=True, brake_pressed=False, cruise_standstill=False
    )
    assert next_state == LongCtrlState.stopping

  def test_pid_remains_when_driving(self):
    """PID remains in PID during normal driving."""
    CP = car.CarParams.new_message()
    next_state = long_control_state_trans(
      CP, active=True, long_control_state=LongCtrlState.pid, v_ego=10.0, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert next_state == LongCtrlState.pid

  def test_any_state_to_off_when_inactive(self):
    """Any state → OFF when active=False."""
    CP = car.CarParams.new_message()
    for state in [LongCtrlState.off, LongCtrlState.stopping, LongCtrlState.starting, LongCtrlState.pid]:
      next_state = long_control_state_trans(
        CP, active=False, long_control_state=state, v_ego=5.0, should_stop=False, brake_pressed=False, cruise_standstill=False
      )
      assert next_state == LongCtrlState.off, f"Expected OFF from {state} when inactive"


class TestLongControlClass:
  """Tests for the LongControl class update() method behavior."""

  def _create_car_state(self, v_ego=0.0, a_ego=0.0, brake_pressed=False, cruise_standstill=False):
    """Create a CarState message with specified values."""
    CS = car.CarState.new_message()
    CS.vEgo = v_ego
    CS.aEgo = a_ego
    CS.brakePressed = brake_pressed
    CS.cruiseState.standstill = cruise_standstill
    return CS

  def _create_car_params(self, starting_state=False, v_ego_starting=0.5, stop_accel=-2.0, stopping_decel_rate=0.8, start_accel=1.2):
    """Create CarParams with longitudinal tuning."""
    CP = car.CarParams.new_message()
    CP.startingState = starting_state
    CP.vEgoStarting = v_ego_starting
    CP.stopAccel = stop_accel
    CP.stoppingDecelRate = stopping_decel_rate
    CP.startAccel = start_accel
    # Set up basic PID tuning
    CP.longitudinalTuning.kpBP = [0.0, 10.0]
    CP.longitudinalTuning.kpV = [1.0, 1.0]
    CP.longitudinalTuning.kiBP = [0.0, 10.0]
    CP.longitudinalTuning.kiV = [0.1, 0.1]
    return CP

  def test_off_state_returns_zero_accel(self):
    """OFF state should return zero accel and reset PID."""
    CP = self._create_car_params()
    lc = LongControl(CP)
    CS = self._create_car_state(v_ego=10.0)

    output = lc.update(active=False, CS=CS, a_target=2.0, should_stop=False, accel_limits=[-3.0, 2.0])

    assert output == 0.0
    assert lc.long_control_state == LongCtrlState.off

  def test_stopping_state_decelerates_gradually(self):
    """STOPPING state should decelerate gradually toward stopAccel."""
    CP = self._create_car_params(stop_accel=-2.0, stopping_decel_rate=0.8)
    lc = LongControl(CP)
    lc.last_output_accel = 0.0  # Start from zero accel
    CS = self._create_car_state(v_ego=0.5)

    # Engage in stopping state
    output = lc.update(active=True, CS=CS, a_target=0.0, should_stop=True, accel_limits=[-3.0, 2.0])

    assert lc.long_control_state == LongCtrlState.stopping
    # Should decelerate by stoppingDecelRate * DT_CTRL from last_output (0.0)
    expected = 0.0 - CP.stoppingDecelRate * DT_CTRL
    assert output == pytest.approx(expected, abs=0.001)

  def test_stopping_state_clamps_to_zero_before_decel(self):
    """STOPPING with positive last_output should clamp to zero before decelerating."""
    CP = self._create_car_params(stop_accel=-2.0, stopping_decel_rate=0.8)
    lc = LongControl(CP)
    lc.last_output_accel = 1.0  # Positive accel before stopping
    lc.long_control_state = LongCtrlState.stopping
    CS = self._create_car_state(v_ego=0.5)

    output = lc.update(active=True, CS=CS, a_target=0.0, should_stop=True, accel_limits=[-3.0, 2.0])

    # Should clamp to 0 then subtract stoppingDecelRate * DT_CTRL
    expected = 0.0 - CP.stoppingDecelRate * DT_CTRL
    assert output == pytest.approx(expected, abs=0.001)

  def test_stopping_state_holds_at_stop_accel(self):
    """STOPPING should hold at stopAccel once reached."""
    CP = self._create_car_params(stop_accel=-2.0, stopping_decel_rate=0.8)
    lc = LongControl(CP)
    lc.last_output_accel = -2.5  # Already below stopAccel
    lc.long_control_state = LongCtrlState.stopping
    CS = self._create_car_state(v_ego=0.1)

    output = lc.update(active=True, CS=CS, a_target=0.0, should_stop=True, accel_limits=[-3.0, 2.0])

    # Should maintain last_output_accel since it's <= stopAccel
    assert output == pytest.approx(-2.5, abs=0.001)

  def test_starting_state_uses_start_accel(self):
    """STARTING state should output startAccel."""
    CP = self._create_car_params(starting_state=True, start_accel=1.2, v_ego_starting=0.5)
    lc = LongControl(CP)
    CS = self._create_car_state(v_ego=0.1)

    # Engage from stopping to starting
    lc.long_control_state = LongCtrlState.stopping
    output = lc.update(active=True, CS=CS, a_target=0.0, should_stop=False, accel_limits=[-3.0, 2.0])

    assert lc.long_control_state == LongCtrlState.starting
    assert output == pytest.approx(CP.startAccel, abs=0.001)

  def test_pid_state_uses_pid_controller(self):
    """PID state should use PID controller with feedforward."""
    CP = self._create_car_params()
    lc = LongControl(CP)
    CS = self._create_car_state(v_ego=10.0, a_ego=0.0)

    # Engage directly to PID (no startingState, no should_stop)
    output = lc.update(active=True, CS=CS, a_target=1.0, should_stop=False, accel_limits=[-3.0, 2.0])

    assert lc.long_control_state == LongCtrlState.pid
    # Should be non-zero and use PID output
    assert output != 0.0

  def test_accel_clipped_to_limits(self):
    """Output accel should be clipped to accel_limits."""
    CP = self._create_car_params()
    lc = LongControl(CP)
    CS = self._create_car_state(v_ego=10.0, a_ego=0.0)

    # Request very high accel target
    output = lc.update(active=True, CS=CS, a_target=10.0, should_stop=False, accel_limits=[-3.0, 2.0])

    assert output <= 2.0  # Should not exceed positive limit

    # Request very negative accel target
    output = lc.update(active=True, CS=CS, a_target=-10.0, should_stop=False, accel_limits=[-3.0, 2.0])

    assert output >= -3.0  # Should not exceed negative limit

  def test_pid_reset_on_stopping(self):
    """PID should reset when entering STOPPING state."""
    CP = self._create_car_params()
    lc = LongControl(CP)
    CS = self._create_car_state(v_ego=10.0, a_ego=0.0)

    # Run in PID mode to accumulate integrator
    for _ in range(10):
      lc.update(active=True, CS=CS, a_target=2.0, should_stop=False, accel_limits=[-3.0, 2.0])

    assert lc.long_control_state == LongCtrlState.pid

    # Now enter stopping state
    CS_stopped = self._create_car_state(v_ego=0.1)
    lc.update(active=True, CS=CS_stopped, a_target=0.0, should_stop=True, accel_limits=[-3.0, 2.0])

    assert lc.long_control_state == LongCtrlState.stopping
    # PID should be reset
    assert lc.pid.i == 0.0

  def test_pid_reset_on_off(self):
    """PID should reset when transitioning to OFF state."""
    CP = self._create_car_params()
    lc = LongControl(CP)
    CS = self._create_car_state(v_ego=10.0, a_ego=0.0)

    # Run in PID mode
    for _ in range(10):
      lc.update(active=True, CS=CS, a_target=2.0, should_stop=False, accel_limits=[-3.0, 2.0])

    # Deactivate
    lc.update(active=False, CS=CS, a_target=0.0, should_stop=False, accel_limits=[-3.0, 2.0])

    assert lc.long_control_state == LongCtrlState.off
    assert lc.pid.i == 0.0

  def test_last_output_accel_persists(self):
    """last_output_accel should persist between updates."""
    CP = self._create_car_params()
    lc = LongControl(CP)
    CS = self._create_car_state(v_ego=10.0, a_ego=0.0)

    output1 = lc.update(active=True, CS=CS, a_target=1.5, should_stop=False, accel_limits=[-3.0, 2.0])

    assert lc.last_output_accel == output1

    output2 = lc.update(active=True, CS=CS, a_target=1.5, should_stop=False, accel_limits=[-3.0, 2.0])

    assert lc.last_output_accel == output2
