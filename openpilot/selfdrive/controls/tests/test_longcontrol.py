import pytest

from openpilot.common.test import OpenpilotTestCase
from opendbc.car.structs import car
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState, long_control_state_trans, LongControl
from openpilot.common.realtime import DT_CTRL

# fork: upstream removed CP.stoppingDecelRate from the stopping ramp in favour of a
# hardcoded 1.0 m/s^2/s. The field still exists in CarParams but is no longer read.
STOPPING_DECEL_RATE = 1.0


class TestLongControlStateTransition(OpenpilotTestCase):

  def test_stay_stopped(self):
    active = True
    current_state = LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid
    active = False
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.off

  def test_engage(self):
    active = True
    current_state = LongCtrlState.off
    next_state = long_control_state_trans(active, current_state,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid


class TestLongControlStateTransitionComplete:
  """Comprehensive tests for all state transitions in the longitudinal control state machine.

  fork: upstream dropped the STARTING state (and the CP/v_ego arguments that gated it),
  so the transitions into and out of STARTING that this class used to cover are gone.
  What remains below is every transition the state machine can still produce.
  """

  def test_off_to_pid_when_starting_conditions_met(self):
    """OFF -> PID when starting conditions are met."""
    next_state = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.off, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert next_state == LongCtrlState.pid

  def test_off_to_stopping_when_should_stop(self):
    """OFF -> STOPPING when should_stop is True."""
    next_state = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.off, should_stop=True, brake_pressed=False, cruise_standstill=False
    )
    assert next_state == LongCtrlState.stopping

  def test_off_to_stopping_when_brake_pressed(self):
    """OFF -> STOPPING when brake is pressed."""
    next_state = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.off, should_stop=False, brake_pressed=True, cruise_standstill=False
    )
    assert next_state == LongCtrlState.stopping

  def test_off_to_stopping_when_cruise_standstill(self):
    """OFF -> STOPPING when cruise is in standstill."""
    next_state = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.off, should_stop=False, brake_pressed=False, cruise_standstill=True
    )
    assert next_state == LongCtrlState.stopping

  def test_stopping_to_pid_when_starting_conditions_met(self):
    """STOPPING -> PID when starting conditions are met."""
    next_state = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.stopping, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert next_state == LongCtrlState.pid

  def test_stopping_remains_when_should_stop(self):
    """STOPPING remains in STOPPING when should_stop is True."""
    next_state = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.stopping, should_stop=True, brake_pressed=False, cruise_standstill=False
    )
    assert next_state == LongCtrlState.stopping

  def test_pid_to_stopping_when_should_stop(self):
    """PID -> STOPPING when stopping condition becomes True."""
    next_state = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.pid, should_stop=True, brake_pressed=False, cruise_standstill=False
    )
    assert next_state == LongCtrlState.stopping

  def test_pid_remains_when_driving(self):
    """PID remains in PID during normal driving."""
    next_state = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.pid, should_stop=False, brake_pressed=False, cruise_standstill=False
    )
    assert next_state == LongCtrlState.pid

  def test_pid_remains_when_brake_pressed(self):
    """PID remains in PID when only the brake is pressed (brake alone does not stop it)."""
    next_state = long_control_state_trans(
      active=True, long_control_state=LongCtrlState.pid, should_stop=False, brake_pressed=True, cruise_standstill=False
    )
    assert next_state == LongCtrlState.pid

  def test_any_state_to_off_when_inactive(self):
    """Any state -> OFF when active=False."""
    for state in [LongCtrlState.off, LongCtrlState.stopping, LongCtrlState.starting, LongCtrlState.pid]:
      next_state = long_control_state_trans(
        active=False, long_control_state=state, should_stop=False, brake_pressed=False, cruise_standstill=False
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

  def _create_car_params(self, stop_accel=-2.0):
    """Create CarParams with longitudinal tuning."""
    CP = car.CarParams.new_message()
    CP.stopAccel = stop_accel
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
    CP = self._create_car_params(stop_accel=-2.0)
    lc = LongControl(CP)
    lc.last_output_accel = 0.0  # Start from zero accel
    CS = self._create_car_state(v_ego=0.5)

    # Engage in stopping state
    output = lc.update(active=True, CS=CS, a_target=0.0, should_stop=True, accel_limits=[-3.0, 2.0])

    assert lc.long_control_state == LongCtrlState.stopping
    # Should decelerate by the stopping ramp rate * DT_CTRL from last_output (0.0)
    expected = 0.0 - STOPPING_DECEL_RATE * DT_CTRL
    assert output == pytest.approx(expected, abs=0.001)

  def test_stopping_state_clamps_to_zero_before_decel(self):
    """STOPPING with positive last_output should clamp to zero before decelerating."""
    CP = self._create_car_params(stop_accel=-2.0)
    lc = LongControl(CP)
    lc.last_output_accel = 1.0  # Positive accel before stopping
    lc.long_control_state = LongCtrlState.stopping
    CS = self._create_car_state(v_ego=0.5)

    output = lc.update(active=True, CS=CS, a_target=0.0, should_stop=True, accel_limits=[-3.0, 2.0])

    # Should clamp to 0 then subtract the stopping ramp rate * DT_CTRL
    expected = 0.0 - STOPPING_DECEL_RATE * DT_CTRL
    assert output == pytest.approx(expected, abs=0.001)

  def test_stopping_state_holds_at_stop_accel(self):
    """STOPPING should hold at stopAccel once reached."""
    CP = self._create_car_params(stop_accel=-2.0)
    lc = LongControl(CP)
    lc.last_output_accel = -2.5  # Already below stopAccel
    lc.long_control_state = LongCtrlState.stopping
    CS = self._create_car_state(v_ego=0.1)

    output = lc.update(active=True, CS=CS, a_target=0.0, should_stop=True, accel_limits=[-3.0, 2.0])

    # Should maintain last_output_accel since it's <= stopAccel
    assert output == pytest.approx(-2.5, abs=0.001)

  def test_pid_state_uses_pid_controller(self):
    """PID state should use PID controller with feedforward."""
    CP = self._create_car_params()
    lc = LongControl(CP)
    CS = self._create_car_state(v_ego=10.0, a_ego=0.0)

    # Engage directly to PID (no should_stop)
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
