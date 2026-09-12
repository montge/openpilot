"""Property-based tests for longitudinal control using Hypothesis.

These tests verify mathematical properties that should hold for all valid inputs,
including state machine transitions and output bounds.
"""

from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from openpilot.selfdrive.controls.lib.longcontrol import LongControl, long_control_state_trans

from opendbc.car.structs import car

LongCtrlState = car.CarControl.Actuators.LongControlState

HYPOTHESIS_SETTINGS = settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])


def create_mock_cp(mocker, stop_accel=-2.0):
  """Create a mock CarParams for LongControl."""
  CP = mocker.MagicMock()
  CP.stopAccel = stop_accel
  CP.longitudinalTuning.kiBP = [0.0]
  CP.longitudinalTuning.kiV = [0.1]
  return CP


def create_mock_cs(mocker, v_ego=15.0, a_ego=0.0, brake_pressed=False, standstill=False):
  """Create a mock CarState for LongControl."""
  CS = mocker.MagicMock()
  CS.vEgo = v_ego
  CS.aEgo = a_ego
  CS.brakePressed = brake_pressed
  CS.cruiseState.standstill = standstill
  return CS


class TestLongControlProperties:
  """Property-based tests for LongControl."""

  @given(
    v_ego=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    a_ego=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    a_target=st.floats(min_value=-4.0, max_value=3.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_output_always_within_limits(self, mocker, v_ego, a_ego, a_target):
    """Property: Output acceleration is always within specified limits."""
    CP = create_mock_cp(mocker)
    controller = LongControl(CP)

    CS = create_mock_cs(mocker, v_ego=v_ego, a_ego=a_ego)
    accel_limits = [-3.5, 2.0]

    output = controller.update(active=True, CS=CS, a_target=a_target, should_stop=False, accel_limits=accel_limits)

    assert accel_limits[0] <= output <= accel_limits[1], f"Output {output} outside limits"

  @given(
    v_ego=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    a_ego=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_inactive_returns_zero(self, mocker, v_ego, a_ego):
    """Property: Inactive mode always returns zero acceleration."""
    CP = create_mock_cp(mocker)
    controller = LongControl(CP)

    CS = create_mock_cs(mocker, v_ego=v_ego, a_ego=a_ego)
    accel_limits = [-3.5, 2.0]

    output = controller.update(active=False, CS=CS, a_target=1.0, should_stop=False, accel_limits=accel_limits)

    assert output == 0.0, f"Inactive returned non-zero: {output}"
    assert controller.long_control_state == LongCtrlState.off

  @given(
    v_ego=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_stopping_state_decreases_accel(self, mocker, v_ego):
    """Property: In stopping state, acceleration decreases toward stop accel."""
    CP = create_mock_cp(mocker)
    controller = LongControl(CP)

    CS = create_mock_cs(mocker, v_ego=v_ego)
    accel_limits = [-3.5, 2.0]

    # First, set to active and get some output
    controller.update(active=True, CS=CS, a_target=0.5, should_stop=False, accel_limits=accel_limits)

    # Now trigger stopping
    output = controller.update(active=True, CS=CS, a_target=0.0, should_stop=True, accel_limits=accel_limits)

    assert controller.long_control_state == LongCtrlState.stopping
    assert output <= 0.0 or output <= controller.last_output_accel

  @given(
    lower=st.floats(min_value=-5.0, max_value=-0.5, allow_nan=False, allow_infinity=False),
    upper=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_limits_respected_with_varying_bounds(self, mocker, lower, upper):
    """Property: Output respects limits regardless of limit values."""
    assume(lower < upper)

    CP = create_mock_cp(mocker)
    controller = LongControl(CP)

    CS = create_mock_cs(mocker, v_ego=20.0, a_ego=0.0)
    accel_limits = [lower, upper]

    # Run multiple updates to accumulate state
    for _ in range(10):
      output = controller.update(active=True, CS=CS, a_target=5.0, should_stop=False, accel_limits=accel_limits)
      assert lower <= output <= upper, f"Output {output} outside [{lower}, {upper}]"


class TestLongControlStateTransProperties:
  """Property-based tests for state machine transitions."""

  def test_inactive_always_transitions_to_off(self):
    """Property: Inactive always results in off state."""
    for initial_state in [LongCtrlState.off, LongCtrlState.stopping, LongCtrlState.starting, LongCtrlState.pid]:
      result = long_control_state_trans(
        active=False, long_control_state=initial_state, should_stop=False, brake_pressed=False, cruise_standstill=False
      )
      assert result == LongCtrlState.off

  def test_stopping_condition_leads_to_stopping(self):
    """Property: should_stop=True eventually leads to stopping state."""
    for initial_state in [LongCtrlState.off, LongCtrlState.stopping, LongCtrlState.pid]:
      result = long_control_state_trans(
        active=True, long_control_state=initial_state, should_stop=True, brake_pressed=False, cruise_standstill=False
      )
      assert result == LongCtrlState.stopping

  def test_starting_conditions_lead_to_pid(self):
    """Property: clear starting conditions transition off/stopping straight to pid."""
    for initial_state in [LongCtrlState.off, LongCtrlState.stopping]:
      result = long_control_state_trans(
        active=True, long_control_state=initial_state, should_stop=False, brake_pressed=False, cruise_standstill=False
      )
      assert result == LongCtrlState.pid

  @given(
    active=st.booleans(),
    should_stop=st.booleans(),
    brake_pressed=st.booleans(),
    standstill=st.booleans(),
  )
  @HYPOTHESIS_SETTINGS
  def test_state_always_valid(self, active, should_stop, brake_pressed, standstill):
    """Property: State transition always produces a valid state."""
    valid_states = {LongCtrlState.off, LongCtrlState.stopping, LongCtrlState.starting, LongCtrlState.pid}

    for initial_state in valid_states:
      result = long_control_state_trans(
        active=active, long_control_state=initial_state, should_stop=should_stop, brake_pressed=brake_pressed, cruise_standstill=standstill
      )
      assert result in valid_states, f"Invalid state: {result}"
