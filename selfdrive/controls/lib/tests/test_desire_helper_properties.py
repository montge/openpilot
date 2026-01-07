"""Property-based tests for desire_helper.py using Hypothesis.

These tests verify mathematical properties that should hold for all valid inputs,
complementing the unit tests with broader coverage.
"""

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from cereal import log

from openpilot.selfdrive.controls.lib.desire_helper import (
  DesireHelper,
  LANE_CHANGE_SPEED_MIN,
  LANE_CHANGE_TIME_MAX,
)

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

HYPOTHESIS_SETTINGS = settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])


def create_mock_carstate(
  mocker, v_ego=15.0, left_blinker=False, right_blinker=False, steering_pressed=False, steering_torque=0, left_blindspot=False, right_blindspot=False
):
  """Create a mock CarState for testing."""
  CS = mocker.MagicMock()
  CS.vEgo = v_ego
  CS.leftBlinker = left_blinker
  CS.rightBlinker = right_blinker
  CS.steeringPressed = steering_pressed
  CS.steeringTorque = steering_torque
  CS.leftBlindspot = left_blindspot
  CS.rightBlindspot = right_blindspot
  return CS


class TestDesireHelperLLProbProperties:
  """Property-based tests for lane_change_ll_prob."""

  @given(
    v_ego=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    left_blinker=st.booleans(),
    right_blinker=st.booleans(),
    lane_change_prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_ll_prob_bounded_0_1(self, mocker, v_ego, left_blinker, right_blinker, lane_change_prob):
    """Property: lane_change_ll_prob is always in [0.0, 1.0]."""
    helper = DesireHelper()

    CS = create_mock_carstate(mocker, v_ego=v_ego, left_blinker=left_blinker, right_blinker=right_blinker)

    # Run multiple updates
    for _ in range(20):
      helper.update(CS, lateral_active=True, lane_change_prob=lane_change_prob)

      assert 0.0 <= helper.lane_change_ll_prob <= 1.0

  @given(
    v_ego=st.floats(min_value=LANE_CHANGE_SPEED_MIN, max_value=50.0, allow_nan=False, allow_infinity=False),
    lane_change_prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_ll_prob_fades_in_starting_state(self, mocker, v_ego, lane_change_prob):
    """Property: ll_prob decreases in laneChangeStarting state."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeStarting
    helper.lane_change_direction = LaneChangeDirection.left
    helper.lane_change_ll_prob = 1.0

    CS = create_mock_carstate(mocker, v_ego=v_ego, left_blinker=True)

    initial_prob = helper.lane_change_ll_prob
    helper.update(CS, lateral_active=True, lane_change_prob=lane_change_prob)

    assert helper.lane_change_ll_prob <= initial_prob

  @given(
    v_ego=st.floats(min_value=LANE_CHANGE_SPEED_MIN, max_value=50.0, allow_nan=False, allow_infinity=False),
    lane_change_prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_ll_prob_fades_out_finishing_state(self, mocker, v_ego, lane_change_prob):
    """Property: ll_prob increases in laneChangeFinishing state."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeFinishing
    helper.lane_change_direction = LaneChangeDirection.left
    helper.lane_change_ll_prob = 0.5

    CS = create_mock_carstate(mocker, v_ego=v_ego, left_blinker=True)

    initial_prob = helper.lane_change_ll_prob
    helper.update(CS, lateral_active=True, lane_change_prob=lane_change_prob)

    assert helper.lane_change_ll_prob >= initial_prob


class TestDesireHelperTimerProperties:
  """Property-based tests for lane_change_timer."""

  @given(
    v_ego=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    left_blinker=st.booleans(),
    right_blinker=st.booleans(),
    lane_change_prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_timer_non_negative(self, mocker, v_ego, left_blinker, right_blinker, lane_change_prob):
    """Property: lane_change_timer is always >= 0."""
    helper = DesireHelper()

    CS = create_mock_carstate(mocker, v_ego=v_ego, left_blinker=left_blinker, right_blinker=right_blinker)

    for _ in range(20):
      helper.update(CS, lateral_active=True, lane_change_prob=lane_change_prob)

      assert helper.lane_change_timer >= 0.0

  @given(
    v_ego=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    lane_change_prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_timer_zero_when_off(self, mocker, v_ego, lane_change_prob):
    """Property: lane_change_timer is 0 in off state."""
    helper = DesireHelper()
    helper.lane_change_timer = 5.0  # Non-zero

    CS = create_mock_carstate(mocker, v_ego=v_ego)  # No blinkers
    helper.update(CS, lateral_active=True, lane_change_prob=lane_change_prob)

    # Should be off with no blinkers
    if helper.lane_change_state == LaneChangeState.off:
      assert helper.lane_change_timer == 0.0

  @given(
    v_ego=st.floats(min_value=LANE_CHANGE_SPEED_MIN + 1, max_value=50.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_timer_zero_in_pre_lane_change(self, mocker, v_ego):
    """Property: lane_change_timer is 0 in preLaneChange state."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.preLaneChange
    helper.lane_change_direction = LaneChangeDirection.left
    helper.lane_change_timer = 5.0
    helper.prev_one_blinker = True

    CS = create_mock_carstate(mocker, v_ego=v_ego, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    if helper.lane_change_state == LaneChangeState.preLaneChange:
      assert helper.lane_change_timer == 0.0


class TestDesireHelperStateProperties:
  """Property-based tests for state machine properties."""

  @given(
    v_ego=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    left_blinker=st.booleans(),
    right_blinker=st.booleans(),
  )
  @HYPOTHESIS_SETTINGS
  def test_not_lateral_active_goes_off(self, mocker, v_ego, left_blinker, right_blinker):
    """Property: Not lateral_active always results in off state."""
    helper = DesireHelper()
    # Set to some active state
    helper.lane_change_state = LaneChangeState.laneChangeStarting
    helper.lane_change_direction = LaneChangeDirection.left

    CS = create_mock_carstate(mocker, v_ego=v_ego, left_blinker=left_blinker, right_blinker=right_blinker)
    helper.update(CS, lateral_active=False, lane_change_prob=0.5)

    assert helper.lane_change_state == LaneChangeState.off
    assert helper.lane_change_direction == LaneChangeDirection.none

  @given(
    v_ego=st.floats(min_value=0.0, max_value=LANE_CHANGE_SPEED_MIN - 0.1, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_below_speed_stays_off(self, mocker, v_ego):
    """Property: Below LANE_CHANGE_SPEED_MIN stays off."""
    helper = DesireHelper()
    helper.prev_one_blinker = False

    CS = create_mock_carstate(mocker, v_ego=v_ego, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.lane_change_state == LaneChangeState.off

  @given(
    v_ego=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_timer_exceeded_goes_off(self, mocker, v_ego):
    """Property: Exceeding LANE_CHANGE_TIME_MAX goes to off."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeStarting
    helper.lane_change_direction = LaneChangeDirection.left
    helper.lane_change_timer = LANE_CHANGE_TIME_MAX + 0.1

    CS = create_mock_carstate(mocker, v_ego=v_ego, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.lane_change_state == LaneChangeState.off


class TestDesireHelperDesireProperties:
  """Property-based tests for desire output."""

  @given(
    v_ego=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    left_blinker=st.booleans(),
    right_blinker=st.booleans(),
    lane_change_prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_desire_is_valid(self, mocker, v_ego, left_blinker, right_blinker, lane_change_prob):
    """Property: desire is always a valid Desire enum value."""
    helper = DesireHelper()

    CS = create_mock_carstate(mocker, v_ego=v_ego, left_blinker=left_blinker, right_blinker=right_blinker)

    for _ in range(10):
      helper.update(CS, lateral_active=True, lane_change_prob=lane_change_prob)

      # Desire should be a valid log.Desire value
      valid_desires = {
        log.Desire.none,
        log.Desire.laneChangeLeft,
        log.Desire.laneChangeRight,
        log.Desire.keepLeft,
        log.Desire.keepRight,
      }
      assert helper.desire in valid_desires

  @given(
    v_ego=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    lane_change_prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_off_state_none_desire(self, mocker, v_ego, lane_change_prob):
    """Property: Off state always has none desire."""
    helper = DesireHelper()

    CS = create_mock_carstate(mocker, v_ego=v_ego)  # No blinkers
    helper.update(CS, lateral_active=True, lane_change_prob=lane_change_prob)

    if helper.lane_change_state == LaneChangeState.off:
      assert helper.desire == log.Desire.none


class TestDesireHelperDirectionProperties:
  """Property-based tests for lane_change_direction."""

  @given(
    v_ego=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    lane_change_prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_direction_none_when_off(self, mocker, v_ego, lane_change_prob):
    """Property: Direction is none when state is off."""
    helper = DesireHelper()

    CS = create_mock_carstate(mocker, v_ego=v_ego)  # No blinkers
    helper.update(CS, lateral_active=True, lane_change_prob=lane_change_prob)

    if helper.lane_change_state == LaneChangeState.off:
      assert helper.lane_change_direction == LaneChangeDirection.none

  @given(
    v_ego=st.floats(min_value=LANE_CHANGE_SPEED_MIN + 1, max_value=50.0, allow_nan=False, allow_infinity=False),
    lane_change_prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_left_blinker_left_direction(self, mocker, v_ego, lane_change_prob):
    """Property: Left blinker results in left direction."""
    helper = DesireHelper()
    helper.prev_one_blinker = False

    CS = create_mock_carstate(mocker, v_ego=v_ego, left_blinker=True, right_blinker=False)
    helper.update(CS, lateral_active=True, lane_change_prob=lane_change_prob)

    # When starting lane change with left blinker, direction should be left
    if helper.lane_change_state in (LaneChangeState.preLaneChange, LaneChangeState.laneChangeStarting):
      assert helper.lane_change_direction == LaneChangeDirection.left

  @given(
    v_ego=st.floats(min_value=LANE_CHANGE_SPEED_MIN + 1, max_value=50.0, allow_nan=False, allow_infinity=False),
    lane_change_prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_right_blinker_right_direction(self, mocker, v_ego, lane_change_prob):
    """Property: Right blinker results in right direction."""
    helper = DesireHelper()
    helper.prev_one_blinker = False

    CS = create_mock_carstate(mocker, v_ego=v_ego, left_blinker=False, right_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=lane_change_prob)

    # When starting lane change with right blinker, direction should be right
    if helper.lane_change_state in (LaneChangeState.preLaneChange, LaneChangeState.laneChangeStarting):
      assert helper.lane_change_direction == LaneChangeDirection.right


class TestDesireHelperKeepPulseProperties:
  """Property-based tests for keep_pulse_timer."""

  @given(
    v_ego=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    left_blinker=st.booleans(),
    right_blinker=st.booleans(),
    lane_change_prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_keep_pulse_timer_non_negative(self, mocker, v_ego, left_blinker, right_blinker, lane_change_prob):
    """Property: keep_pulse_timer is always >= 0."""
    helper = DesireHelper()

    CS = create_mock_carstate(mocker, v_ego=v_ego, left_blinker=left_blinker, right_blinker=right_blinker)

    for _ in range(20):
      helper.update(CS, lateral_active=True, lane_change_prob=lane_change_prob)

      assert helper.keep_pulse_timer >= 0.0

  @given(
    v_ego=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    left_blinker=st.booleans(),
    right_blinker=st.booleans(),
    lane_change_prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_keep_pulse_timer_bounded(self, mocker, v_ego, left_blinker, right_blinker, lane_change_prob):
    """Property: keep_pulse_timer resets at 1.0s, so it's bounded < 1.1."""
    helper = DesireHelper()

    CS = create_mock_carstate(mocker, v_ego=v_ego, left_blinker=left_blinker, right_blinker=right_blinker)

    for _ in range(100):
      helper.update(CS, lateral_active=True, lane_change_prob=lane_change_prob)

      # Timer should reset when exceeding 1.0, plus at most one DT_MDL increment
      assert helper.keep_pulse_timer < 1.1
