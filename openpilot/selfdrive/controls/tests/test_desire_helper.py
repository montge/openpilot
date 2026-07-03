"""Tests for desire_helper.py - lane change state machine."""

import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper, LANE_CHANGE_TIME_MAX, LANE_CHANGE_START_TIME, LaneChangeState, LaneChangeDirection


def make_carstate(
  mocker, v_ego=25.0, left_blinker=False, right_blinker=False, steering_pressed=False, steering_torque=0.0, left_blindspot=False, right_blindspot=False
):
  """Create a mock CarState."""
  cs = mocker.MagicMock()
  cs.vEgo = v_ego
  cs.leftBlinker = left_blinker
  cs.rightBlinker = right_blinker
  cs.steeringPressed = steering_pressed
  cs.steeringTorque = steering_torque
  cs.leftBlindspot = left_blindspot
  cs.rightBlindspot = right_blindspot
  return cs


class TestDesireHelperInit:
  """Test DesireHelper initialization."""

  def test_init_defaults(self):
    """Test initial state is correct."""
    dh = DesireHelper()
    assert dh.lane_change_state == LaneChangeState.off
    assert dh.lane_change_direction == LaneChangeDirection.none
    assert dh.lane_change_timer == 0.0
    assert dh.prev_one_blinker is False
    assert dh.desire == log.Desire.none


class TestGetLaneChangeDirection:
  """Test get_lane_change_direction static method."""

  def test_left_blinker_returns_left(self, mocker):
    """Test left blinker returns left direction."""
    cs = make_carstate(mocker, left_blinker=True, right_blinker=False)
    direction = DesireHelper.get_lane_change_direction(cs)
    assert direction == LaneChangeDirection.left

  def test_right_blinker_returns_right(self, mocker):
    """Test right blinker returns right direction."""
    cs = make_carstate(mocker, left_blinker=False, right_blinker=True)
    direction = DesireHelper.get_lane_change_direction(cs)
    assert direction == LaneChangeDirection.right

  def test_no_blinker_returns_right(self, mocker):
    """Test no blinker returns right (default)."""
    cs = make_carstate(mocker, left_blinker=False, right_blinker=False)
    direction = DesireHelper.get_lane_change_direction(cs)
    assert direction == LaneChangeDirection.right


class TestDesireHelperStateOff:
  """Test transitions from LaneChangeState.off."""

  def test_off_to_prelanechange_left_blinker(self, mocker):
    """Test off->preLaneChange on left blinker activation."""
    dh = DesireHelper()
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True)

    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    assert dh.lane_change_state == LaneChangeState.preLaneChange
    assert dh.lane_change_direction == LaneChangeDirection.left

  def test_off_to_prelanechange_right_blinker(self, mocker):
    """Test off->preLaneChange on right blinker activation."""
    dh = DesireHelper()
    cs = make_carstate(mocker, v_ego=25.0, right_blinker=True)

    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    assert dh.lane_change_state == LaneChangeState.preLaneChange
    assert dh.lane_change_direction == LaneChangeDirection.right

  def test_off_stays_off_below_speed(self, mocker):
    """Test stays off when below lane change speed."""
    dh = DesireHelper()
    cs = make_carstate(mocker, v_ego=5.0, left_blinker=True)  # Below LANE_CHANGE_SPEED_MIN

    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    assert dh.lane_change_state == LaneChangeState.off

  def test_off_stays_off_both_blinkers(self, mocker):
    """Test stays off when both blinkers on (hazards)."""
    dh = DesireHelper()
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True, right_blinker=True)

    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    assert dh.lane_change_state == LaneChangeState.off

  def test_off_stays_off_lateral_not_active(self, mocker):
    """Test stays off when lateral control not active."""
    dh = DesireHelper()
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True)

    dh.update(cs, lateral_active=False, lane_change_prob=0.5)

    assert dh.lane_change_state == LaneChangeState.off

  def test_off_stays_off_blinker_already_on(self, mocker):
    """Test stays off if blinker was already on."""
    dh = DesireHelper()
    dh.prev_one_blinker = True  # Blinker already on
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True)

    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    assert dh.lane_change_state == LaneChangeState.off


class TestDesireHelperStatePreLaneChange:
  """Test transitions from LaneChangeState.preLaneChange."""

  @pytest.fixture
  def setup_prelanechange(self, mocker):
    """Set up helper in preLaneChange state."""
    dh = DesireHelper()
    # Get into preLaneChange state
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    assert dh.lane_change_state == LaneChangeState.preLaneChange
    return dh

  def test_prelanechange_to_starting_with_torque(self, mocker, setup_prelanechange):
    """Test preLaneChange->laneChangeStarting with steering torque."""
    dh = setup_prelanechange
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True, steering_pressed=True, steering_torque=10.0)  # Positive for left

    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    assert dh.lane_change_state == LaneChangeState.laneChangeStarting

  def test_prelanechange_to_starting_right_lane_change(self, mocker):
    """Test preLaneChange->laneChangeStarting for right lane change."""
    # Reset and go to right preLaneChange
    dh = DesireHelper()
    cs = make_carstate(mocker, v_ego=25.0, right_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    # Apply negative torque for right
    cs = make_carstate(mocker, v_ego=25.0, right_blinker=True, steering_pressed=True, steering_torque=-10.0)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    assert dh.lane_change_state == LaneChangeState.laneChangeStarting
    assert dh.lane_change_direction == LaneChangeDirection.right

  def test_prelanechange_blocked_by_blindspot_left(self, mocker, setup_prelanechange):
    """Test lane change blocked by left blindspot."""
    dh = setup_prelanechange
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True, steering_pressed=True, steering_torque=10.0, left_blindspot=True)

    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    assert dh.lane_change_state == LaneChangeState.preLaneChange

  def test_prelanechange_blocked_by_blindspot_right(self, mocker):
    """Test lane change blocked by right blindspot."""
    dh = DesireHelper()
    cs = make_carstate(mocker, v_ego=25.0, right_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    cs = make_carstate(mocker, v_ego=25.0, right_blinker=True, steering_pressed=True, steering_torque=-10.0, right_blindspot=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    assert dh.lane_change_state == LaneChangeState.preLaneChange

  def test_prelanechange_to_off_blinker_removed(self, mocker, setup_prelanechange):
    """Test preLaneChange->off when blinker removed."""
    dh = setup_prelanechange
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=False)

    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    assert dh.lane_change_state == LaneChangeState.off
    assert dh.lane_change_direction == LaneChangeDirection.none

  def test_prelanechange_to_off_below_speed(self, mocker, setup_prelanechange):
    """Test preLaneChange->off when below speed."""
    dh = setup_prelanechange
    cs = make_carstate(mocker, v_ego=5.0, left_blinker=True)

    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    assert dh.lane_change_state == LaneChangeState.off

  def test_prelanechange_updates_direction(self, mocker):
    """Test that direction can be updated in preLaneChange."""
    # Start with left blinker
    dh = DesireHelper()
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    assert dh.lane_change_direction == LaneChangeDirection.left

    # Change to right blinker
    cs = make_carstate(mocker, v_ego=25.0, right_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    assert dh.lane_change_direction == LaneChangeDirection.right


class TestDesireHelperStateLaneChangeStarting:
  """Test transitions from LaneChangeState.laneChangeStarting."""

  @pytest.fixture
  def setup_starting(self, mocker):
    """Set up helper in laneChangeStarting state."""
    dh = DesireHelper()
    # Get into preLaneChange state
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    # Get into laneChangeStarting state
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True, steering_pressed=True, steering_torque=10.0)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    assert dh.lane_change_state == LaneChangeState.laneChangeStarting
    return dh

  def test_starting_to_prelanechange_on_completion_with_blinker(self, mocker, setup_starting):
    """Test laneChangeStarting->preLaneChange when lane change completes with blinker still on."""
    dh = setup_starting
    # Simulate lane change completing
    dh.lane_change_timer = LANE_CHANGE_START_TIME
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True)

    dh.update(cs, lateral_active=True, lane_change_prob=0.01)  # Below 0.02

    assert dh.lane_change_state == LaneChangeState.preLaneChange
    assert dh.lane_change_direction == LaneChangeDirection.left
    assert dh.lane_change_timer == 0.0

  def test_starting_to_off_on_completion_without_blinker(self, mocker, setup_starting):
    """Test laneChangeStarting->off when lane change completes and blinker is off."""
    dh = setup_starting
    dh.lane_change_timer = LANE_CHANGE_START_TIME
    cs = make_carstate(mocker, v_ego=25.0)  # No blinker

    dh.update(cs, lateral_active=True, lane_change_prob=0.01)  # Below 0.02

    assert dh.lane_change_state == LaneChangeState.off
    assert dh.lane_change_direction == LaneChangeDirection.none

  def test_starting_holds_low_probability_before_start_time(self, mocker, setup_starting):
    """Test stays in laneChangeStarting with low prob before LANE_CHANGE_START_TIME elapses."""
    dh = setup_starting
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True)

    dh.update(cs, lateral_active=True, lane_change_prob=0.01)  # Below 0.02, but timer < LANE_CHANGE_START_TIME

    assert dh.lane_change_state == LaneChangeState.laneChangeStarting

  def test_starting_stays_starting_high_probability(self, mocker, setup_starting):
    """Test stays in laneChangeStarting with high probability."""
    dh = setup_starting
    dh.lane_change_timer = LANE_CHANGE_START_TIME
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True)

    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    assert dh.lane_change_state == LaneChangeState.laneChangeStarting

  def test_starting_timer_increments(self, mocker, setup_starting):
    """Test lane change timer increments during starting."""
    dh = setup_starting
    initial_timer = dh.lane_change_timer
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True)

    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    assert dh.lane_change_timer > initial_timer


class TestDesireHelperTimeout:
  """Test lane change timeout behavior."""

  def test_timeout_resets_to_off(self, mocker):
    """Test that exceeding timeout resets state."""
    dh = DesireHelper()
    # Get into laneChangeStarting state
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True, steering_pressed=True, steering_torque=10.0)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    # Simulate timeout
    dh.lane_change_timer = LANE_CHANGE_TIME_MAX + 1.0
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    assert dh.lane_change_state == LaneChangeState.off
    assert dh.lane_change_direction == LaneChangeDirection.none


class TestDesireHelperDesire:
  """Test desire output based on state."""

  def test_desire_none_when_off(self, mocker):
    """Test desire is none when off."""
    dh = DesireHelper()
    cs = make_carstate(mocker, v_ego=25.0)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    assert dh.desire == log.Desire.none

  def test_desire_none_when_prelanechange(self, mocker):
    """Test desire stays none while waiting in preLaneChange."""
    dh = DesireHelper()
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    assert dh.lane_change_state == LaneChangeState.preLaneChange
    assert dh.desire == log.Desire.none

  def test_desire_lane_change_left_when_starting_left(self, mocker):
    """Test desire is laneChangeLeft when starting left lane change."""
    dh = DesireHelper()
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    cs = make_carstate(mocker, v_ego=25.0, left_blinker=True, steering_pressed=True, steering_torque=10.0)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    assert dh.lane_change_state == LaneChangeState.laneChangeStarting
    assert dh.desire == log.Desire.laneChangeLeft

  def test_desire_lane_change_right_when_starting_right(self, mocker):
    """Test desire is laneChangeRight when starting right lane change."""
    dh = DesireHelper()
    cs = make_carstate(mocker, v_ego=25.0, right_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    cs = make_carstate(mocker, v_ego=25.0, right_blinker=True, steering_pressed=True, steering_torque=-10.0)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    assert dh.lane_change_state == LaneChangeState.laneChangeStarting
    assert dh.desire == log.Desire.laneChangeRight
