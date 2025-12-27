"""Tests for selfdrive/controls/lib/desire_helper.py - lane change desire helper."""
import unittest
from unittest.mock import MagicMock

from cereal import log

from openpilot.selfdrive.controls.lib.desire_helper import (
  DesireHelper, DESIRES, LANE_CHANGE_SPEED_MIN, LANE_CHANGE_TIME_MAX,
)

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection


def create_mock_carstate(v_ego=15.0, left_blinker=False, right_blinker=False,
                         steering_pressed=False, steering_torque=0,
                         left_blindspot=False, right_blindspot=False):
  """Create a mock CarState for testing."""
  CS = MagicMock()
  CS.vEgo = v_ego
  CS.leftBlinker = left_blinker
  CS.rightBlinker = right_blinker
  CS.steeringPressed = steering_pressed
  CS.steeringTorque = steering_torque
  CS.leftBlindspot = left_blindspot
  CS.rightBlindspot = right_blindspot
  return CS


class TestDesireHelperInit(unittest.TestCase):
  """Test DesireHelper initialization."""

  def test_init(self):
    """Test DesireHelper initializes correctly."""
    helper = DesireHelper()

    self.assertEqual(helper.lane_change_state, LaneChangeState.off)
    self.assertEqual(helper.lane_change_direction, LaneChangeDirection.none)
    self.assertEqual(helper.lane_change_timer, 0.0)
    self.assertEqual(helper.lane_change_ll_prob, 1.0)
    self.assertEqual(helper.keep_pulse_timer, 0.0)
    self.assertFalse(helper.prev_one_blinker)
    self.assertEqual(helper.desire, log.Desire.none)


class TestGetLaneChangeDirection(unittest.TestCase):
  """Test get_lane_change_direction static method."""

  def test_left_blinker_returns_left(self):
    """Test left blinker returns left direction."""
    CS = create_mock_carstate(left_blinker=True, right_blinker=False)
    result = DesireHelper.get_lane_change_direction(CS)
    self.assertEqual(result, LaneChangeDirection.left)

  def test_right_blinker_returns_right(self):
    """Test right blinker returns right direction."""
    CS = create_mock_carstate(left_blinker=False, right_blinker=True)
    result = DesireHelper.get_lane_change_direction(CS)
    self.assertEqual(result, LaneChangeDirection.right)

  def test_no_blinker_returns_right(self):
    """Test no blinker returns right (default)."""
    CS = create_mock_carstate(left_blinker=False, right_blinker=False)
    result = DesireHelper.get_lane_change_direction(CS)
    self.assertEqual(result, LaneChangeDirection.right)


class TestDesireHelperUpdate(unittest.TestCase):
  """Test DesireHelper update method."""

  def test_not_lateral_active_goes_off(self):
    """Test not lateral active sets state to off."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.preLaneChange
    helper.lane_change_direction = LaneChangeDirection.left

    CS = create_mock_carstate()
    helper.update(CS, lateral_active=False, lane_change_prob=0.0)

    self.assertEqual(helper.lane_change_state, LaneChangeState.off)
    self.assertEqual(helper.lane_change_direction, LaneChangeDirection.none)

  def test_timer_exceeded_goes_off(self):
    """Test timer exceeded sets state to off."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeStarting
    helper.lane_change_timer = LANE_CHANGE_TIME_MAX + 1

    CS = create_mock_carstate()
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(helper.lane_change_state, LaneChangeState.off)

  def test_blinker_starts_pre_lane_change(self):
    """Test blinker starts preLaneChange from off."""
    helper = DesireHelper()
    helper.prev_one_blinker = False

    CS = create_mock_carstate(v_ego=LANE_CHANGE_SPEED_MIN + 1, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(helper.lane_change_state, LaneChangeState.preLaneChange)
    self.assertEqual(helper.lane_change_direction, LaneChangeDirection.left)

  def test_blinker_below_speed_stays_off(self):
    """Test blinker below min speed stays off."""
    helper = DesireHelper()
    helper.prev_one_blinker = False

    CS = create_mock_carstate(v_ego=LANE_CHANGE_SPEED_MIN - 1, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(helper.lane_change_state, LaneChangeState.off)

  def test_pre_lane_change_no_blinker_goes_off(self):
    """Test preLaneChange goes off when blinker released."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.preLaneChange
    helper.lane_change_direction = LaneChangeDirection.left
    helper.prev_one_blinker = True

    CS = create_mock_carstate(v_ego=LANE_CHANGE_SPEED_MIN + 1)  # No blinker
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(helper.lane_change_state, LaneChangeState.off)
    self.assertEqual(helper.lane_change_direction, LaneChangeDirection.none)

  def test_pre_lane_change_torque_starts_change(self):
    """Test preLaneChange with torque starts lane change."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.preLaneChange
    helper.lane_change_direction = LaneChangeDirection.left
    helper.prev_one_blinker = True

    CS = create_mock_carstate(
      v_ego=LANE_CHANGE_SPEED_MIN + 1,
      left_blinker=True,
      steering_pressed=True,
      steering_torque=10  # Positive for left
    )
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(helper.lane_change_state, LaneChangeState.laneChangeStarting)

  def test_pre_lane_change_blindspot_blocks_change(self):
    """Test preLaneChange with blindspot doesn't start change."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.preLaneChange
    helper.lane_change_direction = LaneChangeDirection.left
    helper.prev_one_blinker = True

    CS = create_mock_carstate(
      v_ego=LANE_CHANGE_SPEED_MIN + 1,
      left_blinker=True,
      steering_pressed=True,
      steering_torque=10,
      left_blindspot=True  # Blocking
    )
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(helper.lane_change_state, LaneChangeState.preLaneChange)

  def test_lane_change_starting_fades_ll_prob(self):
    """Test laneChangeStarting fades lane line probability."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeStarting
    helper.lane_change_ll_prob = 1.0
    helper.lane_change_direction = LaneChangeDirection.left

    CS = create_mock_carstate(v_ego=LANE_CHANGE_SPEED_MIN + 1, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    self.assertLess(helper.lane_change_ll_prob, 1.0)

  def test_lane_change_starting_to_finishing(self):
    """Test laneChangeStarting transitions to finishing."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeStarting
    helper.lane_change_ll_prob = 0.0
    helper.lane_change_direction = LaneChangeDirection.left

    CS = create_mock_carstate(v_ego=LANE_CHANGE_SPEED_MIN + 1, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.01)  # < 0.02

    self.assertEqual(helper.lane_change_state, LaneChangeState.laneChangeFinishing)

  def test_lane_change_finishing_fades_in_ll_prob(self):
    """Test laneChangeFinishing fades in lane line probability."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeFinishing
    helper.lane_change_ll_prob = 0.5
    helper.lane_change_direction = LaneChangeDirection.left

    CS = create_mock_carstate(v_ego=LANE_CHANGE_SPEED_MIN + 1, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    self.assertGreater(helper.lane_change_ll_prob, 0.5)

  def test_lane_change_timer_increments_during_change(self):
    """Test lane change timer increments during lane change."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeStarting
    helper.lane_change_timer = 0.0
    helper.lane_change_direction = LaneChangeDirection.left

    CS = create_mock_carstate(v_ego=LANE_CHANGE_SPEED_MIN + 1, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    self.assertGreater(helper.lane_change_timer, 0.0)

  def test_lane_change_timer_resets_off(self):
    """Test lane change timer resets when off."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.off
    helper.lane_change_timer = 5.0

    CS = create_mock_carstate()
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(helper.lane_change_timer, 0.0)


class TestDesireHelperDesire(unittest.TestCase):
  """Test DesireHelper desire output."""

  def test_desire_none_when_off(self):
    """Test desire is none when state is off."""
    helper = DesireHelper()

    CS = create_mock_carstate()
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(helper.desire, log.Desire.none)

  def test_desire_left_when_starting_left(self):
    """Test desire is laneChangeLeft when starting left."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeStarting
    helper.lane_change_direction = LaneChangeDirection.left
    helper.lane_change_ll_prob = 0.5

    CS = create_mock_carstate(v_ego=LANE_CHANGE_SPEED_MIN + 1, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(helper.desire, log.Desire.laneChangeLeft)

  def test_desire_right_when_starting_right(self):
    """Test desire is laneChangeRight when starting right."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeStarting
    helper.lane_change_direction = LaneChangeDirection.right
    helper.lane_change_ll_prob = 0.5

    CS = create_mock_carstate(v_ego=LANE_CHANGE_SPEED_MIN + 1, right_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(helper.desire, log.Desire.laneChangeRight)


class TestDesires(unittest.TestCase):
  """Test DESIRES lookup table."""

  def test_desires_has_all_directions(self):
    """Test DESIRES has entries for all directions."""
    self.assertIn(LaneChangeDirection.none, DESIRES)
    self.assertIn(LaneChangeDirection.left, DESIRES)
    self.assertIn(LaneChangeDirection.right, DESIRES)

  def test_desires_has_all_states(self):
    """Test each direction has entries for all states."""
    for direction in [LaneChangeDirection.none, LaneChangeDirection.left, LaneChangeDirection.right]:
      self.assertIn(LaneChangeState.off, DESIRES[direction])
      self.assertIn(LaneChangeState.preLaneChange, DESIRES[direction])
      self.assertIn(LaneChangeState.laneChangeStarting, DESIRES[direction])
      self.assertIn(LaneChangeState.laneChangeFinishing, DESIRES[direction])

  def test_none_direction_all_none(self):
    """Test none direction returns none for all states."""
    for state in DESIRES[LaneChangeDirection.none].values():
      self.assertEqual(state, log.Desire.none)

  def test_left_direction_starting_is_left(self):
    """Test left direction starting state is laneChangeLeft."""
    self.assertEqual(
      DESIRES[LaneChangeDirection.left][LaneChangeState.laneChangeStarting],
      log.Desire.laneChangeLeft
    )

  def test_right_direction_starting_is_right(self):
    """Test right direction starting state is laneChangeRight."""
    self.assertEqual(
      DESIRES[LaneChangeDirection.right][LaneChangeState.laneChangeStarting],
      log.Desire.laneChangeRight
    )


class TestConstants(unittest.TestCase):
  """Test module constants."""

  def test_lane_change_speed_min_positive(self):
    """Test LANE_CHANGE_SPEED_MIN is positive."""
    self.assertGreater(LANE_CHANGE_SPEED_MIN, 0)

  def test_lane_change_time_max_positive(self):
    """Test LANE_CHANGE_TIME_MAX is positive."""
    self.assertGreater(LANE_CHANGE_TIME_MAX, 0)


if __name__ == '__main__':
  unittest.main()
