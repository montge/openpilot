"""Tests for desire_helper.py - lane change state machine."""
import unittest
from unittest.mock import MagicMock

from cereal import log
from openpilot.selfdrive.controls.lib.desire_helper import (
  DesireHelper, LANE_CHANGE_SPEED_MIN, LANE_CHANGE_TIME_MAX, DESIRES,
  LaneChangeState, LaneChangeDirection
)
from openpilot.common.realtime import DT_MDL


def make_carstate(v_ego=25.0, left_blinker=False, right_blinker=False,
                  steering_pressed=False, steering_torque=0.0,
                  left_blindspot=False, right_blindspot=False):
  """Create a mock CarState."""
  cs = MagicMock()
  cs.vEgo = v_ego
  cs.leftBlinker = left_blinker
  cs.rightBlinker = right_blinker
  cs.steeringPressed = steering_pressed
  cs.steeringTorque = steering_torque
  cs.leftBlindspot = left_blindspot
  cs.rightBlindspot = right_blindspot
  return cs


class TestDesireHelperInit(unittest.TestCase):
  """Test DesireHelper initialization."""

  def test_init_defaults(self):
    """Test initial state is correct."""
    dh = DesireHelper()
    self.assertEqual(dh.lane_change_state, LaneChangeState.off)
    self.assertEqual(dh.lane_change_direction, LaneChangeDirection.none)
    self.assertEqual(dh.lane_change_timer, 0.0)
    self.assertEqual(dh.lane_change_ll_prob, 1.0)
    self.assertEqual(dh.keep_pulse_timer, 0.0)
    self.assertFalse(dh.prev_one_blinker)
    self.assertEqual(dh.desire, log.Desire.none)


class TestGetLaneChangeDirection(unittest.TestCase):
  """Test get_lane_change_direction static method."""

  def test_left_blinker_returns_left(self):
    """Test left blinker returns left direction."""
    cs = make_carstate(left_blinker=True, right_blinker=False)
    direction = DesireHelper.get_lane_change_direction(cs)
    self.assertEqual(direction, LaneChangeDirection.left)

  def test_right_blinker_returns_right(self):
    """Test right blinker returns right direction."""
    cs = make_carstate(left_blinker=False, right_blinker=True)
    direction = DesireHelper.get_lane_change_direction(cs)
    self.assertEqual(direction, LaneChangeDirection.right)

  def test_no_blinker_returns_right(self):
    """Test no blinker returns right (default)."""
    cs = make_carstate(left_blinker=False, right_blinker=False)
    direction = DesireHelper.get_lane_change_direction(cs)
    self.assertEqual(direction, LaneChangeDirection.right)


class TestDesireHelperStateOff(unittest.TestCase):
  """Test transitions from LaneChangeState.off."""

  def test_off_to_prelanechange_left_blinker(self):
    """Test off->preLaneChange on left blinker activation."""
    dh = DesireHelper()
    cs = make_carstate(v_ego=25.0, left_blinker=True)

    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(dh.lane_change_state, LaneChangeState.preLaneChange)
    self.assertEqual(dh.lane_change_direction, LaneChangeDirection.left)
    self.assertEqual(dh.lane_change_ll_prob, 1.0)

  def test_off_to_prelanechange_right_blinker(self):
    """Test off->preLaneChange on right blinker activation."""
    dh = DesireHelper()
    cs = make_carstate(v_ego=25.0, right_blinker=True)

    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(dh.lane_change_state, LaneChangeState.preLaneChange)
    self.assertEqual(dh.lane_change_direction, LaneChangeDirection.right)

  def test_off_stays_off_below_speed(self):
    """Test stays off when below lane change speed."""
    dh = DesireHelper()
    cs = make_carstate(v_ego=5.0, left_blinker=True)  # Below LANE_CHANGE_SPEED_MIN

    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(dh.lane_change_state, LaneChangeState.off)

  def test_off_stays_off_both_blinkers(self):
    """Test stays off when both blinkers on (hazards)."""
    dh = DesireHelper()
    cs = make_carstate(v_ego=25.0, left_blinker=True, right_blinker=True)

    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(dh.lane_change_state, LaneChangeState.off)

  def test_off_stays_off_lateral_not_active(self):
    """Test stays off when lateral control not active."""
    dh = DesireHelper()
    cs = make_carstate(v_ego=25.0, left_blinker=True)

    dh.update(cs, lateral_active=False, lane_change_prob=0.5)

    self.assertEqual(dh.lane_change_state, LaneChangeState.off)

  def test_off_stays_off_blinker_already_on(self):
    """Test stays off if blinker was already on."""
    dh = DesireHelper()
    dh.prev_one_blinker = True  # Blinker already on
    cs = make_carstate(v_ego=25.0, left_blinker=True)

    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(dh.lane_change_state, LaneChangeState.off)


class TestDesireHelperStatePreLaneChange(unittest.TestCase):
  """Test transitions from LaneChangeState.preLaneChange."""

  def setUp(self):
    """Set up helper in preLaneChange state."""
    self.dh = DesireHelper()
    # Get into preLaneChange state
    cs = make_carstate(v_ego=25.0, left_blinker=True)
    self.dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    self.assertEqual(self.dh.lane_change_state, LaneChangeState.preLaneChange)

  def test_prelanechange_to_starting_with_torque(self):
    """Test preLaneChange->laneChangeStarting with steering torque."""
    cs = make_carstate(v_ego=25.0, left_blinker=True,
                       steering_pressed=True, steering_torque=10.0)  # Positive for left

    self.dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(self.dh.lane_change_state, LaneChangeState.laneChangeStarting)

  def test_prelanechange_to_starting_right_lane_change(self):
    """Test preLaneChange->laneChangeStarting for right lane change."""
    # Reset and go to right preLaneChange
    dh = DesireHelper()
    cs = make_carstate(v_ego=25.0, right_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    # Apply negative torque for right
    cs = make_carstate(v_ego=25.0, right_blinker=True,
                       steering_pressed=True, steering_torque=-10.0)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(dh.lane_change_state, LaneChangeState.laneChangeStarting)
    self.assertEqual(dh.lane_change_direction, LaneChangeDirection.right)

  def test_prelanechange_blocked_by_blindspot_left(self):
    """Test lane change blocked by left blindspot."""
    cs = make_carstate(v_ego=25.0, left_blinker=True,
                       steering_pressed=True, steering_torque=10.0,
                       left_blindspot=True)

    self.dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(self.dh.lane_change_state, LaneChangeState.preLaneChange)

  def test_prelanechange_blocked_by_blindspot_right(self):
    """Test lane change blocked by right blindspot."""
    dh = DesireHelper()
    cs = make_carstate(v_ego=25.0, right_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    cs = make_carstate(v_ego=25.0, right_blinker=True,
                       steering_pressed=True, steering_torque=-10.0,
                       right_blindspot=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(dh.lane_change_state, LaneChangeState.preLaneChange)

  def test_prelanechange_to_off_blinker_removed(self):
    """Test preLaneChange->off when blinker removed."""
    cs = make_carstate(v_ego=25.0, left_blinker=False)

    self.dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(self.dh.lane_change_state, LaneChangeState.off)
    self.assertEqual(self.dh.lane_change_direction, LaneChangeDirection.none)

  def test_prelanechange_to_off_below_speed(self):
    """Test preLaneChange->off when below speed."""
    cs = make_carstate(v_ego=5.0, left_blinker=True)

    self.dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(self.dh.lane_change_state, LaneChangeState.off)

  def test_prelanechange_updates_direction(self):
    """Test that direction can be updated in preLaneChange."""
    # Start with left blinker
    dh = DesireHelper()
    cs = make_carstate(v_ego=25.0, left_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    self.assertEqual(dh.lane_change_direction, LaneChangeDirection.left)

    # Change to right blinker
    cs = make_carstate(v_ego=25.0, right_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    self.assertEqual(dh.lane_change_direction, LaneChangeDirection.right)


class TestDesireHelperStateLaneChangeStarting(unittest.TestCase):
  """Test transitions from LaneChangeState.laneChangeStarting."""

  def setUp(self):
    """Set up helper in laneChangeStarting state."""
    self.dh = DesireHelper()
    # Get into preLaneChange state
    cs = make_carstate(v_ego=25.0, left_blinker=True)
    self.dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    # Get into laneChangeStarting state
    cs = make_carstate(v_ego=25.0, left_blinker=True,
                       steering_pressed=True, steering_torque=10.0)
    self.dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    self.assertEqual(self.dh.lane_change_state, LaneChangeState.laneChangeStarting)

  def test_starting_ll_prob_decreases(self):
    """Test lane line probability decreases during starting."""
    initial_prob = self.dh.lane_change_ll_prob
    cs = make_carstate(v_ego=25.0, left_blinker=True)

    self.dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertLess(self.dh.lane_change_ll_prob, initial_prob)

  def test_starting_to_finishing_low_probability(self):
    """Test laneChangeStarting->laneChangeFinishing on low prob."""
    # Simulate lane change completing
    self.dh.lane_change_ll_prob = 0.005  # Below 0.01
    cs = make_carstate(v_ego=25.0, left_blinker=True)

    self.dh.update(cs, lateral_active=True, lane_change_prob=0.01)  # Below 0.02

    self.assertEqual(self.dh.lane_change_state, LaneChangeState.laneChangeFinishing)

  def test_starting_stays_starting_high_probability(self):
    """Test stays in laneChangeStarting with high probability."""
    cs = make_carstate(v_ego=25.0, left_blinker=True)

    self.dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(self.dh.lane_change_state, LaneChangeState.laneChangeStarting)

  def test_starting_timer_increments(self):
    """Test lane change timer increments during starting."""
    initial_timer = self.dh.lane_change_timer
    cs = make_carstate(v_ego=25.0, left_blinker=True)

    self.dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertGreater(self.dh.lane_change_timer, initial_timer)


class TestDesireHelperStateLaneChangeFinishing(unittest.TestCase):
  """Test transitions from LaneChangeState.laneChangeFinishing."""

  def _get_to_finishing_state(self):
    """Helper to get to laneChangeFinishing state."""
    dh = DesireHelper()
    # Get into preLaneChange state
    cs = make_carstate(v_ego=25.0, left_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    # Get into laneChangeStarting state
    cs = make_carstate(v_ego=25.0, left_blinker=True,
                       steering_pressed=True, steering_torque=10.0)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    # Get into laneChangeFinishing state
    dh.lane_change_ll_prob = 0.005
    cs = make_carstate(v_ego=25.0, left_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.01)
    return dh

  def test_finishing_ll_prob_increases(self):
    """Test lane line probability increases during finishing."""
    dh = self._get_to_finishing_state()
    self.assertEqual(dh.lane_change_state, LaneChangeState.laneChangeFinishing)
    initial_prob = dh.lane_change_ll_prob
    cs = make_carstate(v_ego=25.0, left_blinker=True)

    dh.update(cs, lateral_active=True, lane_change_prob=0.01)

    self.assertGreater(dh.lane_change_ll_prob, initial_prob)

  def test_finishing_to_off_high_ll_prob_no_blinker(self):
    """Test laneChangeFinishing->off when ll_prob high and no blinker."""
    dh = self._get_to_finishing_state()
    dh.lane_change_ll_prob = 0.995  # Above 0.99
    cs = make_carstate(v_ego=25.0, left_blinker=False)

    dh.update(cs, lateral_active=True, lane_change_prob=0.01)

    self.assertEqual(dh.lane_change_state, LaneChangeState.off)

  def test_finishing_to_prelanechange_high_ll_prob_with_blinker(self):
    """Test laneChangeFinishing->preLaneChange when ll_prob high with blinker."""
    dh = self._get_to_finishing_state()
    dh.lane_change_ll_prob = 0.995  # Above 0.99
    cs = make_carstate(v_ego=25.0, left_blinker=True)

    dh.update(cs, lateral_active=True, lane_change_prob=0.01)

    self.assertEqual(dh.lane_change_state, LaneChangeState.preLaneChange)


class TestDesireHelperTimeout(unittest.TestCase):
  """Test lane change timeout behavior."""

  def test_timeout_resets_to_off(self):
    """Test that exceeding timeout resets state."""
    dh = DesireHelper()
    # Get into laneChangeStarting state
    cs = make_carstate(v_ego=25.0, left_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    cs = make_carstate(v_ego=25.0, left_blinker=True,
                       steering_pressed=True, steering_torque=10.0)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    # Simulate timeout
    dh.lane_change_timer = LANE_CHANGE_TIME_MAX + 1.0
    cs = make_carstate(v_ego=25.0, left_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(dh.lane_change_state, LaneChangeState.off)
    self.assertEqual(dh.lane_change_direction, LaneChangeDirection.none)


class TestDesireHelperDesire(unittest.TestCase):
  """Test desire output based on state."""

  def test_desire_none_when_off(self):
    """Test desire is none when off."""
    dh = DesireHelper()
    cs = make_carstate(v_ego=25.0)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(dh.desire, log.Desire.none)

  def test_desire_lane_change_left_when_starting_left(self):
    """Test desire is laneChangeLeft when starting left lane change."""
    dh = DesireHelper()
    cs = make_carstate(v_ego=25.0, left_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    cs = make_carstate(v_ego=25.0, left_blinker=True,
                       steering_pressed=True, steering_torque=10.0)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(dh.lane_change_state, LaneChangeState.laneChangeStarting)
    self.assertEqual(dh.desire, log.Desire.laneChangeLeft)

  def test_desire_lane_change_right_when_starting_right(self):
    """Test desire is laneChangeRight when starting right lane change."""
    dh = DesireHelper()
    cs = make_carstate(v_ego=25.0, right_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    cs = make_carstate(v_ego=25.0, right_blinker=True,
                       steering_pressed=True, steering_torque=-10.0)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(dh.lane_change_state, LaneChangeState.laneChangeStarting)
    self.assertEqual(dh.desire, log.Desire.laneChangeRight)


class TestDesireHelperKeepPulseTimer(unittest.TestCase):
  """Test keep pulse timer behavior."""

  def test_keep_pulse_timer_resets_on_off(self):
    """Test keep pulse timer resets when off."""
    dh = DesireHelper()
    dh.keep_pulse_timer = 0.5
    cs = make_carstate(v_ego=25.0)

    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(dh.keep_pulse_timer, 0.0)

  def test_keep_pulse_timer_resets_on_starting(self):
    """Test keep pulse timer resets when starting lane change."""
    dh = DesireHelper()
    # Get to laneChangeStarting
    cs = make_carstate(v_ego=25.0, left_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    cs = make_carstate(v_ego=25.0, left_blinker=True,
                       steering_pressed=True, steering_torque=10.0)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertEqual(dh.keep_pulse_timer, 0.0)

  def test_keep_pulse_timer_increments_in_prelanechange(self):
    """Test keep pulse timer increments in preLaneChange."""
    dh = DesireHelper()
    cs = make_carstate(v_ego=25.0, left_blinker=True)
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)
    initial_timer = dh.keep_pulse_timer

    # Update again in preLaneChange
    dh.update(cs, lateral_active=True, lane_change_prob=0.5)

    self.assertGreater(dh.keep_pulse_timer, initial_timer)


class TestDesiresMapping(unittest.TestCase):
  """Test DESIRES dictionary mapping."""

  def test_desires_none_direction_all_states(self):
    """Test all states for none direction return none desire."""
    for state in LaneChangeState.schema.enumerants:
      if state in DESIRES[LaneChangeDirection.none]:
        self.assertEqual(DESIRES[LaneChangeDirection.none][state], log.Desire.none)

  def test_desires_left_starting_returns_left(self):
    """Test left direction starting returns laneChangeLeft."""
    self.assertEqual(
      DESIRES[LaneChangeDirection.left][LaneChangeState.laneChangeStarting],
      log.Desire.laneChangeLeft
    )

  def test_desires_right_starting_returns_right(self):
    """Test right direction starting returns laneChangeRight."""
    self.assertEqual(
      DESIRES[LaneChangeDirection.right][LaneChangeState.laneChangeStarting],
      log.Desire.laneChangeRight
    )


if __name__ == '__main__':
  unittest.main()
