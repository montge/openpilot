"""Tests for selfdrive/controls/lib/desire_helper.py - lane change desire helper."""

from cereal import log

from openpilot.selfdrive.controls.lib.desire_helper import (
  DesireHelper,
  DESIRES,
  LANE_CHANGE_SPEED_MIN,
  LANE_CHANGE_TIME_MAX,
)

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection


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


class TestDesireHelperInit:
  """Test DesireHelper initialization."""

  def test_init(self):
    """Test DesireHelper initializes correctly."""
    helper = DesireHelper()

    assert helper.lane_change_state == LaneChangeState.off
    assert helper.lane_change_direction == LaneChangeDirection.none
    assert helper.lane_change_timer == 0.0
    assert helper.lane_change_ll_prob == 1.0
    assert helper.keep_pulse_timer == 0.0
    assert not helper.prev_one_blinker
    assert helper.desire == log.Desire.none


class TestGetLaneChangeDirection:
  """Test get_lane_change_direction static method."""

  def test_left_blinker_returns_left(self, mocker):
    """Test left blinker returns left direction."""
    CS = create_mock_carstate(mocker, left_blinker=True, right_blinker=False)
    result = DesireHelper.get_lane_change_direction(CS)
    assert result == LaneChangeDirection.left

  def test_right_blinker_returns_right(self, mocker):
    """Test right blinker returns right direction."""
    CS = create_mock_carstate(mocker, left_blinker=False, right_blinker=True)
    result = DesireHelper.get_lane_change_direction(CS)
    assert result == LaneChangeDirection.right

  def test_no_blinker_returns_right(self, mocker):
    """Test no blinker returns right (default)."""
    CS = create_mock_carstate(mocker, left_blinker=False, right_blinker=False)
    result = DesireHelper.get_lane_change_direction(CS)
    assert result == LaneChangeDirection.right


class TestDesireHelperUpdate:
  """Test DesireHelper update method."""

  def test_not_lateral_active_goes_off(self, mocker):
    """Test not lateral active sets state to off."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.preLaneChange
    helper.lane_change_direction = LaneChangeDirection.left

    CS = create_mock_carstate(mocker)
    helper.update(CS, lateral_active=False, lane_change_prob=0.0)

    assert helper.lane_change_state == LaneChangeState.off
    assert helper.lane_change_direction == LaneChangeDirection.none

  def test_timer_exceeded_goes_off(self, mocker):
    """Test timer exceeded sets state to off."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeStarting
    helper.lane_change_timer = LANE_CHANGE_TIME_MAX + 1

    CS = create_mock_carstate(mocker)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.lane_change_state == LaneChangeState.off

  def test_blinker_starts_pre_lane_change(self, mocker):
    """Test blinker starts preLaneChange from off."""
    helper = DesireHelper()
    helper.prev_one_blinker = False

    CS = create_mock_carstate(mocker, v_ego=LANE_CHANGE_SPEED_MIN + 1, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.lane_change_state == LaneChangeState.preLaneChange
    assert helper.lane_change_direction == LaneChangeDirection.left

  def test_blinker_below_speed_stays_off(self, mocker):
    """Test blinker below min speed stays off."""
    helper = DesireHelper()
    helper.prev_one_blinker = False

    CS = create_mock_carstate(mocker, v_ego=LANE_CHANGE_SPEED_MIN - 1, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.lane_change_state == LaneChangeState.off

  def test_pre_lane_change_no_blinker_goes_off(self, mocker):
    """Test preLaneChange goes off when blinker released."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.preLaneChange
    helper.lane_change_direction = LaneChangeDirection.left
    helper.prev_one_blinker = True

    CS = create_mock_carstate(mocker, v_ego=LANE_CHANGE_SPEED_MIN + 1)  # No blinker
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.lane_change_state == LaneChangeState.off
    assert helper.lane_change_direction == LaneChangeDirection.none

  def test_pre_lane_change_torque_starts_change(self, mocker):
    """Test preLaneChange with torque starts lane change."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.preLaneChange
    helper.lane_change_direction = LaneChangeDirection.left
    helper.prev_one_blinker = True

    CS = create_mock_carstate(
      mocker,
      v_ego=LANE_CHANGE_SPEED_MIN + 1,
      left_blinker=True,
      steering_pressed=True,
      steering_torque=10,  # Positive for left
    )
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.lane_change_state == LaneChangeState.laneChangeStarting

  def test_pre_lane_change_blindspot_blocks_change(self, mocker):
    """Test preLaneChange with blindspot doesn't start change."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.preLaneChange
    helper.lane_change_direction = LaneChangeDirection.left
    helper.prev_one_blinker = True

    CS = create_mock_carstate(
      mocker,
      v_ego=LANE_CHANGE_SPEED_MIN + 1,
      left_blinker=True,
      steering_pressed=True,
      steering_torque=10,
      left_blindspot=True,  # Blocking
    )
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.lane_change_state == LaneChangeState.preLaneChange

  def test_lane_change_starting_fades_ll_prob(self, mocker):
    """Test laneChangeStarting fades lane line probability."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeStarting
    helper.lane_change_ll_prob = 1.0
    helper.lane_change_direction = LaneChangeDirection.left

    CS = create_mock_carstate(mocker, v_ego=LANE_CHANGE_SPEED_MIN + 1, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.lane_change_ll_prob < 1.0

  def test_lane_change_starting_to_finishing(self, mocker):
    """Test laneChangeStarting transitions to finishing."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeStarting
    helper.lane_change_ll_prob = 0.0
    helper.lane_change_direction = LaneChangeDirection.left

    CS = create_mock_carstate(mocker, v_ego=LANE_CHANGE_SPEED_MIN + 1, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.01)  # < 0.02

    assert helper.lane_change_state == LaneChangeState.laneChangeFinishing

  def test_lane_change_finishing_fades_in_ll_prob(self, mocker):
    """Test laneChangeFinishing fades in lane line probability."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeFinishing
    helper.lane_change_ll_prob = 0.5
    helper.lane_change_direction = LaneChangeDirection.left

    CS = create_mock_carstate(mocker, v_ego=LANE_CHANGE_SPEED_MIN + 1, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.lane_change_ll_prob > 0.5

  def test_lane_change_finishing_completes_with_blinker(self, mocker):
    """Test laneChangeFinishing completes to preLaneChange when blinker still on."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeFinishing
    helper.lane_change_ll_prob = 0.995  # Just below 0.99 threshold
    helper.lane_change_direction = LaneChangeDirection.left
    helper.prev_one_blinker = True

    CS = create_mock_carstate(mocker, v_ego=LANE_CHANGE_SPEED_MIN + 1, left_blinker=True)
    # Update enough times to push ll_prob above 0.99
    for _ in range(10):
      helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.lane_change_ll_prob > 0.99
    assert helper.lane_change_state == LaneChangeState.preLaneChange
    # Direction is updated in preLaneChange based on current blinker
    assert helper.lane_change_direction == LaneChangeDirection.left

  def test_lane_change_finishing_completes_without_blinker(self, mocker):
    """Test laneChangeFinishing completes to off when blinker released."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeFinishing
    helper.lane_change_ll_prob = 0.995  # Just below 0.99 threshold
    helper.lane_change_direction = LaneChangeDirection.left
    helper.prev_one_blinker = True

    CS = create_mock_carstate(mocker, v_ego=LANE_CHANGE_SPEED_MIN + 1)  # No blinker
    # Update enough times to push ll_prob above 0.99
    for _ in range(10):
      helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.lane_change_ll_prob > 0.99
    assert helper.lane_change_state == LaneChangeState.off
    assert helper.lane_change_direction == LaneChangeDirection.none

  def test_lane_change_timer_increments_during_change(self, mocker):
    """Test lane change timer increments during lane change."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeStarting
    helper.lane_change_timer = 0.0
    helper.lane_change_direction = LaneChangeDirection.left

    CS = create_mock_carstate(mocker, v_ego=LANE_CHANGE_SPEED_MIN + 1, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.lane_change_timer > 0.0

  def test_lane_change_timer_resets_off(self, mocker):
    """Test lane change timer resets when off."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.off
    helper.lane_change_timer = 5.0

    CS = create_mock_carstate(mocker)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.lane_change_timer == 0.0

  def test_keep_pulse_timer_resets_after_one_second(self, mocker):
    """Test keep pulse timer resets after exceeding 1 second in preLaneChange."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.preLaneChange
    helper.lane_change_direction = LaneChangeDirection.left
    helper.keep_pulse_timer = 1.05  # Just over 1 second
    helper.prev_one_blinker = True

    CS = create_mock_carstate(mocker, v_ego=LANE_CHANGE_SPEED_MIN + 1, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.keep_pulse_timer == 0.0

  def test_keep_pulse_timer_increments_in_pre_lane_change(self, mocker):
    """Test keep pulse timer increments during preLaneChange."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.preLaneChange
    helper.lane_change_direction = LaneChangeDirection.left
    helper.keep_pulse_timer = 0.0
    helper.prev_one_blinker = True

    CS = create_mock_carstate(mocker, v_ego=LANE_CHANGE_SPEED_MIN + 1, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.keep_pulse_timer > 0.0

  def test_keep_desire_resets_to_none_in_pre_lane_change(self, mocker):
    """Test keepLeft/keepRight desire resets to none during pre lane change pulse."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.preLaneChange
    helper.lane_change_direction = LaneChangeDirection.left
    helper.keep_pulse_timer = 0.5  # Below 1.0 threshold
    helper.prev_one_blinker = True
    helper.desire = log.Desire.keepLeft  # Pre-set to keepLeft

    CS = create_mock_carstate(mocker, v_ego=LANE_CHANGE_SPEED_MIN + 1, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    # After the pulse logic, desire should be reset to none (line 119)
    assert helper.desire == log.Desire.none

  def test_keep_desire_right_resets_to_none(self, mocker):
    """Test keepRight desire also resets to none."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.preLaneChange
    helper.lane_change_direction = LaneChangeDirection.right
    helper.keep_pulse_timer = 0.5
    helper.prev_one_blinker = True
    helper.desire = log.Desire.keepRight

    CS = create_mock_carstate(mocker, v_ego=LANE_CHANGE_SPEED_MIN + 1, right_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.desire == log.Desire.none


class TestDesireHelperDesire:
  """Test DesireHelper desire output."""

  def test_desire_none_when_off(self, mocker):
    """Test desire is none when state is off."""
    helper = DesireHelper()

    CS = create_mock_carstate(mocker)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.desire == log.Desire.none

  def test_desire_left_when_starting_left(self, mocker):
    """Test desire is laneChangeLeft when starting left."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeStarting
    helper.lane_change_direction = LaneChangeDirection.left
    helper.lane_change_ll_prob = 0.5

    CS = create_mock_carstate(mocker, v_ego=LANE_CHANGE_SPEED_MIN + 1, left_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.desire == log.Desire.laneChangeLeft

  def test_desire_right_when_starting_right(self, mocker):
    """Test desire is laneChangeRight when starting right."""
    helper = DesireHelper()
    helper.lane_change_state = LaneChangeState.laneChangeStarting
    helper.lane_change_direction = LaneChangeDirection.right
    helper.lane_change_ll_prob = 0.5

    CS = create_mock_carstate(mocker, v_ego=LANE_CHANGE_SPEED_MIN + 1, right_blinker=True)
    helper.update(CS, lateral_active=True, lane_change_prob=0.5)

    assert helper.desire == log.Desire.laneChangeRight


class TestDesires:
  """Test DESIRES lookup table."""

  def test_desires_has_all_directions(self):
    """Test DESIRES has entries for all directions."""
    assert LaneChangeDirection.none in DESIRES
    assert LaneChangeDirection.left in DESIRES
    assert LaneChangeDirection.right in DESIRES

  def test_desires_has_all_states(self):
    """Test each direction has entries for all states."""
    for direction in [LaneChangeDirection.none, LaneChangeDirection.left, LaneChangeDirection.right]:
      assert LaneChangeState.off in DESIRES[direction]
      assert LaneChangeState.preLaneChange in DESIRES[direction]
      assert LaneChangeState.laneChangeStarting in DESIRES[direction]
      assert LaneChangeState.laneChangeFinishing in DESIRES[direction]

  def test_none_direction_all_none(self):
    """Test none direction returns none for all states."""
    for state in DESIRES[LaneChangeDirection.none].values():
      assert state == log.Desire.none

  def test_left_direction_starting_is_left(self):
    """Test left direction starting state is laneChangeLeft."""
    assert DESIRES[LaneChangeDirection.left][LaneChangeState.laneChangeStarting] == log.Desire.laneChangeLeft

  def test_right_direction_starting_is_right(self):
    """Test right direction starting state is laneChangeRight."""
    assert DESIRES[LaneChangeDirection.right][LaneChangeState.laneChangeStarting] == log.Desire.laneChangeRight


class TestConstants:
  """Test module constants."""

  def test_lane_change_speed_min_positive(self):
    """Test LANE_CHANGE_SPEED_MIN is positive."""
    assert LANE_CHANGE_SPEED_MIN > 0

  def test_lane_change_time_max_positive(self):
    """Test LANE_CHANGE_TIME_MAX is positive."""
    assert LANE_CHANGE_TIME_MAX > 0
