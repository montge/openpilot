"""Tests for selfdrive/controls/lib/ldw.py - Lane Departure Warning."""

from cereal import log

from openpilot.common.constants import CV
from openpilot.selfdrive.controls.lib.ldw import (
  LaneDepartureWarning,
  LDW_MIN_SPEED,
  LANE_DEPARTURE_THRESHOLD,
  CAMERA_OFFSET,
)


def create_mock_cs(mocker, v_ego=20.0, left_blinker=False, right_blinker=False):
  """Create a mock CarState."""
  CS = mocker.MagicMock()
  CS.vEgo = v_ego
  CS.leftBlinker = left_blinker
  CS.rightBlinker = right_blinker
  return CS


def create_mock_cc(mocker, lat_active=False):
  """Create a mock CarControl."""
  CC = mocker.MagicMock()
  CC.latActive = lat_active
  return CC


def create_mock_model(mocker, desire_prediction=None, lane_line_probs=None, lane_lines=None):
  """Create a mock modelV2."""
  model = mocker.MagicMock()

  if desire_prediction is None:
    desire_prediction = [0.0] * 8
  model.meta.desirePrediction = desire_prediction

  if lane_line_probs is None:
    lane_line_probs = [0.0, 0.8, 0.8, 0.0]  # Left and right visible
  model.laneLineProbs = lane_line_probs

  if lane_lines is None:
    # Create mock lane lines with y values
    lane_lines = []
    for _ in range(4):
      line = mocker.MagicMock()
      line.y = [0.0] * 33  # Default y positions
      lane_lines.append(line)
    # Set typical lane positions
    lane_lines[1].y[0] = -1.5  # Left lane
    lane_lines[2].y[0] = 1.5  # Right lane
  model.laneLines = lane_lines

  return model


class TestLaneDepartureWarningInit:
  """Test LaneDepartureWarning initialization."""

  def test_init(self):
    """Test LaneDepartureWarning initializes correctly."""
    ldw = LaneDepartureWarning()

    assert not ldw.left
    assert not ldw.right
    assert ldw.last_blinker_frame == 0


class TestLaneDepartureWarningUpdate:
  """Test LaneDepartureWarning update method."""

  def test_blinker_updates_last_frame(self, mocker):
    """Test blinker updates last_blinker_frame."""
    ldw = LaneDepartureWarning()

    CS = create_mock_cs(mocker, left_blinker=True)
    CC = create_mock_cc(mocker)
    model = create_mock_model(mocker)

    ldw.update(frame=100, modelV2=model, CS=CS, CC=CC)

    assert ldw.last_blinker_frame == 100

  def test_right_blinker_updates_last_frame(self, mocker):
    """Test right blinker updates last_blinker_frame."""
    ldw = LaneDepartureWarning()

    CS = create_mock_cs(mocker, right_blinker=True)
    CC = create_mock_cc(mocker)
    model = create_mock_model(mocker)

    ldw.update(frame=50, modelV2=model, CS=CS, CC=CC)

    assert ldw.last_blinker_frame == 50

  def test_ldw_disabled_below_min_speed(self, mocker):
    """Test LDW is disabled below minimum speed."""
    ldw = LaneDepartureWarning()

    CS = create_mock_cs(mocker, v_ego=LDW_MIN_SPEED - 1)
    CC = create_mock_cc(mocker)

    # Set up model to trigger LDW if allowed
    desire_pred = [0.0] * 8
    desire_pred[log.Desire.laneChangeLeft] = 0.5
    model = create_mock_model(mocker, desire_prediction=desire_pred)
    model.laneLines[1].y[0] = -0.5  # Close to left lane

    ldw.update(frame=100, modelV2=model, CS=CS, CC=CC)

    assert not ldw.left
    assert not ldw.right

  def test_ldw_disabled_with_recent_blinker(self, mocker):
    """Test LDW is disabled with recent blinker use."""
    ldw = LaneDepartureWarning()
    ldw.last_blinker_frame = 50  # Recent blinker

    CS = create_mock_cs(mocker, v_ego=LDW_MIN_SPEED + 5)
    CC = create_mock_cc(mocker)

    desire_pred = [0.0] * 8
    desire_pred[log.Desire.laneChangeLeft] = 0.5
    model = create_mock_model(mocker, desire_prediction=desire_pred)
    model.laneLines[1].y[0] = -0.5

    # Frame 100, blinker at 50, so within 5s cooldown
    ldw.update(frame=100, modelV2=model, CS=CS, CC=CC)

    assert not ldw.left
    assert not ldw.right

  def test_ldw_disabled_when_lat_active(self, mocker):
    """Test LDW is disabled when lateral control is active."""
    ldw = LaneDepartureWarning()

    CS = create_mock_cs(mocker, v_ego=LDW_MIN_SPEED + 5)
    CC = create_mock_cc(mocker, lat_active=True)

    desire_pred = [0.0] * 8
    desire_pred[log.Desire.laneChangeLeft] = 0.5
    model = create_mock_model(mocker, desire_prediction=desire_pred)
    model.laneLines[1].y[0] = -0.5

    ldw.update(frame=100, modelV2=model, CS=CS, CC=CC)

    assert not ldw.left
    assert not ldw.right

  def test_left_lane_departure_detected(self, mocker):
    """Test left lane departure is detected."""
    ldw = LaneDepartureWarning()

    CS = create_mock_cs(mocker, v_ego=LDW_MIN_SPEED + 5)
    CC = create_mock_cc(mocker)

    desire_pred = [0.0] * 8
    desire_pred[log.Desire.laneChangeLeft] = LANE_DEPARTURE_THRESHOLD + 0.1
    model = create_mock_model(mocker, desire_prediction=desire_pred, lane_line_probs=[0.0, 0.9, 0.9, 0.0])
    # Left lane close (within threshold)
    model.laneLines[1].y[0] = -(1.08 + CAMERA_OFFSET) + 0.1

    ldw.update(frame=1000, modelV2=model, CS=CS, CC=CC)

    assert ldw.left
    assert not ldw.right

  def test_right_lane_departure_detected(self, mocker):
    """Test right lane departure is detected."""
    ldw = LaneDepartureWarning()

    CS = create_mock_cs(mocker, v_ego=LDW_MIN_SPEED + 5)
    CC = create_mock_cc(mocker)

    desire_pred = [0.0] * 8
    desire_pred[log.Desire.laneChangeRight] = LANE_DEPARTURE_THRESHOLD + 0.1
    model = create_mock_model(mocker, desire_prediction=desire_pred, lane_line_probs=[0.0, 0.9, 0.9, 0.0])
    # Right lane close (within threshold)
    model.laneLines[2].y[0] = (1.08 - CAMERA_OFFSET) - 0.1

    ldw.update(frame=1000, modelV2=model, CS=CS, CC=CC)

    assert not ldw.left
    assert ldw.right

  def test_no_departure_when_lanes_not_visible(self, mocker):
    """Test no departure when lanes are not visible."""
    ldw = LaneDepartureWarning()

    CS = create_mock_cs(mocker, v_ego=LDW_MIN_SPEED + 5)
    CC = create_mock_cc(mocker)

    desire_pred = [0.0] * 8
    desire_pred[log.Desire.laneChangeLeft] = 0.5
    model = create_mock_model(
      mocker,
      desire_prediction=desire_pred,
      lane_line_probs=[0.0, 0.3, 0.3, 0.0],  # Low visibility
    )
    model.laneLines[1].y[0] = -0.5

    ldw.update(frame=1000, modelV2=model, CS=CS, CC=CC)

    assert not ldw.left

  def test_no_departure_when_lane_not_close(self, mocker):
    """Test no departure when lane is not close."""
    ldw = LaneDepartureWarning()

    CS = create_mock_cs(mocker, v_ego=LDW_MIN_SPEED + 5)
    CC = create_mock_cc(mocker)

    desire_pred = [0.0] * 8
    desire_pred[log.Desire.laneChangeLeft] = 0.5
    model = create_mock_model(mocker, desire_prediction=desire_pred, lane_line_probs=[0.0, 0.9, 0.9, 0.0])
    # Left lane far away
    model.laneLines[1].y[0] = -2.0

    ldw.update(frame=1000, modelV2=model, CS=CS, CC=CC)

    assert not ldw.left

  def test_no_departure_below_threshold(self, mocker):
    """Test no departure when probability below threshold."""
    ldw = LaneDepartureWarning()

    CS = create_mock_cs(mocker, v_ego=LDW_MIN_SPEED + 5)
    CC = create_mock_cc(mocker)

    desire_pred = [0.0] * 8
    desire_pred[log.Desire.laneChangeLeft] = LANE_DEPARTURE_THRESHOLD - 0.05
    model = create_mock_model(mocker, desire_prediction=desire_pred, lane_line_probs=[0.0, 0.9, 0.9, 0.0])
    model.laneLines[1].y[0] = -0.5

    ldw.update(frame=1000, modelV2=model, CS=CS, CC=CC)

    assert not ldw.left

  def test_empty_desire_prediction(self, mocker):
    """Test handling of empty desire prediction."""
    ldw = LaneDepartureWarning()

    CS = create_mock_cs(mocker, v_ego=LDW_MIN_SPEED + 5)
    CC = create_mock_cc(mocker)
    model = create_mock_model(mocker, desire_prediction=[])

    ldw.update(frame=1000, modelV2=model, CS=CS, CC=CC)

    assert not ldw.left
    assert not ldw.right


class TestLaneDepartureWarningProperty:
  """Test LaneDepartureWarning warning property."""

  def test_warning_false_when_no_departure(self):
    """Test warning is False when no departure."""
    ldw = LaneDepartureWarning()

    assert not ldw.warning

  def test_warning_true_when_left_departure(self):
    """Test warning is True when left departure."""
    ldw = LaneDepartureWarning()
    ldw.left = True

    assert ldw.warning

  def test_warning_true_when_right_departure(self):
    """Test warning is True when right departure."""
    ldw = LaneDepartureWarning()
    ldw.right = True

    assert ldw.warning

  def test_warning_true_when_both_departure(self):
    """Test warning is True when both departures."""
    ldw = LaneDepartureWarning()
    ldw.left = True
    ldw.right = True

    assert ldw.warning


class TestConstants:
  """Test module constants."""

  def test_ldw_min_speed_positive(self):
    """Test LDW_MIN_SPEED is positive."""
    assert LDW_MIN_SPEED > 0

  def test_ldw_min_speed_reasonable(self):
    """Test LDW_MIN_SPEED is reasonable (~31 mph)."""
    expected_ms = 31 * CV.MPH_TO_MS
    assert abs(LDW_MIN_SPEED - expected_ms) < 0.5

  def test_lane_departure_threshold_in_range(self):
    """Test LANE_DEPARTURE_THRESHOLD is in valid range."""
    assert LANE_DEPARTURE_THRESHOLD > 0
    assert LANE_DEPARTURE_THRESHOLD < 1

  def test_camera_offset_defined(self):
    """Test CAMERA_OFFSET is defined."""
    assert isinstance(CAMERA_OFFSET, float)
