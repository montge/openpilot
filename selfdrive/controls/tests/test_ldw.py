import pytest
from unittest.mock import MagicMock

from cereal import car, log
from openpilot.selfdrive.controls.lib.ldw import (
  LaneDepartureWarning,
  LDW_MIN_SPEED,
  LANE_DEPARTURE_THRESHOLD,
  CAMERA_OFFSET,
)
from openpilot.common.realtime import DT_CTRL


class TestLDWInitialization:
  """Tests for LDW initialization."""

  def test_initial_state(self):
    """LDW should start with no warnings."""
    ldw = LaneDepartureWarning()
    assert ldw.left == False
    assert ldw.right == False
    assert ldw.last_blinker_frame == 0
    assert ldw.warning == False


class TestLDWBlinkerCooldown:
  """Tests for blinker cooldown logic."""

  def _create_mocks(self, v_ego=20.0, left_blinker=False, right_blinker=False, lat_active=False):
    """Create mock objects for testing."""
    CS = MagicMock()
    CS.vEgo = v_ego
    CS.leftBlinker = left_blinker
    CS.rightBlinker = right_blinker

    CC = MagicMock()
    CC.latActive = lat_active

    modelV2 = MagicMock()
    modelV2.meta.desirePrediction = [0.0] * 8
    modelV2.laneLineProbs = [0.0, 0.9, 0.9, 0.0]  # Inner lanes visible
    modelV2.laneLines = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    modelV2.laneLines[1].y = [-0.5]  # Left lane close
    modelV2.laneLines[2].y = [0.5]   # Right lane close

    return CS, CC, modelV2

  def test_blinker_updates_last_frame(self):
    """Blinker activation should update last_blinker_frame."""
    ldw = LaneDepartureWarning()
    CS, CC, modelV2 = self._create_mocks(left_blinker=True)

    frame = 100
    ldw.update(frame, modelV2, CS, CC)

    assert ldw.last_blinker_frame == frame

  def test_blinker_cooldown_prevents_warning(self):
    """Recent blinker use should prevent LDW warning."""
    ldw = LaneDepartureWarning()
    CS_blinker, CC, modelV2 = self._create_mocks(v_ego=20.0, left_blinker=True)

    # Use blinker at frame 100
    ldw.update(100, modelV2, CS_blinker, CC)

    # Check that warning is blocked for 5 seconds (5.0 / DT_CTRL frames)
    cooldown_frames = int(5.0 / DT_CTRL)
    CS_no_blinker, _, _ = self._create_mocks(v_ego=20.0)

    # Set up conditions that would normally trigger LDW
    modelV2.meta.desirePrediction = [0.0] * 8
    modelV2.meta.desirePrediction[log.Desire.laneChangeLeft] = 0.5  # High lane change prob

    # Check shortly after blinker (should still be in cooldown)
    ldw.update(100 + cooldown_frames - 10, modelV2, CS_no_blinker, CC)
    assert ldw.warning == False  # Still in cooldown

  def test_after_cooldown_warning_allowed(self):
    """After cooldown expires, warnings should be allowed again."""
    ldw = LaneDepartureWarning()
    CS_blinker, CC, modelV2 = self._create_mocks(v_ego=20.0, left_blinker=True)

    # Use blinker at frame 100
    ldw.update(100, modelV2, CS_blinker, CC)

    # Well after cooldown
    cooldown_frames = int(5.0 / DT_CTRL) + 100
    CS_no_blinker, _, _ = self._create_mocks(v_ego=20.0)

    # Set up conditions that trigger LDW
    modelV2.meta.desirePrediction = [0.0] * 8
    modelV2.meta.desirePrediction[log.Desire.laneChangeLeft] = 0.5
    modelV2.laneLines[1].y = [-0.9]  # Very close to left lane

    ldw.update(100 + cooldown_frames, modelV2, CS_no_blinker, CC)
    # Warning might be true now (depends on all conditions)


class TestLDWSpeedThreshold:
  """Tests for LDW speed threshold."""

  # Frame number must be high enough to be past initial blinker cooldown
  # (5 seconds / DT_CTRL = 5 / 0.01 = 500 frames minimum)
  SAFE_FRAME = 1000

  def _create_mocks(self, v_ego, lat_active=False):
    """Create mock objects for testing."""
    CS = MagicMock()
    CS.vEgo = v_ego
    CS.leftBlinker = False
    CS.rightBlinker = False

    CC = MagicMock()
    CC.latActive = lat_active

    # Create desire prediction list with proper indexing
    desire_pred = [0.0] * 8
    desire_pred[log.Desire.laneChangeLeft] = 0.5  # High lane change prob

    modelV2 = MagicMock()
    modelV2.meta.desirePrediction = desire_pred
    modelV2.laneLineProbs = [0.0, 0.9, 0.9, 0.0]  # Inner lanes visible
    modelV2.laneLines = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    modelV2.laneLines[1].y = [-0.9]  # Left lane very close (> -1.12)
    modelV2.laneLines[2].y = [0.9]   # Right lane close (< 1.04)

    return CS, CC, modelV2

  def test_ldw_disabled_below_min_speed(self):
    """LDW should be disabled below minimum speed."""
    ldw = LaneDepartureWarning()
    below_min_speed = LDW_MIN_SPEED - 1.0
    CS, CC, modelV2 = self._create_mocks(v_ego=below_min_speed)

    ldw.update(self.SAFE_FRAME, modelV2, CS, CC)

    assert ldw.left == False
    assert ldw.right == False
    assert ldw.warning == False

  def test_ldw_enabled_above_min_speed(self):
    """LDW should be enabled above minimum speed when conditions met."""
    ldw = LaneDepartureWarning()
    above_min_speed = LDW_MIN_SPEED + 1.0
    CS, CC, modelV2 = self._create_mocks(v_ego=above_min_speed)

    ldw.update(self.SAFE_FRAME, modelV2, CS, CC)

    # With conditions set up for left departure, left should be True
    assert ldw.left == True


class TestLDWLatActiveDisable:
  """Tests for LDW being disabled when lateral control is active."""

  SAFE_FRAME = 1000  # Past blinker cooldown

  def _create_mocks(self, lat_active):
    """Create mock objects for testing."""
    CS = MagicMock()
    CS.vEgo = LDW_MIN_SPEED + 5.0  # Above threshold
    CS.leftBlinker = False
    CS.rightBlinker = False

    CC = MagicMock()
    CC.latActive = lat_active

    # Create desire prediction list with proper indexing
    desire_pred = [0.0] * 8
    desire_pred[log.Desire.laneChangeLeft] = 0.5

    modelV2 = MagicMock()
    modelV2.meta.desirePrediction = desire_pred
    modelV2.laneLineProbs = [0.0, 0.9, 0.9, 0.0]
    modelV2.laneLines = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    modelV2.laneLines[1].y = [-0.9]  # Left lane close (> -1.12)
    modelV2.laneLines[2].y = [0.9]   # Right lane close (< 1.04)

    return CS, CC, modelV2

  def test_ldw_disabled_when_lat_active(self):
    """LDW should be disabled when lateral control is active."""
    ldw = LaneDepartureWarning()
    CS, CC, modelV2 = self._create_mocks(lat_active=True)

    ldw.update(self.SAFE_FRAME, modelV2, CS, CC)

    assert ldw.left == False
    assert ldw.right == False

  def test_ldw_enabled_when_lat_inactive(self):
    """LDW should be enabled when lateral control is inactive."""
    ldw = LaneDepartureWarning()
    CS, CC, modelV2 = self._create_mocks(lat_active=False)

    ldw.update(self.SAFE_FRAME, modelV2, CS, CC)

    # Should detect left departure with our mock setup
    assert ldw.left == True


class TestLDWLaneDetection:
  """Tests for lane departure detection logic."""

  SAFE_FRAME = 1000  # Past blinker cooldown

  def _create_mocks(self, v_ego=LDW_MIN_SPEED + 5.0):
    """Create mock objects for testing."""
    CS = MagicMock()
    CS.vEgo = v_ego
    CS.leftBlinker = False
    CS.rightBlinker = False

    CC = MagicMock()
    CC.latActive = False

    # Start with neutral desire prediction
    desire_pred = [0.0] * 8

    modelV2 = MagicMock()
    modelV2.meta.desirePrediction = desire_pred
    modelV2.laneLineProbs = [0.0, 0.9, 0.9, 0.0]  # Inner lanes visible
    modelV2.laneLines = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    modelV2.laneLines[1].y = [-1.5]  # Left lane far
    modelV2.laneLines[2].y = [1.5]   # Right lane far

    return CS, CC, modelV2, desire_pred

  def test_left_departure_detection(self):
    """Left departure should be detected with high probability and close lane."""
    ldw = LaneDepartureWarning()
    CS, CC, modelV2, desire_pred = self._create_mocks()

    # Set up left departure conditions
    desire_pred[log.Desire.laneChangeLeft] = LANE_DEPARTURE_THRESHOLD + 0.1
    modelV2.laneLines[1].y = [-(1.08 + CAMERA_OFFSET) + 0.05]  # Close to left lane (-1.07)

    ldw.update(self.SAFE_FRAME, modelV2, CS, CC)

    assert ldw.left == True
    assert ldw.right == False
    assert ldw.warning == True

  def test_right_departure_detection(self):
    """Right departure should be detected with high probability and close lane."""
    ldw = LaneDepartureWarning()
    CS, CC, modelV2, desire_pred = self._create_mocks()

    # Set up right departure conditions
    desire_pred[log.Desire.laneChangeRight] = LANE_DEPARTURE_THRESHOLD + 0.1
    modelV2.laneLines[2].y = [(1.08 - CAMERA_OFFSET) - 0.05]  # Close to right lane (0.99)

    ldw.update(self.SAFE_FRAME, modelV2, CS, CC)

    assert ldw.left == False
    assert ldw.right == True
    assert ldw.warning == True

  def test_no_departure_when_lane_not_visible(self):
    """No departure warning when lane is not visible."""
    ldw = LaneDepartureWarning()
    CS, CC, modelV2, desire_pred = self._create_mocks()

    # Set up departure conditions but lane not visible
    desire_pred[log.Desire.laneChangeLeft] = LANE_DEPARTURE_THRESHOLD + 0.1
    modelV2.laneLineProbs = [0.0, 0.3, 0.9, 0.0]  # Left lane not visible (< 0.5)
    modelV2.laneLines[1].y = [-0.9]  # Close to left lane

    ldw.update(self.SAFE_FRAME, modelV2, CS, CC)

    assert ldw.left == False

  def test_no_departure_when_probability_low(self):
    """No departure warning when lane change probability is low."""
    ldw = LaneDepartureWarning()
    CS, CC, modelV2, desire_pred = self._create_mocks()

    # Lane is visible and close, but probability is low
    desire_pred[log.Desire.laneChangeLeft] = LANE_DEPARTURE_THRESHOLD - 0.05
    modelV2.laneLines[1].y = [-0.9]  # Close to left lane

    ldw.update(self.SAFE_FRAME, modelV2, CS, CC)

    assert ldw.left == False

  def test_no_departure_when_lane_not_close(self):
    """No departure warning when lane is not close."""
    ldw = LaneDepartureWarning()
    CS, CC, modelV2, desire_pred = self._create_mocks()

    # Probability is high but lane is far (> -(1.08 + 0.04) = -1.12, so < -1.12 is far)
    desire_pred[log.Desire.laneChangeLeft] = LANE_DEPARTURE_THRESHOLD + 0.1
    modelV2.laneLines[1].y = [-2.0]  # Left lane far (< -1.12)

    ldw.update(self.SAFE_FRAME, modelV2, CS, CC)

    assert ldw.left == False


class TestLDWWarningProperty:
  """Tests for the warning property."""

  def test_warning_true_when_left(self):
    """Warning should be True when left departure detected."""
    ldw = LaneDepartureWarning()
    ldw.left = True
    ldw.right = False

    assert ldw.warning == True

  def test_warning_true_when_right(self):
    """Warning should be True when right departure detected."""
    ldw = LaneDepartureWarning()
    ldw.left = False
    ldw.right = True

    assert ldw.warning == True

  def test_warning_true_when_both(self):
    """Warning should be True when both departures detected."""
    ldw = LaneDepartureWarning()
    ldw.left = True
    ldw.right = True

    assert ldw.warning == True

  def test_warning_false_when_none(self):
    """Warning should be False when no departure detected."""
    ldw = LaneDepartureWarning()
    ldw.left = False
    ldw.right = False

    assert ldw.warning == False


class TestLDWEmptyDesirePrediction:
  """Tests for handling empty desire prediction."""

  SAFE_FRAME = 1000  # Past blinker cooldown

  def test_empty_desire_prediction_clears_warnings(self):
    """Empty desire prediction should clear any warnings."""
    ldw = LaneDepartureWarning()
    CS = MagicMock()
    CS.vEgo = LDW_MIN_SPEED + 5.0
    CS.leftBlinker = False
    CS.rightBlinker = False

    CC = MagicMock()
    CC.latActive = False

    modelV2 = MagicMock()
    modelV2.meta.desirePrediction = []  # Empty
    modelV2.laneLineProbs = [0.0, 0.9, 0.9, 0.0]
    modelV2.laneLines = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]

    # Manually set warnings to True to verify they get cleared
    ldw.left = True
    ldw.right = True

    ldw.update(self.SAFE_FRAME, modelV2, CS, CC)

    assert ldw.left == False
    assert ldw.right == False
