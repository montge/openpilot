"""Tests for selfdrive/car/cruise.py - cruise control helper utilities."""
import unittest
from unittest.mock import MagicMock

import numpy as np
from cereal import car

from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import (
  VCruiseHelper, V_CRUISE_MIN, V_CRUISE_MAX, V_CRUISE_UNSET,
  V_CRUISE_INITIAL, V_CRUISE_INITIAL_EXPERIMENTAL_MODE,
  IMPERIAL_INCREMENT, CRUISE_LONG_PRESS,
)

ButtonEvent = car.CarState.ButtonEvent
ButtonType = car.CarState.ButtonEvent.Type


def create_mock_cp(pcm_cruise=False):
  """Create a mock CarParams for testing."""
  CP = MagicMock()
  CP.pcmCruise = pcm_cruise
  return CP


def create_mock_cs(v_ego=20.0, cruise_available=True, cruise_speed=0.0,
                   cruise_standstill=False, gas_pressed=False, button_events=None):
  """Create a mock CarState for testing."""
  CS = MagicMock()
  CS.vEgo = v_ego
  CS.gasPressed = gas_pressed
  CS.cruiseState = MagicMock()
  CS.cruiseState.available = cruise_available
  CS.cruiseState.speed = cruise_speed
  CS.cruiseState.speedCluster = cruise_speed
  CS.cruiseState.standstill = cruise_standstill
  CS.buttonEvents = button_events or []
  return CS


def create_button_event(button_type, pressed=True):
  """Create a mock button event."""
  event = MagicMock()
  # button_type can be enum or raw int
  if hasattr(button_type, 'raw'):
    event.type = button_type
    event.type.raw = button_type.raw
  else:
    event.type = MagicMock()
    event.type.raw = button_type
  event.pressed = pressed
  return event


class TestConstants(unittest.TestCase):
  """Test module constants."""

  def test_v_cruise_min_positive(self):
    """Test V_CRUISE_MIN is positive."""
    self.assertGreater(V_CRUISE_MIN, 0)

  def test_v_cruise_max_greater_than_min(self):
    """Test V_CRUISE_MAX > V_CRUISE_MIN."""
    self.assertGreater(V_CRUISE_MAX, V_CRUISE_MIN)

  def test_v_cruise_unset_is_special_value(self):
    """Test V_CRUISE_UNSET is outside normal range."""
    self.assertGreater(V_CRUISE_UNSET, V_CRUISE_MAX)

  def test_v_cruise_initial_in_range(self):
    """Test V_CRUISE_INITIAL is in valid range."""
    self.assertGreaterEqual(V_CRUISE_INITIAL, V_CRUISE_MIN)
    self.assertLessEqual(V_CRUISE_INITIAL, V_CRUISE_MAX)

  def test_v_cruise_initial_experimental_in_range(self):
    """Test V_CRUISE_INITIAL_EXPERIMENTAL_MODE is in valid range."""
    self.assertGreaterEqual(V_CRUISE_INITIAL_EXPERIMENTAL_MODE, V_CRUISE_MIN)
    self.assertLessEqual(V_CRUISE_INITIAL_EXPERIMENTAL_MODE, V_CRUISE_MAX)

  def test_imperial_increment_positive(self):
    """Test IMPERIAL_INCREMENT is positive."""
    self.assertGreater(IMPERIAL_INCREMENT, 0)

  def test_cruise_long_press_positive(self):
    """Test CRUISE_LONG_PRESS is positive."""
    self.assertGreater(CRUISE_LONG_PRESS, 0)


class TestVCruiseHelperInit(unittest.TestCase):
  """Test VCruiseHelper initialization."""

  def test_init(self):
    """Test VCruiseHelper initializes correctly."""
    CP = create_mock_cp()
    helper = VCruiseHelper(CP)

    self.assertEqual(helper.v_cruise_kph, V_CRUISE_UNSET)
    self.assertEqual(helper.v_cruise_cluster_kph, V_CRUISE_UNSET)
    self.assertEqual(helper.v_cruise_kph_last, 0)
    self.assertEqual(helper.CP, CP)

  def test_v_cruise_initialized_false_initially(self):
    """Test v_cruise_initialized is False initially."""
    CP = create_mock_cp()
    helper = VCruiseHelper(CP)
    self.assertFalse(helper.v_cruise_initialized)

  def test_v_cruise_initialized_true_when_set(self):
    """Test v_cruise_initialized is True when speed is set."""
    CP = create_mock_cp()
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50
    self.assertTrue(helper.v_cruise_initialized)


class TestVCruiseHelperUpdatePCM(unittest.TestCase):
  """Test VCruiseHelper update_v_cruise with PCM cruise."""

  def test_pcm_cruise_uses_car_speed(self):
    """Test PCM cruise uses speed from CarState."""
    CP = create_mock_cp(pcm_cruise=True)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(cruise_speed=30.0 * CV.KPH_TO_MS)  # 30 kph
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    self.assertAlmostEqual(helper.v_cruise_kph, 30.0, places=0)

  def test_pcm_cruise_zero_speed_unsets(self):
    """Test PCM cruise with zero speed sets to UNSET."""
    CP = create_mock_cp(pcm_cruise=True)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(cruise_speed=0.0)
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    self.assertEqual(helper.v_cruise_kph, V_CRUISE_UNSET)

  def test_pcm_cruise_negative_speed(self):
    """Test PCM cruise with -1 speed sets to -1."""
    CP = create_mock_cp(pcm_cruise=True)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(cruise_speed=-1.0)
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    self.assertEqual(helper.v_cruise_kph, -1)

  def test_cruise_not_available_unsets(self):
    """Test cruise not available sets to UNSET."""
    CP = create_mock_cp(pcm_cruise=True)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    CS = create_mock_cs(cruise_available=False)
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    self.assertEqual(helper.v_cruise_kph, V_CRUISE_UNSET)

  def test_updates_kph_last(self):
    """Test v_cruise_kph_last is updated."""
    CP = create_mock_cp(pcm_cruise=True)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    CS = create_mock_cs(cruise_speed=60.0 * CV.KPH_TO_MS)
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    self.assertEqual(helper.v_cruise_kph_last, 50)


class TestVCruiseHelperUpdateNonPCM(unittest.TestCase):
  """Test VCruiseHelper update_v_cruise without PCM cruise."""

  def test_non_pcm_sets_cluster_speed(self):
    """Test non-PCM cruise sets cluster speed same as cruise speed."""
    CP = create_mock_cp(pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    CS = create_mock_cs()
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    self.assertEqual(helper.v_cruise_kph, helper.v_cruise_cluster_kph)

  def test_non_pcm_not_enabled_no_change(self):
    """Test non-PCM cruise doesn't change speed when not enabled."""
    CP = create_mock_cp(pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    CS = create_mock_cs()
    helper.update_v_cruise(CS, enabled=False, is_metric=True)

    # Speed should remain the same (no button processing)
    self.assertEqual(helper.v_cruise_kph, 50)


class TestVCruiseHelperButtonTimers(unittest.TestCase):
  """Test VCruiseHelper button timer handling."""

  def test_button_pressed_starts_timer(self):
    """Test button press starts timer."""
    CP = create_mock_cp(pcm_cruise=False)
    helper = VCruiseHelper(CP)

    button = create_button_event(ButtonType.accelCruise, pressed=True)
    CS = create_mock_cs(button_events=[button])

    helper.update_button_timers(CS, enabled=True)

    self.assertEqual(helper.button_timers[ButtonType.accelCruise], 1)

  def test_button_released_stops_timer(self):
    """Test button release stops timer."""
    CP = create_mock_cp(pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.button_timers[ButtonType.accelCruise] = 10

    button = create_button_event(ButtonType.accelCruise, pressed=False)
    CS = create_mock_cs(button_events=[button])

    helper.update_button_timers(CS, enabled=True)

    self.assertEqual(helper.button_timers[ButtonType.accelCruise], 0)

  def test_timer_increments_while_pressed(self):
    """Test timer increments while button is pressed."""
    CP = create_mock_cp(pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.button_timers[ButtonType.accelCruise] = 5

    CS = create_mock_cs()  # No button events, but timer active
    helper.update_button_timers(CS, enabled=True)

    self.assertEqual(helper.button_timers[ButtonType.accelCruise], 6)

  def test_button_state_stored(self):
    """Test button state is stored on press."""
    CP = create_mock_cp(pcm_cruise=False)
    helper = VCruiseHelper(CP)

    button = create_button_event(ButtonType.accelCruise, pressed=True)
    CS = create_mock_cs(cruise_standstill=True, button_events=[button])

    helper.update_button_timers(CS, enabled=True)

    self.assertTrue(helper.button_change_states[ButtonType.accelCruise]["standstill"])
    self.assertTrue(helper.button_change_states[ButtonType.accelCruise]["enabled"])


class TestVCruiseHelperInitialize(unittest.TestCase):
  """Test VCruiseHelper initialize_v_cruise."""

  def test_pcm_cruise_returns_early(self):
    """Test PCM cruise doesn't initialize manually."""
    CP = create_mock_cp(pcm_cruise=True)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = V_CRUISE_UNSET

    CS = create_mock_cs(v_ego=50.0)
    helper.initialize_v_cruise(CS, experimental_mode=False)

    # Should remain UNSET since PCM handles it
    self.assertEqual(helper.v_cruise_kph, V_CRUISE_UNSET)

  def test_normal_mode_initial_speed(self):
    """Test normal mode uses V_CRUISE_INITIAL."""
    CP = create_mock_cp(pcm_cruise=False)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(v_ego=0.0)  # Stopped
    helper.initialize_v_cruise(CS, experimental_mode=False)

    self.assertEqual(helper.v_cruise_kph, V_CRUISE_INITIAL)

  def test_experimental_mode_initial_speed(self):
    """Test experimental mode uses higher initial speed."""
    CP = create_mock_cp(pcm_cruise=False)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(v_ego=0.0)  # Stopped
    helper.initialize_v_cruise(CS, experimental_mode=True)

    self.assertEqual(helper.v_cruise_kph, V_CRUISE_INITIAL_EXPERIMENTAL_MODE)

  def test_initialize_from_current_speed(self):
    """Test initialization clips to current vehicle speed."""
    CP = create_mock_cp(pcm_cruise=False)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(v_ego=80.0 * CV.KPH_TO_MS)  # 80 kph
    helper.initialize_v_cruise(CS, experimental_mode=False)

    # Should be around 80 kph (above V_CRUISE_INITIAL)
    self.assertGreater(helper.v_cruise_kph, V_CRUISE_INITIAL)
    self.assertAlmostEqual(helper.v_cruise_kph, 80, delta=1)

  def test_initialize_clipped_to_max(self):
    """Test initialization clips to V_CRUISE_MAX."""
    CP = create_mock_cp(pcm_cruise=False)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(v_ego=200.0 * CV.KPH_TO_MS)  # 200 kph (above max)
    helper.initialize_v_cruise(CS, experimental_mode=False)

    self.assertEqual(helper.v_cruise_kph, V_CRUISE_MAX)

  def test_resume_uses_last_speed(self):
    """Test resume button uses last cruise speed."""
    CP = create_mock_cp(pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 60  # Currently set (makes v_cruise_initialized True)
    helper.v_cruise_kph_last = 80  # Previous speed

    # Create a proper button event for resumeCruise
    button = MagicMock()
    button.type = ButtonType.resumeCruise
    CS = create_mock_cs(button_events=[button])
    helper.initialize_v_cruise(CS, experimental_mode=False)

    self.assertEqual(helper.v_cruise_kph, 80)

  def test_sets_cluster_speed(self):
    """Test initialize sets cluster speed too."""
    CP = create_mock_cp(pcm_cruise=False)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(v_ego=50.0 * CV.KPH_TO_MS)
    helper.initialize_v_cruise(CS, experimental_mode=False)

    self.assertEqual(helper.v_cruise_kph, helper.v_cruise_cluster_kph)


class TestVCruiseHelperIntegration(unittest.TestCase):
  """Integration tests for VCruiseHelper."""

  def test_full_cruise_cycle(self):
    """Test full cruise enable/disable cycle."""
    CP = create_mock_cp(pcm_cruise=False)
    helper = VCruiseHelper(CP)

    # Start: cruise not available
    CS = create_mock_cs(cruise_available=False)
    helper.update_v_cruise(CS, enabled=False, is_metric=True)
    self.assertEqual(helper.v_cruise_kph, V_CRUISE_UNSET)

    # Cruise becomes available, initialize
    CS = create_mock_cs(v_ego=50.0 * CV.KPH_TO_MS)
    helper.initialize_v_cruise(CS, experimental_mode=False)
    self.assertGreater(helper.v_cruise_kph, 0)
    self.assertNotEqual(helper.v_cruise_kph, V_CRUISE_UNSET)

    # Update with cruise active
    helper.update_v_cruise(CS, enabled=True, is_metric=True)
    self.assertTrue(helper.v_cruise_initialized)

    # Cruise disabled
    CS = create_mock_cs(cruise_available=False)
    helper.update_v_cruise(CS, enabled=False, is_metric=True)
    self.assertEqual(helper.v_cruise_kph, V_CRUISE_UNSET)


if __name__ == '__main__':
  unittest.main()
