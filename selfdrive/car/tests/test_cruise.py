"""Tests for selfdrive/car/cruise.py - cruise control helper utilities."""

import pytest

from cereal import car

from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import (
  VCruiseHelper,
  V_CRUISE_MIN,
  V_CRUISE_MAX,
  V_CRUISE_UNSET,
  V_CRUISE_INITIAL,
  V_CRUISE_INITIAL_EXPERIMENTAL_MODE,
  IMPERIAL_INCREMENT,
  CRUISE_LONG_PRESS,
)

ButtonEvent = car.CarState.ButtonEvent
ButtonType = car.CarState.ButtonEvent.Type


def create_mock_cp(mocker, pcm_cruise=False):
  """Create a mock CarParams for testing."""
  CP = mocker.MagicMock()
  CP.pcmCruise = pcm_cruise
  return CP


def create_mock_cs(mocker, v_ego=20.0, cruise_available=True, cruise_speed=0.0, cruise_standstill=False, gas_pressed=False, button_events=None):
  """Create a mock CarState for testing."""
  CS = mocker.MagicMock()
  CS.vEgo = v_ego
  CS.gasPressed = gas_pressed
  CS.cruiseState = mocker.MagicMock()
  CS.cruiseState.available = cruise_available
  CS.cruiseState.speed = cruise_speed
  CS.cruiseState.speedCluster = cruise_speed
  CS.cruiseState.standstill = cruise_standstill
  CS.buttonEvents = button_events or []
  return CS


def create_button_event(mocker, button_type, pressed=True):
  """Create a mock button event."""
  event = mocker.MagicMock()
  # button_type can be enum or raw int
  if hasattr(button_type, 'raw'):
    event.type = button_type
    event.type.raw = button_type.raw
  else:
    event.type = mocker.MagicMock()
    event.type.raw = button_type
  event.pressed = pressed
  return event


class TestConstants:
  """Test module constants."""

  def test_v_cruise_min_positive(self):
    """Test V_CRUISE_MIN is positive."""
    assert V_CRUISE_MIN > 0

  def test_v_cruise_max_greater_than_min(self):
    """Test V_CRUISE_MAX > V_CRUISE_MIN."""
    assert V_CRUISE_MAX > V_CRUISE_MIN

  def test_v_cruise_unset_is_special_value(self):
    """Test V_CRUISE_UNSET is outside normal range."""
    assert V_CRUISE_UNSET > V_CRUISE_MAX

  def test_v_cruise_initial_in_range(self):
    """Test V_CRUISE_INITIAL is in valid range."""
    assert V_CRUISE_INITIAL >= V_CRUISE_MIN
    assert V_CRUISE_INITIAL <= V_CRUISE_MAX

  def test_v_cruise_initial_experimental_in_range(self):
    """Test V_CRUISE_INITIAL_EXPERIMENTAL_MODE is in valid range."""
    assert V_CRUISE_INITIAL_EXPERIMENTAL_MODE >= V_CRUISE_MIN
    assert V_CRUISE_INITIAL_EXPERIMENTAL_MODE <= V_CRUISE_MAX

  def test_imperial_increment_positive(self):
    """Test IMPERIAL_INCREMENT is positive."""
    assert IMPERIAL_INCREMENT > 0

  def test_cruise_long_press_positive(self):
    """Test CRUISE_LONG_PRESS is positive."""
    assert CRUISE_LONG_PRESS > 0


class TestVCruiseHelperInit:
  """Test VCruiseHelper initialization."""

  def test_init(self, mocker):
    """Test VCruiseHelper initializes correctly."""
    CP = create_mock_cp(mocker)
    helper = VCruiseHelper(CP)

    assert helper.v_cruise_kph == V_CRUISE_UNSET
    assert helper.v_cruise_cluster_kph == V_CRUISE_UNSET
    assert helper.v_cruise_kph_last == 0
    assert helper.CP == CP

  def test_v_cruise_initialized_false_initially(self, mocker):
    """Test v_cruise_initialized is False initially."""
    CP = create_mock_cp(mocker)
    helper = VCruiseHelper(CP)
    assert helper.v_cruise_initialized is False

  def test_v_cruise_initialized_true_when_set(self, mocker):
    """Test v_cruise_initialized is True when speed is set."""
    CP = create_mock_cp(mocker)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50
    assert helper.v_cruise_initialized is True


class TestVCruiseHelperUpdatePCM:
  """Test VCruiseHelper update_v_cruise with PCM cruise."""

  def test_pcm_cruise_uses_car_speed(self, mocker):
    """Test PCM cruise uses speed from CarState."""
    CP = create_mock_cp(mocker, pcm_cruise=True)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(mocker, cruise_speed=30.0 * CV.KPH_TO_MS)  # 30 kph
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert helper.v_cruise_kph == pytest.approx(30.0, abs=1)

  def test_pcm_cruise_zero_speed_unsets(self, mocker):
    """Test PCM cruise with zero speed sets to UNSET."""
    CP = create_mock_cp(mocker, pcm_cruise=True)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(mocker, cruise_speed=0.0)
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert helper.v_cruise_kph == V_CRUISE_UNSET

  def test_pcm_cruise_negative_speed(self, mocker):
    """Test PCM cruise with -1 speed sets to -1."""
    CP = create_mock_cp(mocker, pcm_cruise=True)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(mocker, cruise_speed=-1.0)
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert helper.v_cruise_kph == -1

  def test_cruise_not_available_unsets(self, mocker):
    """Test cruise not available sets to UNSET."""
    CP = create_mock_cp(mocker, pcm_cruise=True)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    CS = create_mock_cs(mocker, cruise_available=False)
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert helper.v_cruise_kph == V_CRUISE_UNSET

  def test_updates_kph_last(self, mocker):
    """Test v_cruise_kph_last is updated."""
    CP = create_mock_cp(mocker, pcm_cruise=True)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    CS = create_mock_cs(mocker, cruise_speed=60.0 * CV.KPH_TO_MS)
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert helper.v_cruise_kph_last == 50


class TestVCruiseHelperUpdateNonPCM:
  """Test VCruiseHelper update_v_cruise without PCM cruise."""

  def test_non_pcm_sets_cluster_speed(self, mocker):
    """Test non-PCM cruise sets cluster speed same as cruise speed."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    CS = create_mock_cs(mocker)
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert helper.v_cruise_kph == helper.v_cruise_cluster_kph

  def test_non_pcm_not_enabled_no_change(self, mocker):
    """Test non-PCM cruise doesn't change speed when not enabled."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    CS = create_mock_cs(mocker)
    helper.update_v_cruise(CS, enabled=False, is_metric=True)

    # Speed should remain the same (no button processing)
    assert helper.v_cruise_kph == 50


class TestVCruiseHelperButtonTimers:
  """Test VCruiseHelper button timer handling."""

  def test_button_pressed_starts_timer(self, mocker):
    """Test button press starts timer."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)

    button = create_button_event(mocker, ButtonType.accelCruise, pressed=True)
    CS = create_mock_cs(mocker, button_events=[button])

    helper.update_button_timers(CS, enabled=True)

    assert helper.button_timers[ButtonType.accelCruise] == 1

  def test_button_released_stops_timer(self, mocker):
    """Test button release stops timer."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.button_timers[ButtonType.accelCruise] = 10

    button = create_button_event(mocker, ButtonType.accelCruise, pressed=False)
    CS = create_mock_cs(mocker, button_events=[button])

    helper.update_button_timers(CS, enabled=True)

    assert helper.button_timers[ButtonType.accelCruise] == 0

  def test_timer_increments_while_pressed(self, mocker):
    """Test timer increments while button is pressed."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.button_timers[ButtonType.accelCruise] = 5

    CS = create_mock_cs(mocker)  # No button events, but timer active
    helper.update_button_timers(CS, enabled=True)

    assert helper.button_timers[ButtonType.accelCruise] == 6

  def test_button_state_stored(self, mocker):
    """Test button state is stored on press."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)

    button = create_button_event(mocker, ButtonType.accelCruise, pressed=True)
    CS = create_mock_cs(mocker, cruise_standstill=True, button_events=[button])

    helper.update_button_timers(CS, enabled=True)

    assert helper.button_change_states[ButtonType.accelCruise]["standstill"] is True
    assert helper.button_change_states[ButtonType.accelCruise]["enabled"] is True


class TestVCruiseHelperInitialize:
  """Test VCruiseHelper initialize_v_cruise."""

  def test_pcm_cruise_returns_early(self, mocker):
    """Test PCM cruise doesn't initialize manually."""
    CP = create_mock_cp(mocker, pcm_cruise=True)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = V_CRUISE_UNSET

    CS = create_mock_cs(mocker, v_ego=50.0)
    helper.initialize_v_cruise(CS, experimental_mode=False)

    # Should remain UNSET since PCM handles it
    assert helper.v_cruise_kph == V_CRUISE_UNSET

  def test_normal_mode_initial_speed(self, mocker):
    """Test normal mode uses V_CRUISE_INITIAL."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(mocker, v_ego=0.0)  # Stopped
    helper.initialize_v_cruise(CS, experimental_mode=False)

    assert helper.v_cruise_kph == V_CRUISE_INITIAL

  def test_experimental_mode_initial_speed(self, mocker):
    """Test experimental mode uses higher initial speed."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(mocker, v_ego=0.0)  # Stopped
    helper.initialize_v_cruise(CS, experimental_mode=True)

    assert helper.v_cruise_kph == V_CRUISE_INITIAL_EXPERIMENTAL_MODE

  def test_initialize_from_current_speed(self, mocker):
    """Test initialization clips to current vehicle speed."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(mocker, v_ego=80.0 * CV.KPH_TO_MS)  # 80 kph
    helper.initialize_v_cruise(CS, experimental_mode=False)

    # Should be around 80 kph (above V_CRUISE_INITIAL)
    assert helper.v_cruise_kph > V_CRUISE_INITIAL
    assert helper.v_cruise_kph == pytest.approx(80, abs=1)

  def test_initialize_clipped_to_max(self, mocker):
    """Test initialization clips to V_CRUISE_MAX."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(mocker, v_ego=200.0 * CV.KPH_TO_MS)  # 200 kph (above max)
    helper.initialize_v_cruise(CS, experimental_mode=False)

    assert helper.v_cruise_kph == V_CRUISE_MAX

  def test_resume_uses_last_speed(self, mocker):
    """Test resume button uses last cruise speed."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 60  # Currently set (makes v_cruise_initialized True)
    helper.v_cruise_kph_last = 80  # Previous speed

    # Create a proper button event for resumeCruise
    button = mocker.MagicMock()
    button.type = ButtonType.resumeCruise
    CS = create_mock_cs(mocker, button_events=[button])
    helper.initialize_v_cruise(CS, experimental_mode=False)

    assert helper.v_cruise_kph == 80

  def test_sets_cluster_speed(self, mocker):
    """Test initialize sets cluster speed too."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(mocker, v_ego=50.0 * CV.KPH_TO_MS)
    helper.initialize_v_cruise(CS, experimental_mode=False)

    assert helper.v_cruise_kph == helper.v_cruise_cluster_kph


class TestVCruiseHelperSpeedAdjustment:
  """Test VCruiseHelper speed adjustment from button presses."""

  def test_short_press_accel_increases_speed(self, mocker):
    """Test short press on accel button increases speed."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    # First, simulate button press to start timer and store state
    press_event = create_button_event(mocker, ButtonType.accelCruise, pressed=True)
    CS = create_mock_cs(mocker, button_events=[press_event])
    helper.update_button_timers(CS, enabled=True)

    # Then, simulate button release (short press)
    release_event = create_button_event(mocker, ButtonType.accelCruise, pressed=False)
    CS = create_mock_cs(mocker, button_events=[release_event])
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert helper.v_cruise_kph == 51  # +1 kph for metric

  def test_short_press_decel_decreases_speed(self, mocker):
    """Test short press on decel button decreases speed."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    # First, simulate button press
    press_event = create_button_event(mocker, ButtonType.decelCruise, pressed=True)
    CS = create_mock_cs(mocker, button_events=[press_event])
    helper.update_button_timers(CS, enabled=True)

    # Then, simulate button release (short press)
    release_event = create_button_event(mocker, ButtonType.decelCruise, pressed=False)
    CS = create_mock_cs(mocker, button_events=[release_event])
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert helper.v_cruise_kph == 49  # -1 kph for metric

  def test_long_press_increases_by_5(self, mocker):
    """Test long press increases/decreases by 5."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    # Set up timer at exactly CRUISE_LONG_PRESS
    helper.button_timers[ButtonType.accelCruise] = CRUISE_LONG_PRESS
    helper.button_change_states[ButtonType.accelCruise] = {"standstill": False, "enabled": True}

    CS = create_mock_cs(mocker)
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert helper.v_cruise_kph == 55  # +5 kph for long press

  def test_long_press_release_ends_long_press(self, mocker):
    """Test releasing button after long press doesn't change speed again."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    # Simulate timer > CRUISE_LONG_PRESS (long press already processed)
    helper.button_timers[ButtonType.accelCruise] = CRUISE_LONG_PRESS + 5

    # Release button - should return early without changing speed
    release_event = create_button_event(mocker, ButtonType.accelCruise, pressed=False)
    CS = create_mock_cs(mocker, button_events=[release_event])
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert helper.v_cruise_kph == 50  # No change

  def test_accel_in_standstill_no_change(self, mocker):
    """Test accel button in standstill doesn't change speed."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    # Press button while in standstill
    press_event = create_button_event(mocker, ButtonType.accelCruise, pressed=True)
    CS = create_mock_cs(mocker, cruise_standstill=True, button_events=[press_event])
    helper.update_button_timers(CS, enabled=True)

    # Release button
    release_event = create_button_event(mocker, ButtonType.accelCruise, pressed=False)
    CS = create_mock_cs(mocker, cruise_standstill=True, button_events=[release_event])
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert helper.v_cruise_kph == 50  # No change in standstill

  def test_decel_with_gas_pressed_clips_to_vego(self, mocker):
    """Test decel with gas pressed clips speed to vehicle speed."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    # Press decel button
    press_event = create_button_event(mocker, ButtonType.decelCruise, pressed=True)
    CS = create_mock_cs(mocker, gas_pressed=False, button_events=[press_event])
    helper.update_button_timers(CS, enabled=True)

    # Release button with gas pressed and high vehicle speed
    release_event = create_button_event(mocker, ButtonType.decelCruise, pressed=False)
    CS = create_mock_cs(mocker, v_ego=60.0 * CV.KPH_TO_MS, gas_pressed=True, button_events=[release_event])
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    # Speed should be max(49, 60) = 60 due to gas pressed
    assert helper.v_cruise_kph == 60

  def test_speed_clipped_to_max(self, mocker):
    """Test speed is clipped to V_CRUISE_MAX."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = V_CRUISE_MAX

    # Set up long press at max speed
    helper.button_timers[ButtonType.accelCruise] = CRUISE_LONG_PRESS
    helper.button_change_states[ButtonType.accelCruise] = {"standstill": False, "enabled": True}

    CS = create_mock_cs(mocker)
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert helper.v_cruise_kph == V_CRUISE_MAX  # Clipped to max

  def test_speed_clipped_to_min(self, mocker):
    """Test speed is clipped to V_CRUISE_MIN."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = V_CRUISE_MIN

    # Set up long press at min speed
    helper.button_timers[ButtonType.decelCruise] = CRUISE_LONG_PRESS
    helper.button_change_states[ButtonType.decelCruise] = {"standstill": False, "enabled": True}

    CS = create_mock_cs(mocker)
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert helper.v_cruise_kph == V_CRUISE_MIN  # Clipped to min

  def test_enabled_after_button_press_no_change(self, mocker):
    """Test no speed change if enabled after button was pressed."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    # Press button while not enabled
    press_event = create_button_event(mocker, ButtonType.accelCruise, pressed=True)
    CS = create_mock_cs(mocker, button_events=[press_event])
    helper.update_button_timers(CS, enabled=False)  # Not enabled when pressed

    # Release button while now enabled
    release_event = create_button_event(mocker, ButtonType.accelCruise, pressed=False)
    CS = create_mock_cs(mocker, button_events=[release_event])
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert helper.v_cruise_kph == 50  # No change


class TestVCruiseHelperUnknownButtons:
  """Test VCruiseHelper with unrecognized button types."""

  def test_update_v_cruise_ignores_unknown_button(self, mocker):
    """Test update_v_cruise ignores unrecognized button types."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    # Create button event with unknown type (high number not in button_timers)
    unknown_button = mocker.MagicMock()
    unknown_button.type = mocker.MagicMock()
    unknown_button.type.raw = 999  # Unrecognized type
    unknown_button.pressed = False

    CS = create_mock_cs(mocker, button_events=[unknown_button])
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    # Speed should remain unchanged
    assert helper.v_cruise_kph == 50

  def test_update_button_timers_ignores_unknown_button(self, mocker):
    """Test update_button_timers ignores unrecognized button types."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    initial_timers = dict(helper.button_timers)

    # Create button event with unknown type
    unknown_button = mocker.MagicMock()
    unknown_button.type = mocker.MagicMock()
    unknown_button.type.raw = 999  # Unrecognized type
    unknown_button.pressed = True

    CS = create_mock_cs(mocker, button_events=[unknown_button])
    helper.update_button_timers(CS, enabled=True)

    # Timers should remain unchanged
    assert helper.button_timers == initial_timers


class TestVCruiseHelperIntegration:
  """Integration tests for VCruiseHelper."""

  def test_full_cruise_cycle(self, mocker):
    """Test full cruise enable/disable cycle."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)

    # Start: cruise not available
    CS = create_mock_cs(mocker, cruise_available=False)
    helper.update_v_cruise(CS, enabled=False, is_metric=True)
    assert helper.v_cruise_kph == V_CRUISE_UNSET

    # Cruise becomes available, initialize
    CS = create_mock_cs(mocker, v_ego=50.0 * CV.KPH_TO_MS)
    helper.initialize_v_cruise(CS, experimental_mode=False)
    assert helper.v_cruise_kph > 0
    assert helper.v_cruise_kph != V_CRUISE_UNSET

    # Update with cruise active
    helper.update_v_cruise(CS, enabled=True, is_metric=True)
    assert helper.v_cruise_initialized is True

    # Cruise disabled
    CS = create_mock_cs(mocker, cruise_available=False)
    helper.update_v_cruise(CS, enabled=False, is_metric=True)
    assert helper.v_cruise_kph == V_CRUISE_UNSET
