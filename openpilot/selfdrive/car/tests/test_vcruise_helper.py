import numpy as np
import pytest
from parameterized import parameterized_class

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
  CRUISE_NEAREST_FUNC,
  CRUISE_INTERVAL_SIGN,
)

ButtonEvent = car.CarState.ButtonEvent
ButtonType = car.CarState.ButtonEvent.Type


class TestVCruiseConstants:
  """Test cruise control constants"""

  def test_cruise_speed_limits(self):
    assert V_CRUISE_MIN == 8
    assert V_CRUISE_MAX == 145
    assert V_CRUISE_UNSET == 255

  def test_initial_speeds(self):
    assert V_CRUISE_INITIAL == 40
    assert V_CRUISE_INITIAL_EXPERIMENTAL_MODE == 105
    assert V_CRUISE_INITIAL_EXPERIMENTAL_MODE > V_CRUISE_INITIAL

  def test_imperial_increment(self):
    # IMPERIAL_INCREMENT is rounded to 1 decimal place (1.6) to avoid rounding errors
    assert IMPERIAL_INCREMENT == round(CV.MPH_TO_KPH, 1)
    assert IMPERIAL_INCREMENT == 1.6

  def test_long_press_threshold(self):
    assert CRUISE_LONG_PRESS == 50

  def test_cruise_nearest_func_mapping(self):
    # Accel should round up, decel should round down
    assert CRUISE_NEAREST_FUNC[ButtonType.accelCruise](1.1) == 2
    assert CRUISE_NEAREST_FUNC[ButtonType.accelCruise](1.9) == 2
    assert CRUISE_NEAREST_FUNC[ButtonType.decelCruise](1.1) == 1
    assert CRUISE_NEAREST_FUNC[ButtonType.decelCruise](1.9) == 1

  def test_cruise_interval_sign_mapping(self):
    assert CRUISE_INTERVAL_SIGN[ButtonType.accelCruise] == +1
    assert CRUISE_INTERVAL_SIGN[ButtonType.decelCruise] == -1


class TestVCruiseHelperInit:
  """Test VCruiseHelper initialization"""

  def test_init_with_non_pcm_cruise(self):
    CP = car.CarParams.new_message(pcmCruise=False)
    helper = VCruiseHelper(CP)

    assert helper.CP == CP
    assert helper.v_cruise_kph == V_CRUISE_UNSET
    assert helper.v_cruise_cluster_kph == V_CRUISE_UNSET
    assert helper.v_cruise_kph_last == 0

  def test_init_with_pcm_cruise(self):
    CP = car.CarParams.new_message(pcmCruise=True)
    helper = VCruiseHelper(CP)

    assert helper.CP == CP
    assert helper.v_cruise_kph == V_CRUISE_UNSET

  def test_init_button_timers(self):
    CP = car.CarParams.new_message()
    helper = VCruiseHelper(CP)

    assert ButtonType.decelCruise in helper.button_timers
    assert ButtonType.accelCruise in helper.button_timers
    assert helper.button_timers[ButtonType.decelCruise] == 0
    assert helper.button_timers[ButtonType.accelCruise] == 0

  def test_init_button_change_states(self):
    CP = car.CarParams.new_message()
    helper = VCruiseHelper(CP)

    for btn in helper.button_timers:
      assert btn in helper.button_change_states
      assert helper.button_change_states[btn]["standstill"] is False
      assert helper.button_change_states[btn]["enabled"] is False


class TestVCruiseInitialized:
  """Test v_cruise_initialized property"""

  def test_not_initialized_when_unset(self):
    CP = car.CarParams.new_message()
    helper = VCruiseHelper(CP)

    assert helper.v_cruise_initialized is False

  def test_initialized_when_set(self):
    CP = car.CarParams.new_message()
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    assert helper.v_cruise_initialized is True

  def test_initialized_at_min_speed(self):
    CP = car.CarParams.new_message()
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = V_CRUISE_MIN

    assert helper.v_cruise_initialized is True


class TestUpdateVCruisePCM:
  """Test update_v_cruise with PCM cruise control"""

  def setup_method(self):
    self.CP = car.CarParams.new_message(pcmCruise=True)
    self.helper = VCruiseHelper(self.CP)

  def test_cruise_not_available(self):
    CS = car.CarState.new_message(cruiseState={"available": False})
    self.helper.update_v_cruise(CS, enabled=False, is_metric=True)

    assert self.helper.v_cruise_kph == V_CRUISE_UNSET
    assert self.helper.v_cruise_cluster_kph == V_CRUISE_UNSET

  def test_cruise_available_with_speed(self):
    # Speed in m/s, should be converted to kph
    speed_ms = 30.0  # 30 m/s = 108 kph
    CS = car.CarState.new_message(cruiseState={"available": True, "speed": speed_ms, "speedCluster": speed_ms})
    self.helper.update_v_cruise(CS, enabled=True, is_metric=True)

    expected_kph = speed_ms * CV.MS_TO_KPH
    assert self.helper.v_cruise_kph == pytest.approx(expected_kph, rel=1e-3)
    assert self.helper.v_cruise_cluster_kph == pytest.approx(expected_kph, rel=1e-3)

  def test_cruise_speed_zero_sets_unset(self):
    CS = car.CarState.new_message(cruiseState={"available": True, "speed": 0, "speedCluster": 0})
    self.helper.v_cruise_kph = 50  # Previously set
    self.helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert self.helper.v_cruise_kph == V_CRUISE_UNSET
    assert self.helper.v_cruise_cluster_kph == V_CRUISE_UNSET

  def test_cruise_speed_negative_one(self):
    CS = car.CarState.new_message(cruiseState={"available": True, "speed": -1, "speedCluster": -1})
    self.helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert self.helper.v_cruise_kph == -1
    assert self.helper.v_cruise_cluster_kph == -1

  def test_v_cruise_kph_last_updated(self):
    CS = car.CarState.new_message(cruiseState={"available": True, "speed": 20, "speedCluster": 20})
    self.helper.v_cruise_kph = 50
    self.helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert self.helper.v_cruise_kph_last == 50


@parameterized_class(('pcm_cruise',), [(False,)])
class TestUpdateVCruiseNonPCM:
  """Test update_v_cruise with non-PCM cruise (button-based)"""

  def setup_method(self):
    self.CP = car.CarParams.new_message(pcmCruise=self.pcm_cruise)
    self.helper = VCruiseHelper(self.CP)
    self.reset_cruise_speed_state()

  def reset_cruise_speed_state(self):
    # Two resets to clear previous cruise speed
    for _ in range(2):
      self.helper.update_v_cruise(car.CarState.new_message(cruiseState={"available": False}), enabled=False, is_metric=False)

  def enable(self, v_ego, experimental_mode):
    self.helper.initialize_v_cruise(car.CarState.new_message(vEgo=v_ego), experimental_mode)

  def test_disabled_does_not_update(self):
    """When disabled, button presses should not change speed"""
    self.enable(V_CRUISE_INITIAL * CV.KPH_TO_MS, False)
    initial_speed = self.helper.v_cruise_kph

    CS = car.CarState.new_message(cruiseState={"available": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.accelCruise, pressed=False)]
    self.helper.update_v_cruise(CS, enabled=False, is_metric=True)

    assert self.helper.v_cruise_kph == initial_speed

  def test_metric_increment(self):
    """In metric mode, increment should be 1 kph"""
    self.enable(V_CRUISE_INITIAL * CV.KPH_TO_MS, False)

    # Press accel button
    CS = car.CarState.new_message(cruiseState={"available": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.accelCruise, pressed=True)]
    self.helper.update_v_cruise(CS, enabled=True, is_metric=True)

    # Release accel button - speed should change
    initial_speed = self.helper.v_cruise_kph
    CS = car.CarState.new_message(cruiseState={"available": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.accelCruise, pressed=False)]
    self.helper.update_v_cruise(CS, enabled=True, is_metric=True)

    # Speed should increase by 1 kph in metric mode
    assert self.helper.v_cruise_kph == initial_speed + 1

  def test_imperial_increment(self):
    """In imperial mode, increment should be about 1.6 kph (1 mph)"""
    self.enable(V_CRUISE_INITIAL * CV.KPH_TO_MS, False)

    # Press accel button
    CS = car.CarState.new_message(cruiseState={"available": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.accelCruise, pressed=True)]
    self.helper.update_v_cruise(CS, enabled=True, is_metric=False)

    # Release accel button - speed should change
    initial_speed = self.helper.v_cruise_kph
    CS = car.CarState.new_message(cruiseState={"available": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.accelCruise, pressed=False)]
    self.helper.update_v_cruise(CS, enabled=True, is_metric=False)

    # Speed should increase by IMPERIAL_INCREMENT in imperial mode
    assert self.helper.v_cruise_kph == pytest.approx(initial_speed + IMPERIAL_INCREMENT, rel=1e-2)

  def test_decel_decreases_speed(self):
    """Decel button should decrease speed"""
    self.enable(V_CRUISE_INITIAL * CV.KPH_TO_MS, False)

    # Press decel button
    CS = car.CarState.new_message(cruiseState={"available": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.decelCruise, pressed=True)]
    self.helper.update_v_cruise(CS, enabled=True, is_metric=True)

    initial_speed = self.helper.v_cruise_kph

    # Release decel button
    CS = car.CarState.new_message(cruiseState={"available": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.decelCruise, pressed=False)]
    self.helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert self.helper.v_cruise_kph == initial_speed - 1

  def test_speed_clipped_to_min(self):
    """Speed should not go below V_CRUISE_MIN"""
    self.enable(V_CRUISE_MIN * CV.KPH_TO_MS, False)
    self.helper.v_cruise_kph = V_CRUISE_MIN

    # Press decel button
    CS = car.CarState.new_message(cruiseState={"available": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.decelCruise, pressed=True)]
    self.helper.update_v_cruise(CS, enabled=True, is_metric=True)

    # Release decel button
    CS = car.CarState.new_message(cruiseState={"available": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.decelCruise, pressed=False)]
    self.helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert self.helper.v_cruise_kph >= V_CRUISE_MIN

  def test_speed_clipped_to_max(self):
    """Speed should not exceed V_CRUISE_MAX"""
    self.enable(V_CRUISE_MAX * CV.KPH_TO_MS, False)
    self.helper.v_cruise_kph = V_CRUISE_MAX

    # Press accel button
    CS = car.CarState.new_message(cruiseState={"available": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.accelCruise, pressed=True)]
    self.helper.update_v_cruise(CS, enabled=True, is_metric=True)

    # Release accel button
    CS = car.CarState.new_message(cruiseState={"available": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.accelCruise, pressed=False)]
    self.helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert self.helper.v_cruise_kph <= V_CRUISE_MAX


class TestLongPress:
  """Test long press behavior for cruise buttons"""

  def setup_method(self):
    self.CP = car.CarParams.new_message(pcmCruise=False)
    self.helper = VCruiseHelper(self.CP)

  def test_long_press_increment_5x(self):
    """Long press should increment by 5x the normal amount"""
    self.helper.v_cruise_kph = 50
    self.helper.button_timers[ButtonType.accelCruise] = CRUISE_LONG_PRESS
    self.helper.button_change_states[ButtonType.accelCruise] = {"standstill": False, "enabled": True}

    # Simulate long press tick (timer at multiple of CRUISE_LONG_PRESS)
    CS = car.CarState.new_message(cruiseState={"available": True})
    self.helper._update_v_cruise_non_pcm(CS, enabled=True, is_metric=True)

    # Should increase by 5 kph in metric mode
    assert self.helper.v_cruise_kph == 55

  def test_long_press_decel_5x(self):
    """Long press decel should decrement by 5x the normal amount"""
    self.helper.v_cruise_kph = 50
    self.helper.button_timers[ButtonType.decelCruise] = CRUISE_LONG_PRESS
    self.helper.button_change_states[ButtonType.decelCruise] = {"standstill": False, "enabled": True}

    CS = car.CarState.new_message(cruiseState={"available": True})
    self.helper._update_v_cruise_non_pcm(CS, enabled=True, is_metric=True)

    # Should decrease by 5 kph in metric mode
    assert self.helper.v_cruise_kph == 45

  def test_long_press_rounds_to_nearest_interval(self):
    """Long press should round to nearest interval if not aligned"""
    # Start with a speed not aligned to 5 kph
    self.helper.v_cruise_kph = 52
    self.helper.button_timers[ButtonType.accelCruise] = CRUISE_LONG_PRESS
    self.helper.button_change_states[ButtonType.accelCruise] = {"standstill": False, "enabled": True}

    CS = car.CarState.new_message(cruiseState={"available": True})
    self.helper._update_v_cruise_non_pcm(CS, enabled=True, is_metric=True)

    # Should round up to 55 (nearest 5 above 52)
    assert self.helper.v_cruise_kph == 55

  def test_long_press_decel_rounds_down(self):
    """Long press decel should round down to nearest interval"""
    # Start with a speed not aligned to 5 kph
    self.helper.v_cruise_kph = 52
    self.helper.button_timers[ButtonType.decelCruise] = CRUISE_LONG_PRESS
    self.helper.button_change_states[ButtonType.decelCruise] = {"standstill": False, "enabled": True}

    CS = car.CarState.new_message(cruiseState={"available": True})
    self.helper._update_v_cruise_non_pcm(CS, enabled=True, is_metric=True)

    # Should round down to 50 (nearest 5 below 52)
    assert self.helper.v_cruise_kph == 50


class TestButtonTimers:
  """Test button timer management"""

  def setup_method(self):
    self.CP = car.CarParams.new_message(pcmCruise=False)
    self.helper = VCruiseHelper(self.CP)

  def test_button_press_starts_timer(self):
    CS = car.CarState.new_message(cruiseState={"available": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.accelCruise, pressed=True)]

    self.helper.update_button_timers(CS, enabled=True)

    assert self.helper.button_timers[ButtonType.accelCruise] == 1

  def test_button_release_stops_timer(self):
    self.helper.button_timers[ButtonType.accelCruise] = 10

    CS = car.CarState.new_message(cruiseState={"available": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.accelCruise, pressed=False)]

    self.helper.update_button_timers(CS, enabled=True)

    assert self.helper.button_timers[ButtonType.accelCruise] == 0

  def test_timer_increments_while_pressed(self):
    self.helper.button_timers[ButtonType.accelCruise] = 5

    CS = car.CarState.new_message(cruiseState={"available": True})
    CS.buttonEvents = []  # No button events

    self.helper.update_button_timers(CS, enabled=True)

    assert self.helper.button_timers[ButtonType.accelCruise] == 6

  def test_timer_does_not_increment_when_zero(self):
    self.helper.button_timers[ButtonType.accelCruise] = 0

    CS = car.CarState.new_message(cruiseState={"available": True})
    CS.buttonEvents = []

    self.helper.update_button_timers(CS, enabled=True)

    assert self.helper.button_timers[ButtonType.accelCruise] == 0

  def test_button_change_state_recorded(self):
    CS = car.CarState.new_message(cruiseState={"available": True, "standstill": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.accelCruise, pressed=True)]

    self.helper.update_button_timers(CS, enabled=False)

    assert self.helper.button_change_states[ButtonType.accelCruise]["standstill"] is True
    assert self.helper.button_change_states[ButtonType.accelCruise]["enabled"] is False


class TestInitializeVCruise:
  """Test initialize_v_cruise method"""

  def setup_method(self):
    self.CP = car.CarParams.new_message(pcmCruise=False)
    self.helper = VCruiseHelper(self.CP)

  def test_pcm_cruise_does_nothing(self):
    CP = car.CarParams.new_message(pcmCruise=True)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = V_CRUISE_UNSET

    helper.initialize_v_cruise(car.CarState.new_message(vEgo=20), experimental_mode=False)

    assert helper.v_cruise_kph == V_CRUISE_UNSET

  def test_initialize_from_v_ego(self):
    v_ego_ms = 30.0  # 30 m/s = 108 kph
    CS = car.CarState.new_message(vEgo=v_ego_ms)

    self.helper.initialize_v_cruise(CS, experimental_mode=False)

    expected = int(round(np.clip(v_ego_ms * CV.MS_TO_KPH, V_CRUISE_INITIAL, V_CRUISE_MAX)))
    assert self.helper.v_cruise_kph == expected

  def test_initialize_clips_to_initial_min(self):
    """When v_ego is low, should use V_CRUISE_INITIAL"""
    v_ego_ms = 5.0  # 5 m/s = 18 kph (below V_CRUISE_INITIAL)
    CS = car.CarState.new_message(vEgo=v_ego_ms)

    self.helper.initialize_v_cruise(CS, experimental_mode=False)

    assert self.helper.v_cruise_kph == V_CRUISE_INITIAL

  def test_initialize_clips_to_max(self):
    """When v_ego is high, should use V_CRUISE_MAX"""
    v_ego_ms = 50.0  # 50 m/s = 180 kph (above V_CRUISE_MAX)
    CS = car.CarState.new_message(vEgo=v_ego_ms)

    self.helper.initialize_v_cruise(CS, experimental_mode=False)

    assert self.helper.v_cruise_kph == V_CRUISE_MAX

  def test_initialize_experimental_mode_uses_higher_initial(self):
    """Experimental mode should use V_CRUISE_INITIAL_EXPERIMENTAL_MODE"""
    v_ego_ms = 5.0  # Low speed
    CS = car.CarState.new_message(vEgo=v_ego_ms)

    self.helper.initialize_v_cruise(CS, experimental_mode=True)

    assert self.helper.v_cruise_kph == V_CRUISE_INITIAL_EXPERIMENTAL_MODE

  def test_resume_restores_last_speed(self):
    """Pressing accelCruise or resumeCruise should restore last speed"""
    # Set up previous speed
    self.helper.v_cruise_kph = 80
    self.helper.v_cruise_kph_last = 80
    # Mark as initialized
    assert self.helper.v_cruise_initialized

    # Simulate disabling (v_cruise_kph becomes UNSET but v_cruise_kph_last remains)
    self.helper.v_cruise_kph_last = 80
    self.helper.v_cruise_kph = V_CRUISE_UNSET

    # Re-initialize but update v_cruise_kph_last to preserve old value
    self.helper.v_cruise_kph_last = 80

    # Now initialize with resume button
    CS = car.CarState.new_message(vEgo=20)
    CS.buttonEvents = [ButtonEvent(type=ButtonType.resumeCruise, pressed=False)]

    # Need to have been initialized before
    self.helper.v_cruise_kph = 50  # Set to something to mark as initialized
    self.helper.v_cruise_kph_last = 80

    self.helper.initialize_v_cruise(CS, experimental_mode=False)

    assert self.helper.v_cruise_kph == 80

  def test_initialize_sets_cluster_speed(self):
    v_ego_ms = 25.0
    CS = car.CarState.new_message(vEgo=v_ego_ms)

    self.helper.initialize_v_cruise(CS, experimental_mode=False)

    assert self.helper.v_cruise_cluster_kph == self.helper.v_cruise_kph


class TestStandstillBehavior:
  """Test behavior during standstill"""

  def setup_method(self):
    self.CP = car.CarParams.new_message(pcmCruise=False)
    self.helper = VCruiseHelper(self.CP)
    self.helper.v_cruise_kph = 50
    self.helper.button_change_states[ButtonType.accelCruise] = {"standstill": False, "enabled": True}

  def test_accel_at_standstill_does_not_increase_speed(self):
    """Pressing accel while in standstill should not increase speed (used for resume)"""
    initial_speed = self.helper.v_cruise_kph

    # Mark that we entered button press at standstill
    self.helper.button_change_states[ButtonType.accelCruise] = {"standstill": True, "enabled": True}

    CS = car.CarState.new_message(cruiseState={"available": True, "standstill": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.accelCruise, pressed=False)]
    # Simulate the timer as if button was just released
    self.helper.button_timers[ButtonType.accelCruise] = 1

    self.helper._update_v_cruise_non_pcm(CS, enabled=True, is_metric=True)

    assert self.helper.v_cruise_kph == initial_speed

  def test_decel_at_standstill_decreases_speed(self):
    """Pressing decel while in standstill should still decrease speed"""
    initial_speed = self.helper.v_cruise_kph
    self.helper.button_change_states[ButtonType.decelCruise] = {"standstill": False, "enabled": True}

    CS = car.CarState.new_message(cruiseState={"available": True, "standstill": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.decelCruise, pressed=False)]
    self.helper.button_timers[ButtonType.decelCruise] = 1

    self.helper._update_v_cruise_non_pcm(CS, enabled=True, is_metric=True)

    assert self.helper.v_cruise_kph == initial_speed - 1


class TestGasPressedBehavior:
  """Test behavior when gas is pressed"""

  def setup_method(self):
    self.CP = car.CarParams.new_message(pcmCruise=False)
    self.helper = VCruiseHelper(self.CP)
    self.helper.v_cruise_kph = 50
    self.helper.button_change_states[ButtonType.decelCruise] = {"standstill": False, "enabled": True}

  def test_decel_with_gas_clips_to_vego(self):
    """When gas pressed and decel button used, clip to max of vEgo"""
    v_ego_kph = 60
    v_ego_ms = v_ego_kph * CV.KPH_TO_MS

    CS = car.CarState.new_message(vEgo=v_ego_ms, gasPressed=True, cruiseState={"available": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.decelCruise, pressed=False)]
    self.helper.button_timers[ButtonType.decelCruise] = 1

    self.helper._update_v_cruise_non_pcm(CS, enabled=True, is_metric=True)

    # Speed should be at least v_ego (since gas is pressed and we're overriding)
    assert self.helper.v_cruise_kph >= v_ego_kph - 1  # Account for rounding


class TestEdgeCases:
  """Test edge cases and boundary conditions"""

  def setup_method(self):
    self.CP = car.CarParams.new_message(pcmCruise=False)
    self.helper = VCruiseHelper(self.CP)

  def test_no_button_events_no_change(self):
    self.helper.v_cruise_kph = 50
    self.helper.button_change_states[ButtonType.accelCruise] = {"standstill": False, "enabled": True}

    CS = car.CarState.new_message(cruiseState={"available": True})
    CS.buttonEvents = []

    self.helper._update_v_cruise_non_pcm(CS, enabled=True, is_metric=True)

    assert self.helper.v_cruise_kph == 50

  def test_multiple_button_events(self):
    """Only first relevant button event should be processed"""
    self.helper.v_cruise_kph = 50
    self.helper.button_change_states[ButtonType.accelCruise] = {"standstill": False, "enabled": True}
    self.helper.button_change_states[ButtonType.decelCruise] = {"standstill": False, "enabled": True}

    CS = car.CarState.new_message(cruiseState={"available": True})
    # Both buttons released at same time (unlikely but possible)
    CS.buttonEvents = [
      ButtonEvent(type=ButtonType.accelCruise, pressed=False),
      ButtonEvent(type=ButtonType.decelCruise, pressed=False),
    ]
    self.helper.button_timers[ButtonType.accelCruise] = 1

    self.helper._update_v_cruise_non_pcm(CS, enabled=True, is_metric=True)

    # Should process first button (accel)
    assert self.helper.v_cruise_kph == 51

  def test_very_low_speed_at_init(self):
    """Test initialization at zero speed"""
    CS = car.CarState.new_message(vEgo=0)

    self.helper.initialize_v_cruise(CS, experimental_mode=False)

    assert self.helper.v_cruise_kph == V_CRUISE_INITIAL

  def test_speed_precision(self):
    """Test that speed values are rounded to 1 decimal place"""
    self.helper.v_cruise_kph = 50.0
    self.helper.button_change_states[ButtonType.accelCruise] = {"standstill": False, "enabled": True}

    CS = car.CarState.new_message(cruiseState={"available": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.accelCruise, pressed=False)]
    self.helper.button_timers[ButtonType.accelCruise] = 1

    # Use imperial mode which has non-integer increment
    self.helper._update_v_cruise_non_pcm(CS, enabled=True, is_metric=False)

    # Result should be rounded to 1 decimal
    assert self.helper.v_cruise_kph == round(self.helper.v_cruise_kph, 1)
