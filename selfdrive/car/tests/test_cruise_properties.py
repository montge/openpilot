"""Property-based tests for selfdrive/car/cruise.py using Hypothesis.

These tests verify mathematical properties that should hold for all valid inputs,
complementing the unit tests with broader coverage.
"""

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from cereal import car

from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import (
  VCruiseHelper,
  V_CRUISE_MIN,
  V_CRUISE_MAX,
  V_CRUISE_UNSET,
)

ButtonType = car.CarState.ButtonEvent.Type

HYPOTHESIS_SETTINGS = settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])


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


class TestVCruiseHelperSpeedBoundsProperties:
  """Property-based tests for speed bounds."""

  @given(
    v_cruise_kph=st.floats(min_value=V_CRUISE_MIN, max_value=V_CRUISE_MAX, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_v_cruise_non_pcm_always_valid(self, mocker, v_cruise_kph):
    """Property: Non-PCM v_cruise_kph is always in [V_CRUISE_MIN, V_CRUISE_MAX] after update."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = v_cruise_kph

    CS = create_mock_cs(mocker)
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    # For non-PCM, speed should remain in valid range or be UNSET
    assert helper.v_cruise_kph == V_CRUISE_UNSET or V_CRUISE_MIN <= helper.v_cruise_kph <= V_CRUISE_MAX

  @given(
    v_ego=st.floats(min_value=0.0, max_value=80.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_initialize_clips_to_bounds(self, mocker, v_ego):
    """Property: initialize_v_cruise clips speed to [initial, V_CRUISE_MAX]."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(mocker, v_ego=v_ego * CV.KPH_TO_MS)  # v_ego in m/s
    helper.initialize_v_cruise(CS, experimental_mode=False)

    # Should be within bounds
    assert V_CRUISE_MIN <= helper.v_cruise_kph <= V_CRUISE_MAX

  @given(
    v_ego=st.floats(min_value=0.0, max_value=80.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_initialize_experimental_clips_to_bounds(self, mocker, v_ego):
    """Property: experimental mode initialization also clips to bounds."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(mocker, v_ego=v_ego * CV.KPH_TO_MS)
    helper.initialize_v_cruise(CS, experimental_mode=True)

    assert V_CRUISE_MIN <= helper.v_cruise_kph <= V_CRUISE_MAX


class TestVCruiseHelperNonPCMProperties:
  """Property-based tests for non-PCM cruise behavior."""

  @given(
    v_cruise_kph=st.floats(min_value=V_CRUISE_MIN, max_value=V_CRUISE_MAX, allow_nan=False, allow_infinity=False),
    is_metric=st.booleans(),
  )
  @HYPOTHESIS_SETTINGS
  def test_speed_stays_in_bounds_after_adjustment(self, mocker, v_cruise_kph, is_metric):
    """Property: Speed adjustments keep v_cruise_kph in valid range."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = v_cruise_kph

    CS = create_mock_cs(mocker)
    helper.update_v_cruise(CS, enabled=True, is_metric=is_metric)

    # After update, speed should still be valid
    assert V_CRUISE_MIN <= helper.v_cruise_kph <= V_CRUISE_MAX

  @given(
    v_ego=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    is_metric=st.booleans(),
  )
  @HYPOTHESIS_SETTINGS
  def test_cluster_speed_matches_cruise_speed(self, mocker, v_ego, is_metric):
    """Property: For non-PCM, cluster speed equals cruise speed after update."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    CS = create_mock_cs(mocker, v_ego=v_ego)
    helper.update_v_cruise(CS, enabled=True, is_metric=is_metric)

    assert helper.v_cruise_kph == helper.v_cruise_cluster_kph


class TestVCruiseHelperButtonTimerProperties:
  """Property-based tests for button timers."""

  @given(
    num_updates=st.integers(min_value=0, max_value=100),
  )
  @HYPOTHESIS_SETTINGS
  def test_button_timers_non_negative(self, mocker, num_updates):
    """Property: Button timers are always >= 0."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(mocker)

    for _ in range(num_updates):
      helper.update_button_timers(CS, enabled=True)

    for timer in helper.button_timers.values():
      assert timer >= 0

  @given(
    num_updates=st.integers(min_value=1, max_value=100),
  )
  @HYPOTHESIS_SETTINGS
  def test_button_timers_increment_correctly(self, mocker, num_updates):
    """Property: Button timers increment by 1 each update when active."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)

    # Start with a timer active
    helper.button_timers[ButtonType.accelCruise] = 1

    CS = create_mock_cs(mocker)

    for _ in range(num_updates):
      initial_timer = helper.button_timers[ButtonType.accelCruise]
      helper.update_button_timers(CS, enabled=True)
      # Timer should increment by 1 if it was > 0
      if initial_timer > 0:
        assert helper.button_timers[ButtonType.accelCruise] == initial_timer + 1


class TestVCruiseHelperPCMProperties:
  """Property-based tests for PCM cruise behavior."""

  @given(
    cruise_speed=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_pcm_cruise_follows_car_speed(self, mocker, cruise_speed):
    """Property: PCM cruise uses car's cruise speed."""
    CP = create_mock_cp(mocker, pcm_cruise=True)
    helper = VCruiseHelper(CP)

    CS = create_mock_cs(mocker, cruise_speed=cruise_speed)
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    if cruise_speed == 0:
      assert helper.v_cruise_kph == V_CRUISE_UNSET
    elif cruise_speed == -1:
      assert helper.v_cruise_kph == -1
    else:
      expected = cruise_speed * CV.MS_TO_KPH
      assert abs(helper.v_cruise_kph - expected) < 0.01

  @given(
    cruise_speed=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
    cruise_available=st.booleans(),
  )
  @HYPOTHESIS_SETTINGS
  def test_cruise_not_available_unsets(self, mocker, cruise_speed, cruise_available):
    """Property: Cruise unavailable sets speed to UNSET."""
    CP = create_mock_cp(mocker, pcm_cruise=True)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = 50

    CS = create_mock_cs(mocker, cruise_speed=cruise_speed, cruise_available=cruise_available)
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    if not cruise_available:
      assert helper.v_cruise_kph == V_CRUISE_UNSET


class TestVCruiseHelperInitializedProperty:
  """Property-based tests for v_cruise_initialized property."""

  @given(
    v_cruise=st.floats(min_value=-10.0, max_value=300.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_initialized_correct(self, mocker, v_cruise):
    """Property: v_cruise_initialized is True exactly when v_cruise_kph != V_CRUISE_UNSET."""
    CP = create_mock_cp(mocker)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = v_cruise

    expected = v_cruise != V_CRUISE_UNSET
    assert helper.v_cruise_initialized == expected


class TestVCruiseHelperLastSpeedProperties:
  """Property-based tests for v_cruise_kph_last tracking."""

  @given(
    initial_speed=st.floats(min_value=V_CRUISE_MIN, max_value=V_CRUISE_MAX, allow_nan=False, allow_infinity=False),
    cruise_speed=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_last_speed_updated(self, mocker, initial_speed, cruise_speed):
    """Property: v_cruise_kph_last stores previous value after update."""
    CP = create_mock_cp(mocker, pcm_cruise=True)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = initial_speed

    CS = create_mock_cs(mocker, cruise_speed=cruise_speed)
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    assert helper.v_cruise_kph_last == initial_speed


class TestVCruiseHelperGasPressedProperties:
  """Property-based tests for gas pressed behavior."""

  @given(
    v_cruise_kph=st.floats(min_value=V_CRUISE_MIN, max_value=V_CRUISE_MAX, allow_nan=False, allow_infinity=False),
    v_ego=st.floats(min_value=5.0, max_value=40.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_speed_stays_valid_with_gas(self, mocker, v_cruise_kph, v_ego):
    """Property: Speed remains in valid bounds even with gas pressed."""
    CP = create_mock_cp(mocker, pcm_cruise=False)
    helper = VCruiseHelper(CP)
    helper.v_cruise_kph = v_cruise_kph

    CS = create_mock_cs(mocker, v_ego=v_ego, gas_pressed=True)
    helper.update_v_cruise(CS, enabled=True, is_metric=True)

    # Speed should be within valid bounds
    assert V_CRUISE_MIN <= helper.v_cruise_kph <= V_CRUISE_MAX
