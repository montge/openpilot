"""Tests for selfdrive/car/car_specific.py - car-specific event handling."""

from cereal import car, log
from opendbc.car import structs

from openpilot.selfdrive.car.car_specific import CarSpecificEvents

EventName = log.OnroadEvent.EventName
GearShifter = structs.CarState.GearShifter


def create_car_params(
  brand: str = "mock", min_steer_speed: float = 0.0, pcm_cruise: bool = False, op_long: bool = False, min_enable_speed: float = 0.0
) -> structs.CarParams:
  """Create a CarParams with specified settings."""
  CP = structs.CarParams.new_message()
  CP.brand = brand
  CP.minSteerSpeed = min_steer_speed
  CP.pcmCruise = pcm_cruise
  CP.openpilotLongitudinalControl = op_long
  CP.minEnableSpeed = min_enable_speed
  CP.carFingerprint = "MOCK"
  return CP


def create_car_state(**kwargs) -> car.CarState:
  """Create a CarState with specified fields."""
  msg = car.CarState.new_message()
  for key, value in kwargs.items():
    if hasattr(msg, key):
      setattr(msg, key, value)
  return msg


class TestCarSpecificEventsInit:
  """Test CarSpecificEvents initialization."""

  def test_init_sets_default_values(self):
    """Test init sets default state values."""
    CP = create_car_params()
    cse = CarSpecificEvents(CP)

    assert cse.steering_unpressed == 0
    assert cse.low_speed_alert is False
    assert cse.no_steer_warning is False
    assert cse.silent_steer_warning is True

  def test_init_stores_car_params(self):
    """Test init stores CarParams reference."""
    CP = create_car_params(brand="toyota")
    cse = CarSpecificEvents(CP)

    assert cse.CP.brand == "toyota"


class TestCarSpecificEventsUpdateMockBrand:
  """Test update method with mock/body brand."""

  def test_mock_brand_returns_empty_events(self):
    """Test mock brand returns empty events."""
    CP = create_car_params(brand="mock")
    cse = CarSpecificEvents(CP)

    CS = create_car_state()
    CS_prev = create_car_state()
    CC = car.CarControl.new_message()

    events = cse.update(CS, CS_prev, CC)

    assert len(events) == 0

  def test_body_brand_returns_empty_events(self):
    """Test body brand returns empty events."""
    CP = create_car_params(brand="body")
    cse = CarSpecificEvents(CP)

    CS = create_car_state()
    CS_prev = create_car_state()
    CC = car.CarControl.new_message()

    events = cse.update(CS, CS_prev, CC)

    assert len(events) == 0


class TestCarSpecificEventsChrysler:
  """Test Chrysler-specific events."""

  def test_chrysler_low_speed_alert_triggered(self):
    """Test Chrysler triggers low speed alert below minSteerSpeed."""
    CP = create_car_params(brand="chrysler", min_steer_speed=10.0)
    cse = CarSpecificEvents(CP)

    CS = create_car_state(vEgo=5.0)  # Below minSteerSpeed + 0.5
    CS_prev = create_car_state()
    CC = car.CarControl.new_message()

    events = cse.update(CS, CS_prev, CC)

    assert cse.low_speed_alert is True
    assert EventName.belowSteerSpeed in events.names

  def test_chrysler_low_speed_alert_cleared(self):
    """Test Chrysler clears low speed alert above threshold."""
    CP = create_car_params(brand="chrysler", min_steer_speed=10.0)
    cse = CarSpecificEvents(CP)
    cse.low_speed_alert = True

    CS = create_car_state(vEgo=12.0)  # Above minSteerSpeed + 1.0
    CS_prev = create_car_state()
    CC = car.CarControl.new_message()

    cse.update(CS, CS_prev, CC)

    assert cse.low_speed_alert is False


class TestCarSpecificEventsHonda:
  """Test Honda-specific events."""

  def test_honda_below_engage_speed(self):
    """Test Honda triggers belowEngageSpeed when too slow."""
    CP = create_car_params(brand="honda", pcm_cruise=True, min_enable_speed=5.0)
    cse = CarSpecificEvents(CP)

    CS = create_car_state(vEgo=3.0)
    CS_prev = create_car_state()
    CC = car.CarControl.new_message()

    events = cse.update(CS, CS_prev, CC)

    assert EventName.belowEngageSpeed in events.names

  def test_honda_pcm_enable_event(self):
    """Test Honda triggers pcmEnable on cruise activation."""
    CP = create_car_params(brand="honda", pcm_cruise=True, min_enable_speed=0.0)
    cse = CarSpecificEvents(CP)

    CS = create_car_state(vEgo=20.0)
    CS.cruiseState.enabled = True
    CS_prev = create_car_state()
    CS_prev.cruiseState.enabled = False
    CC = car.CarControl.new_message()

    events = cse.update(CS, CS_prev, CC)

    assert EventName.pcmEnable in events.names

  def test_honda_manual_restart_at_standstill(self):
    """Test Honda triggers manualRestart at standstill with minEnableSpeed."""
    CP = create_car_params(brand="honda", pcm_cruise=True, min_enable_speed=5.0)
    cse = CarSpecificEvents(CP)

    CS = create_car_state(vEgo=0.0)
    CS_prev = create_car_state()
    CC = car.CarControl.new_message()

    events = cse.update(CS, CS_prev, CC)

    assert EventName.manualRestart in events.names


class TestCarSpecificEventsToyota:
  """Test Toyota-specific events."""

  def test_toyota_resume_required_in_standstill(self):
    """Test Toyota triggers resumeRequired when in standstill."""
    CP = create_car_params(brand="toyota", op_long=True)
    cse = CarSpecificEvents(CP)

    CS = create_car_state(vEgo=0.0)
    CS.cruiseState.standstill = True
    CS.brakePressed = False
    CS_prev = create_car_state()
    CC = car.CarControl.new_message()
    CC.cruiseControl.resume = True

    events = cse.update(CS, CS_prev, CC)

    assert EventName.resumeRequired in events.names

  def test_toyota_below_engage_speed(self):
    """Test Toyota triggers belowEngageSpeed when too slow."""
    CP = create_car_params(brand="toyota", op_long=True, min_enable_speed=5.0)
    cse = CarSpecificEvents(CP)

    CS = create_car_state(vEgo=3.0)
    CS_prev = create_car_state()
    CC = car.CarControl.new_message()

    events = cse.update(CS, CS_prev, CC)

    assert EventName.belowEngageSpeed in events.names


class TestCarSpecificEventsGM:
  """Test GM-specific events."""

  def test_gm_below_engage_speed(self):
    """Test GM triggers belowEngageSpeed when too slow."""
    CP = create_car_params(brand="gm", min_enable_speed=5.0)
    cse = CarSpecificEvents(CP)

    CS = create_car_state(vEgo=3.0, standstill=False)
    CS_prev = create_car_state()
    CC = car.CarControl.new_message()

    events = cse.update(CS, CS_prev, CC)

    assert EventName.belowEngageSpeed in events.names

  def test_gm_resume_required_in_standstill(self):
    """Test GM triggers resumeRequired when cruise in standstill."""
    CP = create_car_params(brand="gm")
    cse = CarSpecificEvents(CP)

    CS = create_car_state(vEgo=0.0)
    CS.cruiseState.standstill = True
    CS_prev = create_car_state()
    CC = car.CarControl.new_message()

    events = cse.update(CS, CS_prev, CC)

    assert EventName.resumeRequired in events.names


class TestCarSpecificEventsVolkswagen:
  """Test Volkswagen-specific events."""

  def test_vw_below_engage_speed(self):
    """Test VW triggers belowEngageSpeed when too slow."""
    CP = create_car_params(brand="volkswagen", op_long=True, min_enable_speed=5.0)
    cse = CarSpecificEvents(CP)

    CS = create_car_state(vEgo=4.0)  # Below minEnableSpeed + 0.5
    CS_prev = create_car_state()
    CC = car.CarControl.new_message()

    events = cse.update(CS, CS_prev, CC)

    assert EventName.belowEngageSpeed in events.names

  def test_vw_speed_too_low_when_enabled(self):
    """Test VW triggers speedTooLow when enabled and too slow."""
    CP = create_car_params(brand="volkswagen", op_long=True, min_enable_speed=5.0)
    cse = CarSpecificEvents(CP)

    CS = create_car_state(vEgo=3.0)
    CS_prev = create_car_state()
    CC = car.CarControl.new_message()
    CC.enabled = True

    events = cse.update(CS, CS_prev, CC)

    assert EventName.speedTooLow in events.names


class TestCreateCommonEvents:
  """Test create_common_events method."""

  def test_door_open_event(self):
    """Test doorOpen triggers event."""
    CP = create_car_params(brand="toyota")
    cse = CarSpecificEvents(CP)

    CS = create_car_state(doorOpen=True)
    CS_prev = create_car_state()

    events = cse.create_common_events(CS, CS_prev)

    assert EventName.doorOpen in events.names

  def test_seatbelt_unlatched_event(self):
    """Test seatbeltUnlatched triggers event."""
    CP = create_car_params(brand="toyota")
    cse = CarSpecificEvents(CP)

    CS = create_car_state(seatbeltUnlatched=True)
    CS_prev = create_car_state()

    events = cse.create_common_events(CS, CS_prev)

    assert EventName.seatbeltNotLatched in events.names

  def test_reverse_gear_event(self):
    """Test reverse gear triggers event."""
    CP = create_car_params(brand="toyota")
    cse = CarSpecificEvents(CP)

    CS = create_car_state()
    CS.gearShifter = GearShifter.reverse
    CS_prev = create_car_state()

    events = cse.create_common_events(CS, CS_prev)

    assert EventName.reverseGear in events.names

  def test_esp_disabled_event(self):
    """Test espDisabled triggers event."""
    CP = create_car_params(brand="toyota")
    cse = CarSpecificEvents(CP)

    CS = create_car_state(espDisabled=True)
    CS_prev = create_car_state()

    events = cse.create_common_events(CS, CS_prev)

    assert EventName.espDisabled in events.names

  def test_steering_pressed_event(self):
    """Test steeringPressed triggers steerOverride event."""
    CP = create_car_params(brand="toyota")
    cse = CarSpecificEvents(CP)

    CS = create_car_state(steeringPressed=True)
    CS_prev = create_car_state()

    events = cse.create_common_events(CS, CS_prev)

    assert EventName.steerOverride in events.names

  def test_gas_pressed_event(self):
    """Test gasPressed triggers gasPressedOverride event."""
    CP = create_car_params(brand="toyota")
    cse = CarSpecificEvents(CP)

    CS = create_car_state(gasPressed=True)
    CS_prev = create_car_state()

    events = cse.create_common_events(CS, CS_prev)

    assert EventName.gasPressedOverride in events.names

  def test_steer_fault_permanent_event(self):
    """Test steerFaultPermanent triggers steerUnavailable event."""
    CP = create_car_params(brand="toyota")
    cse = CarSpecificEvents(CP)

    CS = create_car_state(steerFaultPermanent=True)
    CS_prev = create_car_state()

    events = cse.create_common_events(CS, CS_prev)

    assert EventName.steerUnavailable in events.names

  def test_steer_fault_temporary_silent(self):
    """Test steerFaultTemporary triggers silent event initially."""
    CP = create_car_params(brand="toyota")
    cse = CarSpecificEvents(CP)
    cse.silent_steer_warning = True

    CS = create_car_state(steerFaultTemporary=True, standstill=False)
    CS_prev = create_car_state()

    events = cse.create_common_events(CS, CS_prev)

    assert EventName.steerTempUnavailableSilent in events.names

  def test_pcm_enable_event(self):
    """Test pcmEnable event on cruise activation."""
    CP = create_car_params(brand="toyota", pcm_cruise=True)
    cse = CarSpecificEvents(CP)

    CS = create_car_state()
    CS.cruiseState.enabled = True
    CS.blockPcmEnable = False
    CS_prev = create_car_state()
    CS_prev.cruiseState.enabled = False

    events = cse.create_common_events(CS, CS_prev)

    assert EventName.pcmEnable in events.names

  def test_pcm_disable_event(self):
    """Test pcmDisable event when cruise disabled."""
    CP = create_car_params(brand="toyota", pcm_cruise=True)
    cse = CarSpecificEvents(CP)

    CS = create_car_state()
    CS.cruiseState.enabled = False
    CS_prev = create_car_state()

    events = cse.create_common_events(CS, CS_prev)

    assert EventName.pcmDisable in events.names
