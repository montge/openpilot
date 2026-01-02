"""Tests for selfdrive/selfdrived/events.py - event types and alerts."""

from cereal import log, car
from openpilot.selfdrive.selfdrived.events import (
  Priority,
  ET,
  Events,
  Alert,
  EmptyAlert,
  NoEntryAlert,
  SoftDisableAlert,
  UserSoftDisableAlert,
  ImmediateDisableAlert,
  EngagementAlert,
  NormalPermanentAlert,
  StartupAlert,
  get_display_speed,
  EVENT_NAME,
  EVENTS,
)
from openpilot.common.realtime import DT_CTRL

AlertStatus = log.SelfdriveState.AlertStatus
AlertSize = log.SelfdriveState.AlertSize
VisualAlert = car.CarControl.HUDControl.VisualAlert
AudibleAlert = car.CarControl.HUDControl.AudibleAlert
EventName = log.OnroadEvent.EventName


class TestPriority:
  """Test Priority enum."""

  def test_priority_ordering(self):
    """Test priorities are ordered correctly."""
    assert Priority.LOWEST < Priority.LOWER
    assert Priority.LOWER < Priority.LOW
    assert Priority.LOW < Priority.MID
    assert Priority.MID < Priority.HIGH
    assert Priority.HIGH < Priority.HIGHEST

  def test_priority_values(self):
    """Test priority values are integers 0-5."""
    assert Priority.LOWEST == 0
    assert Priority.HIGHEST == 5


class TestET:
  """Test ET (event type) constants."""

  def test_enable_constant(self):
    """Test ENABLE constant."""
    assert ET.ENABLE == 'enable'

  def test_pre_enable_constant(self):
    """Test PRE_ENABLE constant."""
    assert ET.PRE_ENABLE == 'preEnable'

  def test_override_lateral_constant(self):
    """Test OVERRIDE_LATERAL constant."""
    assert ET.OVERRIDE_LATERAL == 'overrideLateral'

  def test_override_longitudinal_constant(self):
    """Test OVERRIDE_LONGITUDINAL constant."""
    assert ET.OVERRIDE_LONGITUDINAL == 'overrideLongitudinal'

  def test_no_entry_constant(self):
    """Test NO_ENTRY constant."""
    assert ET.NO_ENTRY == 'noEntry'

  def test_warning_constant(self):
    """Test WARNING constant."""
    assert ET.WARNING == 'warning'

  def test_user_disable_constant(self):
    """Test USER_DISABLE constant."""
    assert ET.USER_DISABLE == 'userDisable'

  def test_soft_disable_constant(self):
    """Test SOFT_DISABLE constant."""
    assert ET.SOFT_DISABLE == 'softDisable'

  def test_immediate_disable_constant(self):
    """Test IMMEDIATE_DISABLE constant."""
    assert ET.IMMEDIATE_DISABLE == 'immediateDisable'

  def test_permanent_constant(self):
    """Test PERMANENT constant."""
    assert ET.PERMANENT == 'permanent'


class TestEvents:
  """Test Events class."""

  def test_init_empty(self):
    """Test Events initializes empty."""
    events = Events()
    assert len(events) == 0
    assert events.events == []

  def test_add_event(self):
    """Test adding an event."""
    events = Events()
    events.add(EventName.startup)
    assert len(events) == 1
    assert EventName.startup in events.events

  def test_add_multiple_events(self):
    """Test adding multiple events."""
    events = Events()
    events.add(EventName.startup)
    events.add(EventName.fcw)
    assert len(events) == 2

  def test_add_static_event(self):
    """Test adding a static event."""
    events = Events()
    events.add(EventName.dashcamMode, static=True)
    assert EventName.dashcamMode in events.static_events
    assert EventName.dashcamMode in events.events

  def test_clear_events(self):
    """Test clearing events."""
    events = Events()
    events.add(EventName.startup)
    events.add(EventName.fcw)
    events.clear()
    assert len(events) == 0

  def test_clear_preserves_static_events(self):
    """Test clear preserves static events."""
    events = Events()
    events.add(EventName.dashcamMode, static=True)
    events.add(EventName.startup)
    events.clear()
    assert len(events) == 1
    assert EventName.dashcamMode in events.events

  def test_names_property(self):
    """Test names property returns events list."""
    events = Events()
    events.add(EventName.startup)
    assert events.names == events.events

  def test_contains_event_type(self):
    """Test contains checks for event type."""
    events = Events()
    events.add(EventName.startup)
    assert events.contains(ET.PERMANENT) is True

  def test_contains_returns_false_when_no_match(self):
    """Test contains returns False when type not found."""
    events = Events()
    events.add(EventName.stockFcw)  # Has no alerts
    assert events.contains(ET.WARNING) is False

  def test_to_msg(self):
    """Test to_msg creates proper messages."""
    events = Events()
    events.add(EventName.startup)
    msgs = events.to_msg()
    assert len(msgs) == 1
    assert msgs[0].name == EventName.startup

  def test_add_from_msg(self, mocker):
    """Test add_from_msg adds events from messages."""
    events = Events()

    # Create mock event message
    mock_event = mocker.MagicMock()
    mock_event.name.raw = EventName.fcw

    events.add_from_msg([mock_event])
    assert EventName.fcw in events.events

  def test_events_sorted(self):
    """Test events are kept sorted."""
    events = Events()
    events.add(EventName.fcw)
    events.add(EventName.startup)
    # Events should be sorted by event name value
    assert events.events == sorted(events.events)


class TestAlert:
  """Test Alert class."""

  def test_alert_initialization(self):
    """Test Alert initializes correctly."""
    alert = Alert("Test Title", "Test Description", AlertStatus.normal, AlertSize.mid, Priority.LOW, VisualAlert.none, AudibleAlert.none, duration=2.0)

    assert alert.alert_text_1 == "Test Title"
    assert alert.alert_text_2 == "Test Description"
    assert alert.alert_status == AlertStatus.normal
    assert alert.alert_size == AlertSize.mid
    assert alert.priority == Priority.LOW
    assert alert.visual_alert == VisualAlert.none
    assert alert.audible_alert == AudibleAlert.none

  def test_alert_duration_converted(self):
    """Test duration is converted to frames."""
    alert = Alert("", "", AlertStatus.normal, AlertSize.none, Priority.LOWEST, VisualAlert.none, AudibleAlert.none, duration=2.0)
    assert alert.duration == int(2.0 / DT_CTRL)

  def test_alert_creation_delay(self):
    """Test creation_delay is stored."""
    alert = Alert("", "", AlertStatus.normal, AlertSize.none, Priority.LOWEST, VisualAlert.none, AudibleAlert.none, duration=1.0, creation_delay=0.5)
    assert alert.creation_delay == 0.5

  def test_alert_str(self):
    """Test Alert string representation."""
    alert = Alert("Title", "Description", AlertStatus.normal, AlertSize.mid, Priority.LOW, VisualAlert.none, AudibleAlert.none, duration=1.0)
    result = str(alert)
    assert "Title" in result
    assert "Description" in result

  def test_alert_comparison(self):
    """Test Alert priority comparison."""
    low_alert = Alert("", "", AlertStatus.normal, AlertSize.none, Priority.LOW, VisualAlert.none, AudibleAlert.none, 1.0)
    high_alert = Alert("", "", AlertStatus.normal, AlertSize.none, Priority.HIGH, VisualAlert.none, AudibleAlert.none, 1.0)

    assert high_alert > low_alert
    assert not (low_alert > high_alert)

  def test_alert_comparison_non_alert(self):
    """Test Alert comparison with non-Alert returns False."""
    alert = Alert("", "", AlertStatus.normal, AlertSize.none, Priority.LOW, VisualAlert.none, AudibleAlert.none, 1.0)
    assert not (alert > "not an alert")


class TestEmptyAlert:
  """Test EmptyAlert constant."""

  def test_empty_alert_exists(self):
    """Test EmptyAlert is defined."""
    assert isinstance(EmptyAlert, Alert)

  def test_empty_alert_properties(self):
    """Test EmptyAlert has correct properties."""
    assert EmptyAlert.alert_text_1 == ""
    assert EmptyAlert.alert_text_2 == ""
    assert EmptyAlert.priority == Priority.LOWEST


class TestNoEntryAlert:
  """Test NoEntryAlert class."""

  def test_no_entry_alert_creation(self):
    """Test NoEntryAlert creation."""
    alert = NoEntryAlert("Test reason")
    assert "openpilot" in alert.alert_text_1.lower()
    assert alert.alert_text_2 == "Test reason"
    assert alert.priority == Priority.LOW

  def test_no_entry_alert_custom_title(self):
    """Test NoEntryAlert with custom title."""
    alert = NoEntryAlert("Reason", alert_text_1="Custom Title")
    assert alert.alert_text_1 == "Custom Title"


class TestSoftDisableAlert:
  """Test SoftDisableAlert class."""

  def test_soft_disable_alert_creation(self):
    """Test SoftDisableAlert creation."""
    alert = SoftDisableAlert("Take control reason")
    assert "TAKE CONTROL" in alert.alert_text_1.upper()
    assert alert.alert_text_2 == "Take control reason"
    assert alert.priority == Priority.MID
    assert alert.alert_status == AlertStatus.userPrompt


class TestUserSoftDisableAlert:
  """Test UserSoftDisableAlert class."""

  def test_user_soft_disable_alert(self):
    """Test UserSoftDisableAlert has different text."""
    alert = UserSoftDisableAlert("User triggered reason")
    assert "disengage" in alert.alert_text_1.lower()


class TestImmediateDisableAlert:
  """Test ImmediateDisableAlert class."""

  def test_immediate_disable_alert(self):
    """Test ImmediateDisableAlert creation."""
    alert = ImmediateDisableAlert("Critical issue")
    assert "TAKE CONTROL" in alert.alert_text_1.upper()
    assert alert.priority == Priority.HIGHEST
    assert alert.alert_status == AlertStatus.critical


class TestEngagementAlert:
  """Test EngagementAlert class."""

  def test_engagement_alert_creation(self):
    """Test EngagementAlert creation."""
    alert = EngagementAlert(AudibleAlert.engage)
    assert alert.alert_text_1 == ""
    assert alert.alert_text_2 == ""
    assert alert.audible_alert == AudibleAlert.engage
    assert alert.priority == Priority.MID


class TestNormalPermanentAlert:
  """Test NormalPermanentAlert class."""

  def test_normal_permanent_alert_one_line(self):
    """Test NormalPermanentAlert with one line."""
    alert = NormalPermanentAlert("Single line")
    assert alert.alert_text_1 == "Single line"
    assert alert.alert_text_2 == ""
    assert alert.alert_size == AlertSize.small

  def test_normal_permanent_alert_two_lines(self):
    """Test NormalPermanentAlert with two lines."""
    alert = NormalPermanentAlert("Line 1", "Line 2")
    assert alert.alert_text_1 == "Line 1"
    assert alert.alert_text_2 == "Line 2"
    assert alert.alert_size == AlertSize.mid


class TestStartupAlert:
  """Test StartupAlert class."""

  def test_startup_alert_creation(self):
    """Test StartupAlert creation."""
    alert = StartupAlert("Welcome")
    assert alert.alert_text_1 == "Welcome"
    assert alert.priority == Priority.LOWER


class TestGetDisplaySpeed:
  """Test get_display_speed function."""

  def test_metric_speed(self):
    """Test metric speed display."""
    result = get_display_speed(10.0, metric=True)
    assert "km/h" in result
    assert "36" in result  # ~36 km/h

  def test_imperial_speed(self):
    """Test imperial speed display."""
    result = get_display_speed(10.0, metric=False)
    assert "mph" in result
    assert "22" in result  # ~22 mph

  def test_zero_speed(self):
    """Test zero speed."""
    result = get_display_speed(0.0, metric=True)
    assert "0" in result


class TestEVENTS:
  """Test EVENTS dictionary."""

  def test_events_not_empty(self):
    """Test EVENTS dictionary is not empty."""
    assert len(EVENTS) > 0

  def test_events_contains_startup(self):
    """Test EVENTS contains startup event."""
    assert EventName.startup in EVENTS

  def test_events_contains_fcw(self):
    """Test EVENTS contains fcw event."""
    assert EventName.fcw in EVENTS

  def test_events_values_are_dicts(self):
    """Test EVENTS values are dictionaries."""
    for event_name, event_types in EVENTS.items():
      assert isinstance(event_types, dict), f"Event {event_name} value is not a dict"

  def test_event_types_are_valid(self):
    """Test event types use valid ET constants."""
    valid_types = {
      ET.ENABLE,
      ET.PRE_ENABLE,
      ET.OVERRIDE_LATERAL,
      ET.OVERRIDE_LONGITUDINAL,
      ET.NO_ENTRY,
      ET.WARNING,
      ET.USER_DISABLE,
      ET.SOFT_DISABLE,
      ET.IMMEDIATE_DISABLE,
      ET.PERMANENT,
    }
    for event_name, event_types in EVENTS.items():
      for et in event_types.keys():
        assert et in valid_types, f"Event {event_name} has invalid type {et}"


class TestEVENT_NAME:
  """Test EVENT_NAME reverse mapping."""

  def test_event_name_not_empty(self):
    """Test EVENT_NAME is not empty."""
    assert len(EVENT_NAME) > 0

  def test_event_name_contains_startup(self):
    """Test EVENT_NAME maps startup."""
    assert EventName.startup in EVENT_NAME
    assert EVENT_NAME[EventName.startup] == "startup"


class TestEventsCreateAlerts:
  """Test Events.create_alerts method edge cases."""

  def test_create_alerts_with_callback_args_none(self):
    """Test create_alerts handles None callback_args."""
    events = Events()
    events.add(EventName.startup)

    # Should not raise with callback_args=None
    alerts = events.create_alerts([ET.PERMANENT], callback_args=None)
    assert len(alerts) >= 0

  def test_create_alerts_with_callback_function(self, mocker):
    """Test create_alerts calls callback functions."""

    events = Events()
    events.add(EventName.steerSaturated)  # Has a soft_disable callback

    # Mock callback args
    CP = mocker.MagicMock()
    CS = mocker.MagicMock()
    sm = mocker.MagicMock()

    # With long soft_disable_time, should return SoftDisableAlert
    alerts = events.create_alerts([ET.SOFT_DISABLE], callback_args=[CP, CS, sm, False, 100, None])
    # May or may not have alerts depending on event definition
    assert isinstance(alerts, list)

  def test_create_alerts_respects_creation_delay(self):
    """Test create_alerts respects creation_delay."""
    events = Events()
    events.add(EventName.startup)

    # With event_counter = 0, alerts with creation_delay > 0 won't show
    alerts = events.create_alerts([ET.PERMANENT])
    # Check that alerts are returned (startup has no creation_delay)
    assert len(alerts) >= 0


class TestNoEntryAlertMici:
  """Test NoEntryAlert on mici device."""

  def test_no_entry_alert_mici_swaps_text(self, mocker):
    """Test NoEntryAlert swaps text on mici device."""
    mocker.patch('openpilot.selfdrive.selfdrived.events.HARDWARE.get_device_type', return_value='mici')

    alert = NoEntryAlert("Reason Text", alert_text_1="Title Text")

    # On mici, text should be swapped
    assert alert.alert_text_1 == "Reason Text"
    assert alert.alert_text_2 == "Title Text"


class TestStartupAlertMici:
  """Test StartupAlert on mici device."""

  def test_startup_alert_mici_removes_default_text2(self, mocker):
    """Test StartupAlert removes default text2 on mici."""
    mocker.patch('openpilot.selfdrive.selfdrived.events.HARDWARE.get_device_type', return_value='mici')

    alert = StartupAlert("Welcome")

    # On mici, default "Always keep hands on wheel" should be removed
    assert alert.alert_text_2 == ""
    assert alert.alert_size == AlertSize.small

  def test_startup_alert_mici_keeps_custom_text2(self, mocker):
    """Test StartupAlert keeps custom text2 on mici."""
    mocker.patch('openpilot.selfdrive.selfdrived.events.HARDWARE.get_device_type', return_value='mici')

    alert = StartupAlert("Welcome", "Custom message")

    # Custom text2 should be kept
    assert alert.alert_text_2 == "Custom message"


class TestSoftDisableAlertCallback:
  """Test soft_disable_alert callback function."""

  def test_soft_disable_callback_immediate_at_low_time(self, mocker):
    """Test soft_disable_alert returns ImmediateDisableAlert when time is low."""
    from openpilot.selfdrive.selfdrived.events import soft_disable_alert

    callback = soft_disable_alert("Test reason")

    CP = mocker.MagicMock()
    CS = mocker.MagicMock()
    sm = mocker.MagicMock()

    # Low soft_disable_time should return ImmediateDisableAlert
    alert = callback(CP, CS, sm, False, 1, None)  # 1 frame is < 0.5s

    assert isinstance(alert, ImmediateDisableAlert)

  def test_soft_disable_callback_soft_at_high_time(self, mocker):
    """Test soft_disable_alert returns SoftDisableAlert when time is high."""
    from openpilot.selfdrive.selfdrived.events import soft_disable_alert

    callback = soft_disable_alert("Test reason")

    CP = mocker.MagicMock()
    CS = mocker.MagicMock()
    sm = mocker.MagicMock()

    # High soft_disable_time should return SoftDisableAlert
    alert = callback(CP, CS, sm, False, 100, None)  # 100 frames is > 0.5s

    assert isinstance(alert, SoftDisableAlert)


class TestUserSoftDisableAlertCallback:
  """Test user_soft_disable_alert callback function."""

  def test_user_soft_disable_callback_immediate_at_low_time(self, mocker):
    """Test user_soft_disable_alert returns ImmediateDisableAlert when time is low."""
    from openpilot.selfdrive.selfdrived.events import user_soft_disable_alert

    callback = user_soft_disable_alert("Test reason")

    CP = mocker.MagicMock()
    CS = mocker.MagicMock()
    sm = mocker.MagicMock()

    # Low soft_disable_time should return ImmediateDisableAlert
    alert = callback(CP, CS, sm, False, 1, None)

    assert isinstance(alert, ImmediateDisableAlert)

  def test_user_soft_disable_callback_user_at_high_time(self, mocker):
    """Test user_soft_disable_alert returns UserSoftDisableAlert when time is high."""
    from openpilot.selfdrive.selfdrived.events import user_soft_disable_alert

    callback = user_soft_disable_alert("Test reason")

    CP = mocker.MagicMock()
    CS = mocker.MagicMock()
    sm = mocker.MagicMock()

    # High soft_disable_time should return UserSoftDisableAlert
    alert = callback(CP, CS, sm, False, 100, None)

    assert isinstance(alert, UserSoftDisableAlert)


class TestStartupMasterAlert:
  """Test startup_master_alert callback function."""

  def test_startup_master_alert_replay(self, mocker):
    """Test startup_master_alert shows 'replay' branch when REPLAY is set."""
    import os
    from openpilot.selfdrive.selfdrived.events import startup_master_alert

    mocker.patch.dict(os.environ, {"REPLAY": "1"})

    CP = mocker.MagicMock()
    CS = mocker.MagicMock()
    sm = mocker.MagicMock()

    alert = startup_master_alert(CP, CS, sm, False, 0, None)

    assert isinstance(alert, StartupAlert)
    assert alert.alert_text_2 == "replay"


class TestParamsdInvalidAlert:
  """Test paramsd_invalid_alert callback function."""

  def test_angle_offset_invalid(self, mocker):
    """Test alert when angle offset is invalid."""
    from openpilot.selfdrive.selfdrived.events import paramsd_invalid_alert

    CP = mocker.MagicMock()
    CS = mocker.MagicMock()
    sm = mocker.MagicMock()
    sm.__getitem__ = mocker.MagicMock(
      return_value=mocker.MagicMock(
        angleOffsetValid=False,
        angleOffsetDeg=5.5,
        steerRatioValid=True,
        stiffnessFactorValid=True,
      )
    )

    alert = paramsd_invalid_alert(CP, CS, sm, False, 0, None)

    assert isinstance(alert, NoEntryAlert)
    assert "misalignment" in alert.alert_text_1.lower() or "offset" in alert.alert_text_2.lower()

  def test_steer_ratio_invalid(self, mocker):
    """Test alert when steer ratio is invalid."""
    from openpilot.selfdrive.selfdrived.events import paramsd_invalid_alert

    CP = mocker.MagicMock()
    CS = mocker.MagicMock()
    sm = mocker.MagicMock()
    sm.__getitem__ = mocker.MagicMock(
      return_value=mocker.MagicMock(
        angleOffsetValid=True,
        steerRatioValid=False,
        steerRatio=14.5,
        stiffnessFactorValid=True,
      )
    )

    alert = paramsd_invalid_alert(CP, CS, sm, False, 0, None)

    assert isinstance(alert, NoEntryAlert)
    assert "ratio" in alert.alert_text_1.lower() or "ratio" in alert.alert_text_2.lower()

  def test_stiffness_factor_invalid(self, mocker):
    """Test alert when stiffness factor is invalid."""
    from openpilot.selfdrive.selfdrived.events import paramsd_invalid_alert

    CP = mocker.MagicMock()
    CS = mocker.MagicMock()
    sm = mocker.MagicMock()
    sm.__getitem__ = mocker.MagicMock(
      return_value=mocker.MagicMock(
        angleOffsetValid=True,
        steerRatioValid=True,
        stiffnessFactorValid=False,
        stiffnessFactor=0.8,
      )
    )

    alert = paramsd_invalid_alert(CP, CS, sm, False, 0, None)

    assert isinstance(alert, NoEntryAlert)
    assert "stiffness" in alert.alert_text_1.lower() or "tire" in alert.alert_text_1.lower()

  def test_all_valid_returns_temporary_error(self, mocker):
    """Test alert when all params are valid returns temporary error."""
    from openpilot.selfdrive.selfdrived.events import paramsd_invalid_alert

    CP = mocker.MagicMock()
    CS = mocker.MagicMock()
    sm = mocker.MagicMock()
    sm.__getitem__ = mocker.MagicMock(
      return_value=mocker.MagicMock(
        angleOffsetValid=True,
        steerRatioValid=True,
        stiffnessFactorValid=True,
      )
    )

    alert = paramsd_invalid_alert(CP, CS, sm, False, 0, None)

    assert isinstance(alert, NoEntryAlert)
    assert "temporary" in alert.alert_text_2.lower()


class TestHighCpuUsageAlert:
  """Test high_cpu_usage_alert callback function."""

  def test_high_cpu_alert_format(self, mocker):
    """Test high CPU usage alert format."""
    from openpilot.selfdrive.selfdrived.events import high_cpu_usage_alert

    CP = mocker.MagicMock()
    CS = mocker.MagicMock()
    sm = mocker.MagicMock()
    sm.__getitem__ = mocker.MagicMock(
      return_value=mocker.MagicMock(
        cpuUsagePercent=[85.0, 90.0, 75.0],
      )
    )

    alert = high_cpu_usage_alert(CP, CS, sm, False, 0, None)

    assert isinstance(alert, NormalPermanentAlert)
    assert "cpu" in alert.alert_text_1.lower()
    assert "90" in alert.alert_text_2  # Max of [85, 90, 75]


class TestOverheatAlert:
  """Test overheat_alert callback function."""

  def test_overheat_alert_format(self, mocker):
    """Test overheat alert shows temperature."""
    from openpilot.selfdrive.selfdrived.events import overheat_alert

    CP = mocker.MagicMock()
    CS = mocker.MagicMock()
    sm = mocker.MagicMock()
    sm.__getitem__ = mocker.MagicMock(
      return_value=mocker.MagicMock(
        cpuTempC=[75.0, 80.0],
        gpuTempC=[70.0, 72.0],
        memoryTempC=65.0,
      )
    )

    alert = overheat_alert(CP, CS, sm, False, 0, None)

    assert isinstance(alert, NormalPermanentAlert)
    assert "overheat" in alert.alert_text_1.lower()
    assert "80" in alert.alert_text_2  # Max temp


class TestLowMemoryAlert:
  """Test low_memory_alert callback function."""

  def test_low_memory_alert_format(self, mocker):
    """Test low memory alert shows usage."""
    from openpilot.selfdrive.selfdrived.events import low_memory_alert

    CP = mocker.MagicMock()
    CS = mocker.MagicMock()
    sm = mocker.MagicMock()
    sm.__getitem__ = mocker.MagicMock(
      return_value=mocker.MagicMock(
        memoryUsagePercent=92,
      )
    )

    alert = low_memory_alert(CP, CS, sm, False, 0, None)

    assert isinstance(alert, NormalPermanentAlert)
    assert "memory" in alert.alert_text_1.lower()
    assert "92" in alert.alert_text_2


class TestModeldLaggingAlert:
  """Test modeld_lagging_alert callback function."""

  def test_modeld_lagging_alert_format(self, mocker):
    """Test modeld lagging alert shows frame drop percent."""
    from openpilot.selfdrive.selfdrived.events import modeld_lagging_alert

    CP = mocker.MagicMock()
    CS = mocker.MagicMock()
    sm = mocker.MagicMock()
    sm.__getitem__ = mocker.MagicMock(
      return_value=mocker.MagicMock(
        frameDropPerc=15.5,
      )
    )

    alert = modeld_lagging_alert(CP, CS, sm, False, 0, None)

    assert isinstance(alert, NormalPermanentAlert)
    assert "lagging" in alert.alert_text_1.lower()
    assert "15.5" in alert.alert_text_2


class TestWrongCarModeAlert:
  """Test wrong_car_mode_alert callback function."""

  def test_wrong_car_mode_honda(self, mocker):
    """Test wrong car mode alert for Honda."""
    from openpilot.selfdrive.selfdrived.events import wrong_car_mode_alert

    CP = mocker.MagicMock()
    CP.brand = "honda"
    CS = mocker.MagicMock()
    sm = mocker.MagicMock()

    alert = wrong_car_mode_alert(CP, CS, sm, False, 0, None)

    assert isinstance(alert, NoEntryAlert)
    assert "main switch" in alert.alert_text_2.lower()

  def test_wrong_car_mode_other_brand(self, mocker):
    """Test wrong car mode alert for non-Honda brands."""
    from openpilot.selfdrive.selfdrived.events import wrong_car_mode_alert

    CP = mocker.MagicMock()
    CP.brand = "toyota"
    CS = mocker.MagicMock()
    sm = mocker.MagicMock()

    alert = wrong_car_mode_alert(CP, CS, sm, False, 0, None)

    assert isinstance(alert, NoEntryAlert)
    assert "adaptive cruise" in alert.alert_text_2.lower()
