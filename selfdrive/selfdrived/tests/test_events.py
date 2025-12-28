"""Tests for selfdrive/selfdrived/events.py - event types and alerts."""
import unittest
from unittest.mock import MagicMock

from cereal import log, car
from openpilot.selfdrive.selfdrived.events import (
  Priority, ET, Events, Alert, EmptyAlert,
  NoEntryAlert, SoftDisableAlert, UserSoftDisableAlert, ImmediateDisableAlert,
  EngagementAlert, NormalPermanentAlert, StartupAlert,
  get_display_speed, EVENT_NAME, EVENTS,
)
from openpilot.common.realtime import DT_CTRL

AlertStatus = log.SelfdriveState.AlertStatus
AlertSize = log.SelfdriveState.AlertSize
VisualAlert = car.CarControl.HUDControl.VisualAlert
AudibleAlert = car.CarControl.HUDControl.AudibleAlert
EventName = log.OnroadEvent.EventName


class TestPriority(unittest.TestCase):
  """Test Priority enum."""

  def test_priority_ordering(self):
    """Test priorities are ordered correctly."""
    self.assertLess(Priority.LOWEST, Priority.LOWER)
    self.assertLess(Priority.LOWER, Priority.LOW)
    self.assertLess(Priority.LOW, Priority.MID)
    self.assertLess(Priority.MID, Priority.HIGH)
    self.assertLess(Priority.HIGH, Priority.HIGHEST)

  def test_priority_values(self):
    """Test priority values are integers 0-5."""
    self.assertEqual(Priority.LOWEST, 0)
    self.assertEqual(Priority.HIGHEST, 5)


class TestET(unittest.TestCase):
  """Test ET (event type) constants."""

  def test_enable_constant(self):
    """Test ENABLE constant."""
    self.assertEqual(ET.ENABLE, 'enable')

  def test_pre_enable_constant(self):
    """Test PRE_ENABLE constant."""
    self.assertEqual(ET.PRE_ENABLE, 'preEnable')

  def test_override_lateral_constant(self):
    """Test OVERRIDE_LATERAL constant."""
    self.assertEqual(ET.OVERRIDE_LATERAL, 'overrideLateral')

  def test_override_longitudinal_constant(self):
    """Test OVERRIDE_LONGITUDINAL constant."""
    self.assertEqual(ET.OVERRIDE_LONGITUDINAL, 'overrideLongitudinal')

  def test_no_entry_constant(self):
    """Test NO_ENTRY constant."""
    self.assertEqual(ET.NO_ENTRY, 'noEntry')

  def test_warning_constant(self):
    """Test WARNING constant."""
    self.assertEqual(ET.WARNING, 'warning')

  def test_user_disable_constant(self):
    """Test USER_DISABLE constant."""
    self.assertEqual(ET.USER_DISABLE, 'userDisable')

  def test_soft_disable_constant(self):
    """Test SOFT_DISABLE constant."""
    self.assertEqual(ET.SOFT_DISABLE, 'softDisable')

  def test_immediate_disable_constant(self):
    """Test IMMEDIATE_DISABLE constant."""
    self.assertEqual(ET.IMMEDIATE_DISABLE, 'immediateDisable')

  def test_permanent_constant(self):
    """Test PERMANENT constant."""
    self.assertEqual(ET.PERMANENT, 'permanent')


class TestEvents(unittest.TestCase):
  """Test Events class."""

  def test_init_empty(self):
    """Test Events initializes empty."""
    events = Events()
    self.assertEqual(len(events), 0)
    self.assertEqual(events.events, [])

  def test_add_event(self):
    """Test adding an event."""
    events = Events()
    events.add(EventName.startup)
    self.assertEqual(len(events), 1)
    self.assertIn(EventName.startup, events.events)

  def test_add_multiple_events(self):
    """Test adding multiple events."""
    events = Events()
    events.add(EventName.startup)
    events.add(EventName.fcw)
    self.assertEqual(len(events), 2)

  def test_add_static_event(self):
    """Test adding a static event."""
    events = Events()
    events.add(EventName.dashcamMode, static=True)
    self.assertIn(EventName.dashcamMode, events.static_events)
    self.assertIn(EventName.dashcamMode, events.events)

  def test_clear_events(self):
    """Test clearing events."""
    events = Events()
    events.add(EventName.startup)
    events.add(EventName.fcw)
    events.clear()
    self.assertEqual(len(events), 0)

  def test_clear_preserves_static_events(self):
    """Test clear preserves static events."""
    events = Events()
    events.add(EventName.dashcamMode, static=True)
    events.add(EventName.startup)
    events.clear()
    self.assertEqual(len(events), 1)
    self.assertIn(EventName.dashcamMode, events.events)

  def test_names_property(self):
    """Test names property returns events list."""
    events = Events()
    events.add(EventName.startup)
    self.assertEqual(events.names, events.events)

  def test_contains_event_type(self):
    """Test contains checks for event type."""
    events = Events()
    events.add(EventName.startup)
    self.assertTrue(events.contains(ET.PERMANENT))

  def test_contains_returns_false_when_no_match(self):
    """Test contains returns False when type not found."""
    events = Events()
    events.add(EventName.stockFcw)  # Has no alerts
    self.assertFalse(events.contains(ET.WARNING))

  def test_to_msg(self):
    """Test to_msg creates proper messages."""
    events = Events()
    events.add(EventName.startup)
    msgs = events.to_msg()
    self.assertEqual(len(msgs), 1)
    self.assertEqual(msgs[0].name, EventName.startup)

  def test_add_from_msg(self):
    """Test add_from_msg adds events from messages."""
    events = Events()

    # Create mock event message
    mock_event = MagicMock()
    mock_event.name.raw = EventName.fcw

    events.add_from_msg([mock_event])
    self.assertIn(EventName.fcw, events.events)

  def test_events_sorted(self):
    """Test events are kept sorted."""
    events = Events()
    events.add(EventName.fcw)
    events.add(EventName.startup)
    # Events should be sorted by event name value
    self.assertEqual(events.events, sorted(events.events))


class TestAlert(unittest.TestCase):
  """Test Alert class."""

  def test_alert_initialization(self):
    """Test Alert initializes correctly."""
    alert = Alert(
      "Test Title", "Test Description",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none,
      duration=2.0
    )

    self.assertEqual(alert.alert_text_1, "Test Title")
    self.assertEqual(alert.alert_text_2, "Test Description")
    self.assertEqual(alert.alert_status, AlertStatus.normal)
    self.assertEqual(alert.alert_size, AlertSize.mid)
    self.assertEqual(alert.priority, Priority.LOW)
    self.assertEqual(alert.visual_alert, VisualAlert.none)
    self.assertEqual(alert.audible_alert, AudibleAlert.none)

  def test_alert_duration_converted(self):
    """Test duration is converted to frames."""
    alert = Alert(
      "", "", AlertStatus.normal, AlertSize.none,
      Priority.LOWEST, VisualAlert.none, AudibleAlert.none,
      duration=2.0
    )
    self.assertEqual(alert.duration, int(2.0 / DT_CTRL))

  def test_alert_creation_delay(self):
    """Test creation_delay is stored."""
    alert = Alert(
      "", "", AlertStatus.normal, AlertSize.none,
      Priority.LOWEST, VisualAlert.none, AudibleAlert.none,
      duration=1.0, creation_delay=0.5
    )
    self.assertEqual(alert.creation_delay, 0.5)

  def test_alert_str(self):
    """Test Alert string representation."""
    alert = Alert(
      "Title", "Description",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none,
      duration=1.0
    )
    result = str(alert)
    self.assertIn("Title", result)
    self.assertIn("Description", result)

  def test_alert_comparison(self):
    """Test Alert priority comparison."""
    low_alert = Alert("", "", AlertStatus.normal, AlertSize.none,
                      Priority.LOW, VisualAlert.none, AudibleAlert.none, 1.0)
    high_alert = Alert("", "", AlertStatus.normal, AlertSize.none,
                       Priority.HIGH, VisualAlert.none, AudibleAlert.none, 1.0)

    self.assertGreater(high_alert, low_alert)
    self.assertFalse(low_alert > high_alert)

  def test_alert_comparison_non_alert(self):
    """Test Alert comparison with non-Alert returns False."""
    alert = Alert("", "", AlertStatus.normal, AlertSize.none,
                  Priority.LOW, VisualAlert.none, AudibleAlert.none, 1.0)
    self.assertFalse(alert > "not an alert")


class TestEmptyAlert(unittest.TestCase):
  """Test EmptyAlert constant."""

  def test_empty_alert_exists(self):
    """Test EmptyAlert is defined."""
    self.assertIsInstance(EmptyAlert, Alert)

  def test_empty_alert_properties(self):
    """Test EmptyAlert has correct properties."""
    self.assertEqual(EmptyAlert.alert_text_1, "")
    self.assertEqual(EmptyAlert.alert_text_2, "")
    self.assertEqual(EmptyAlert.priority, Priority.LOWEST)


class TestNoEntryAlert(unittest.TestCase):
  """Test NoEntryAlert class."""

  def test_no_entry_alert_creation(self):
    """Test NoEntryAlert creation."""
    alert = NoEntryAlert("Test reason")
    self.assertIn("openpilot", alert.alert_text_1.lower())
    self.assertEqual(alert.alert_text_2, "Test reason")
    self.assertEqual(alert.priority, Priority.LOW)

  def test_no_entry_alert_custom_title(self):
    """Test NoEntryAlert with custom title."""
    alert = NoEntryAlert("Reason", alert_text_1="Custom Title")
    self.assertEqual(alert.alert_text_1, "Custom Title")


class TestSoftDisableAlert(unittest.TestCase):
  """Test SoftDisableAlert class."""

  def test_soft_disable_alert_creation(self):
    """Test SoftDisableAlert creation."""
    alert = SoftDisableAlert("Take control reason")
    self.assertIn("TAKE CONTROL", alert.alert_text_1.upper())
    self.assertEqual(alert.alert_text_2, "Take control reason")
    self.assertEqual(alert.priority, Priority.MID)
    self.assertEqual(alert.alert_status, AlertStatus.userPrompt)


class TestUserSoftDisableAlert(unittest.TestCase):
  """Test UserSoftDisableAlert class."""

  def test_user_soft_disable_alert(self):
    """Test UserSoftDisableAlert has different text."""
    alert = UserSoftDisableAlert("User triggered reason")
    self.assertIn("disengage", alert.alert_text_1.lower())


class TestImmediateDisableAlert(unittest.TestCase):
  """Test ImmediateDisableAlert class."""

  def test_immediate_disable_alert(self):
    """Test ImmediateDisableAlert creation."""
    alert = ImmediateDisableAlert("Critical issue")
    self.assertIn("TAKE CONTROL", alert.alert_text_1.upper())
    self.assertEqual(alert.priority, Priority.HIGHEST)
    self.assertEqual(alert.alert_status, AlertStatus.critical)


class TestEngagementAlert(unittest.TestCase):
  """Test EngagementAlert class."""

  def test_engagement_alert_creation(self):
    """Test EngagementAlert creation."""
    alert = EngagementAlert(AudibleAlert.engage)
    self.assertEqual(alert.alert_text_1, "")
    self.assertEqual(alert.alert_text_2, "")
    self.assertEqual(alert.audible_alert, AudibleAlert.engage)
    self.assertEqual(alert.priority, Priority.MID)


class TestNormalPermanentAlert(unittest.TestCase):
  """Test NormalPermanentAlert class."""

  def test_normal_permanent_alert_one_line(self):
    """Test NormalPermanentAlert with one line."""
    alert = NormalPermanentAlert("Single line")
    self.assertEqual(alert.alert_text_1, "Single line")
    self.assertEqual(alert.alert_text_2, "")
    self.assertEqual(alert.alert_size, AlertSize.small)

  def test_normal_permanent_alert_two_lines(self):
    """Test NormalPermanentAlert with two lines."""
    alert = NormalPermanentAlert("Line 1", "Line 2")
    self.assertEqual(alert.alert_text_1, "Line 1")
    self.assertEqual(alert.alert_text_2, "Line 2")
    self.assertEqual(alert.alert_size, AlertSize.mid)


class TestStartupAlert(unittest.TestCase):
  """Test StartupAlert class."""

  def test_startup_alert_creation(self):
    """Test StartupAlert creation."""
    alert = StartupAlert("Welcome")
    self.assertEqual(alert.alert_text_1, "Welcome")
    self.assertEqual(alert.priority, Priority.LOWER)


class TestGetDisplaySpeed(unittest.TestCase):
  """Test get_display_speed function."""

  def test_metric_speed(self):
    """Test metric speed display."""
    result = get_display_speed(10.0, metric=True)
    self.assertIn("km/h", result)
    self.assertIn("36", result)  # ~36 km/h

  def test_imperial_speed(self):
    """Test imperial speed display."""
    result = get_display_speed(10.0, metric=False)
    self.assertIn("mph", result)
    self.assertIn("22", result)  # ~22 mph

  def test_zero_speed(self):
    """Test zero speed."""
    result = get_display_speed(0.0, metric=True)
    self.assertIn("0", result)


class TestEVENTS(unittest.TestCase):
  """Test EVENTS dictionary."""

  def test_events_not_empty(self):
    """Test EVENTS dictionary is not empty."""
    self.assertGreater(len(EVENTS), 0)

  def test_events_contains_startup(self):
    """Test EVENTS contains startup event."""
    self.assertIn(EventName.startup, EVENTS)

  def test_events_contains_fcw(self):
    """Test EVENTS contains fcw event."""
    self.assertIn(EventName.fcw, EVENTS)

  def test_events_values_are_dicts(self):
    """Test EVENTS values are dictionaries."""
    for event_name, event_types in EVENTS.items():
      self.assertIsInstance(event_types, dict,
                            f"Event {event_name} value is not a dict")

  def test_event_types_are_valid(self):
    """Test event types use valid ET constants."""
    valid_types = {
      ET.ENABLE, ET.PRE_ENABLE, ET.OVERRIDE_LATERAL, ET.OVERRIDE_LONGITUDINAL,
      ET.NO_ENTRY, ET.WARNING, ET.USER_DISABLE, ET.SOFT_DISABLE,
      ET.IMMEDIATE_DISABLE, ET.PERMANENT
    }
    for event_name, event_types in EVENTS.items():
      for et in event_types.keys():
        self.assertIn(et, valid_types,
                      f"Event {event_name} has invalid type {et}")


class TestEVENT_NAME(unittest.TestCase):
  """Test EVENT_NAME reverse mapping."""

  def test_event_name_not_empty(self):
    """Test EVENT_NAME is not empty."""
    self.assertGreater(len(EVENT_NAME), 0)

  def test_event_name_contains_startup(self):
    """Test EVENT_NAME maps startup."""
    self.assertIn(EventName.startup, EVENT_NAME)
    self.assertEqual(EVENT_NAME[EventName.startup], "startup")


if __name__ == '__main__':
  unittest.main()
