import random

from openpilot.selfdrive.selfdrived.events import Alert, EmptyAlert, EVENTS, ET
from openpilot.selfdrive.selfdrived.alertmanager import AlertManager, AlertEntry, set_offroad_alert, OFFROAD_ALERTS
from openpilot.common.params import Params


class TestAlertManager:
  def test_duration(self):
    """
    Enforce that an alert lasts for max(alert duration, duration the alert is added)
    """
    for duration in range(1, 100):
      alert = None
      while not isinstance(alert, Alert):
        event = random.choice([e for e in EVENTS.values() if len(e)])
        alert = random.choice(list(event.values()))

      alert.duration = duration

      # check two cases:
      # - alert is added to AM for <= the alert's duration
      # - alert is added to AM for > alert's duration
      for greater in (True, False):
        if greater:
          add_duration = duration + random.randint(1, 10)
        else:
          add_duration = random.randint(1, duration)
        show_duration = max(duration, add_duration)

        AM = AlertManager()
        for frame in range(duration + 10):
          if frame < add_duration:
            AM.add_many(
              frame,
              [
                alert,
              ],
            )
          AM.process_alerts(frame, set())

          shown = AM.current_alert != EmptyAlert
          should_show = frame <= show_duration
          assert shown == should_show, f"{frame=} {add_duration=} {duration=}"

      # check one case:
      # - if alert is re-added to AM before it ends the duration is extended
      if duration > 1:
        AM = AlertManager()
        show_duration = duration * 2
        for frame in range(duration * 2 + 10):
          if frame == 0:
            AM.add_many(
              frame,
              [
                alert,
              ],
            )

          if frame == duration:
            # add alert one frame before it ends
            assert AM.current_alert == alert
            AM.add_many(
              frame,
              [
                alert,
              ],
            )
          AM.process_alerts(frame, set())

          shown = AM.current_alert != EmptyAlert
          should_show = frame <= show_duration
          assert shown == should_show, f"{frame=} {duration=}"


class TestAlertManagerClearEventTypes:
  """Test clear_event_types functionality in process_alerts."""

  def test_clear_event_types_clears_matching_alerts(self):
    """Test that alerts with matching event_type are cleared."""
    AM = AlertManager()

    # Find an alert with a known event type
    alert = None
    for event in EVENTS.values():
      for _et, a in event.items():
        if isinstance(a, Alert):
          alert = a
          break
      if alert:
        break

    assert alert is not None, "Need at least one Alert in EVENTS"

    # Add the alert
    AM.add_many(0, [alert])
    AM.process_alerts(0, set())
    assert AM.current_alert == alert

    # Clear it using clear_event_types
    AM.add_many(1, [alert])
    AM.process_alerts(1, {alert.event_type})

    # Alert's end_frame should be set to -1, making it inactive
    entry = AM.alerts[alert.alert_type]
    assert entry.end_frame == -1

  def test_clear_event_types_does_not_affect_other_types(self):
    """Test that alerts with different event_type are not cleared."""
    AM = AlertManager()

    # Find an alert
    alert = None
    for event in EVENTS.values():
      for _et, a in event.items():
        if isinstance(a, Alert):
          alert = a
          break
      if alert:
        break

    assert alert is not None

    # Add the alert
    AM.add_many(0, [alert])
    AM.process_alerts(0, set())

    # Try to clear with a different event type
    other_type = ET.PERMANENT if alert.event_type != ET.PERMANENT else ET.WARNING
    AM.add_many(1, [alert])
    AM.process_alerts(1, {other_type})

    # Alert should still be active
    assert AM.current_alert == alert


class TestProcessAlertsWithNoneAlert:
  """Test process_alerts handles AlertEntry with alert=None."""

  def test_process_alerts_skips_entry_with_none_alert(self):
    """Test process_alerts skips AlertEntry where alert is None."""
    AM = AlertManager()

    # Manually add an AlertEntry with no alert set
    AM.alerts['test_type'] = AlertEntry()
    assert AM.alerts['test_type'].alert is None

    # process_alerts should skip this entry without error
    AM.process_alerts(0, set())

    # Current alert should be EmptyAlert since there are no valid alerts
    assert AM.current_alert == EmptyAlert


class TestAlertEntry:
  """Test AlertEntry dataclass."""

  def test_alert_entry_default_values(self):
    """Test AlertEntry has correct default values."""
    entry = AlertEntry()
    assert entry.alert is None
    assert entry.start_frame == -1
    assert entry.end_frame == -1
    assert entry.added_frame == -1

  def test_alert_entry_active(self):
    """Test active method returns True when frame <= end_frame."""
    entry = AlertEntry(end_frame=10)
    assert entry.active(5)
    assert entry.active(10)
    assert not entry.active(11)

  def test_alert_entry_just_added(self):
    """Test just_added returns True only on the frame after added_frame."""
    entry = AlertEntry(end_frame=10, added_frame=5)
    assert not entry.just_added(5)  # Same frame as added
    assert entry.just_added(6)  # One frame after added
    assert not entry.just_added(7)  # Two frames after added

  def test_alert_entry_just_added_inactive(self):
    """Test just_added returns False if not active."""
    entry = AlertEntry(end_frame=5, added_frame=5)
    assert not entry.just_added(10)  # Frame past end_frame


class TestSetOffroadAlert:
  """Test set_offroad_alert function."""

  def test_set_offroad_alert_show(self):
    """Test set_offroad_alert puts alert in params."""
    params = Params()
    # Get first offroad alert key
    alert_key = list(OFFROAD_ALERTS.keys())[0]

    set_offroad_alert(alert_key, show_alert=True)

    # Alert should be in params
    assert params.get(alert_key) is not None

    # Clean up
    params.remove(alert_key)

  def test_set_offroad_alert_hide(self):
    """Test set_offroad_alert removes alert from params."""
    params = Params()
    alert_key = list(OFFROAD_ALERTS.keys())[0]

    # First set it
    set_offroad_alert(alert_key, show_alert=True)
    assert params.get(alert_key) is not None

    # Then hide it
    set_offroad_alert(alert_key, show_alert=False)
    assert params.get(alert_key) is None

  def test_set_offroad_alert_with_extra_text(self):
    """Test set_offroad_alert includes extra text."""
    params = Params()
    alert_key = list(OFFROAD_ALERTS.keys())[0]

    set_offroad_alert(alert_key, show_alert=True, extra_text="Extra info")

    data = params.get(alert_key)
    assert data is not None
    # Params.get returns dict directly for msgpack-encoded data
    if isinstance(data, bytes):
      import json

      alert_data = json.loads(data)
    else:
      alert_data = data
    assert alert_data.get('extra') == "Extra info"

    # Clean up
    params.remove(alert_key)
