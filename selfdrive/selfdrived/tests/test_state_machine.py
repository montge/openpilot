from cereal import log
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.selfdrived.state import StateMachine, SOFT_DISABLE_TIME, ACTIVE_STATES
from openpilot.selfdrive.selfdrived.events import Events, ET, EVENTS, NormalPermanentAlert

State = log.SelfdriveState.OpenpilotState

# The event types that maintain the current state
MAINTAIN_STATES = {
  State.enabled: (None,),
  State.disabled: (None,),
  State.softDisabling: (ET.SOFT_DISABLE,),
  State.preEnabled: (ET.PRE_ENABLE,),
  State.overriding: (ET.OVERRIDE_LATERAL, ET.OVERRIDE_LONGITUDINAL),
}
ALL_STATES = tuple(State.schema.enumerants.values())
# The event types checked in DISABLED section of state machine
ENABLE_EVENT_TYPES = (ET.ENABLE, ET.PRE_ENABLE, ET.OVERRIDE_LATERAL, ET.OVERRIDE_LONGITUDINAL)


def make_event(event_types):
  event = {}
  for ev in event_types:
    event[ev] = NormalPermanentAlert("alert")
  EVENTS[0] = event
  return 0


class TestStateMachine:
  def setup_method(self):
    self.events = Events()
    self.state_machine = StateMachine()
    self.state_machine.soft_disable_timer = int(SOFT_DISABLE_TIME / DT_CTRL)

  def test_immediate_disable(self):
    for state in ALL_STATES:
      for et in MAINTAIN_STATES[state]:
        self.events.add(make_event([et, ET.IMMEDIATE_DISABLE]))
        self.state_machine.state = state
        self.state_machine.update(self.events)
        assert State.disabled == self.state_machine.state
        self.events.clear()

  def test_user_disable(self):
    for state in ALL_STATES:
      for et in MAINTAIN_STATES[state]:
        self.events.add(make_event([et, ET.USER_DISABLE]))
        self.state_machine.state = state
        self.state_machine.update(self.events)
        assert State.disabled == self.state_machine.state
        self.events.clear()

  def test_soft_disable(self):
    for state in ALL_STATES:
      if state == State.preEnabled:  # preEnabled considers NO_ENTRY instead
        continue
      for et in MAINTAIN_STATES[state]:
        self.events.add(make_event([et, ET.SOFT_DISABLE]))
        self.state_machine.state = state
        self.state_machine.update(self.events)
        assert self.state_machine.state == State.disabled if state == State.disabled else State.softDisabling
        self.events.clear()

  def test_soft_disable_timer(self):
    self.state_machine.state = State.enabled
    self.events.add(make_event([ET.SOFT_DISABLE]))
    self.state_machine.update(self.events)
    for _ in range(int(SOFT_DISABLE_TIME / DT_CTRL)):
      assert self.state_machine.state == State.softDisabling
      self.state_machine.update(self.events)

    assert self.state_machine.state == State.disabled

  def test_no_entry(self):
    # Make sure noEntry keeps us disabled
    for et in ENABLE_EVENT_TYPES:
      self.events.add(make_event([ET.NO_ENTRY, et]))
      self.state_machine.update(self.events)
      assert self.state_machine.state == State.disabled
      self.events.clear()

  def test_no_entry_pre_enable(self):
    # preEnabled with noEntry event
    self.state_machine.state = State.preEnabled
    self.events.add(make_event([ET.NO_ENTRY, ET.PRE_ENABLE]))
    self.state_machine.update(self.events)
    assert self.state_machine.state == State.preEnabled

  def test_maintain_states(self):
    # Given current state's event type, we should maintain state
    for state in ALL_STATES:
      for et in MAINTAIN_STATES[state]:
        self.state_machine.state = state
        self.events.add(make_event([et]))
        self.state_machine.update(self.events)
        assert self.state_machine.state == state
        self.events.clear()


class TestStateMachineEdgeCases:
  """Edge case tests for state machine transitions."""

  def setup_method(self):
    self.events = Events()
    self.state_machine = StateMachine()

  def test_soft_disable_recovery(self):
    """softDisabling -> enabled when SOFT_DISABLE clears."""
    self.state_machine.state = State.softDisabling
    self.state_machine.soft_disable_timer = int(SOFT_DISABLE_TIME / DT_CTRL)

    # No SOFT_DISABLE event, should recover to enabled
    self.state_machine.update(self.events)
    assert self.state_machine.state == State.enabled

  def test_pre_enable_to_enabled(self):
    """preEnabled -> enabled when PRE_ENABLE clears."""
    self.state_machine.state = State.preEnabled

    # No PRE_ENABLE event, should transition to enabled
    self.state_machine.update(self.events)
    assert self.state_machine.state == State.enabled

  def test_override_to_enabled(self):
    """overriding -> enabled when override events clear."""
    self.state_machine.state = State.overriding

    # No override events, should return to enabled
    self.state_machine.update(self.events)
    assert self.state_machine.state == State.enabled

  def test_override_to_soft_disabling(self):
    """overriding -> softDisabling when SOFT_DISABLE occurs."""
    self.state_machine.state = State.overriding

    self.events.add(make_event([ET.SOFT_DISABLE]))
    self.state_machine.update(self.events)
    assert self.state_machine.state == State.softDisabling
    assert self.state_machine.soft_disable_timer == int(SOFT_DISABLE_TIME / DT_CTRL)

  def test_enabled_to_override_lateral_only(self):
    """enabled -> overriding with lateral override only."""
    self.state_machine.state = State.enabled

    self.events.add(make_event([ET.OVERRIDE_LATERAL]))
    self.state_machine.update(self.events)
    assert self.state_machine.state == State.overriding

  def test_enabled_to_override_longitudinal_only(self):
    """enabled -> overriding with longitudinal override only."""
    self.state_machine.state = State.enabled

    self.events.add(make_event([ET.OVERRIDE_LONGITUDINAL]))
    self.state_machine.update(self.events)
    assert self.state_machine.state == State.overriding

  def test_disabled_to_override_lateral(self):
    """disabled -> overriding with ENABLE + OVERRIDE_LATERAL."""
    self.state_machine.state = State.disabled

    self.events.add(make_event([ET.ENABLE, ET.OVERRIDE_LATERAL]))
    self.state_machine.update(self.events)
    assert self.state_machine.state == State.overriding

  def test_disabled_to_override_longitudinal(self):
    """disabled -> overriding with ENABLE + OVERRIDE_LONGITUDINAL."""
    self.state_machine.state = State.disabled

    self.events.add(make_event([ET.ENABLE, ET.OVERRIDE_LONGITUDINAL]))
    self.state_machine.update(self.events)
    assert self.state_machine.state == State.overriding

  def test_disabled_to_pre_enabled(self):
    """disabled -> preEnabled with ENABLE + PRE_ENABLE."""
    self.state_machine.state = State.disabled

    self.events.add(make_event([ET.ENABLE, ET.PRE_ENABLE]))
    self.state_machine.update(self.events)
    assert self.state_machine.state == State.preEnabled

  def test_disabled_to_enabled(self):
    """disabled -> enabled with ENABLE only."""
    self.state_machine.state = State.disabled

    self.events.add(make_event([ET.ENABLE]))
    self.state_machine.update(self.events)
    assert self.state_machine.state == State.enabled

  def test_soft_disable_timer_resets_on_new_entry(self):
    """Timer resets when entering softDisabling from overriding."""
    self.state_machine.state = State.overriding
    self.state_machine.soft_disable_timer = 10  # Some remaining time

    self.events.add(make_event([ET.SOFT_DISABLE]))
    self.state_machine.update(self.events)

    # Timer should be reset to full duration
    assert self.state_machine.soft_disable_timer == int(SOFT_DISABLE_TIME / DT_CTRL)

  def test_soft_disable_timer_decrements_each_update(self):
    """Timer decrements on each update call."""
    self.state_machine.state = State.softDisabling
    self.state_machine.soft_disable_timer = 100

    self.events.add(make_event([ET.SOFT_DISABLE]))
    self.state_machine.update(self.events)

    assert self.state_machine.soft_disable_timer == 99

  def test_soft_disable_timer_floor_at_zero(self):
    """Timer should not go negative."""
    self.state_machine.soft_disable_timer = 0

    self.state_machine.update(self.events)
    assert self.state_machine.soft_disable_timer == 0


class TestStateMachineAlertTypes:
  """Tests for current_alert_types in each state."""

  def setup_method(self):
    self.events = Events()
    self.state_machine = StateMachine()

  def test_alert_types_always_includes_permanent(self):
    """PERMANENT should always be in current_alert_types."""
    for state in ALL_STATES:
      self.state_machine.state = state
      self.state_machine.update(self.events)
      assert ET.PERMANENT in self.state_machine.current_alert_types

  def test_user_disable_alert_type(self):
    """USER_DISABLE should be added when user disables."""
    self.state_machine.state = State.enabled
    self.events.add(make_event([ET.USER_DISABLE]))
    self.state_machine.update(self.events)

    assert ET.USER_DISABLE in self.state_machine.current_alert_types

  def test_immediate_disable_alert_type(self):
    """IMMEDIATE_DISABLE should be added on immediate disable."""
    self.state_machine.state = State.enabled
    self.events.add(make_event([ET.IMMEDIATE_DISABLE]))
    self.state_machine.update(self.events)

    assert ET.IMMEDIATE_DISABLE in self.state_machine.current_alert_types

  def test_soft_disable_alert_type(self):
    """SOFT_DISABLE should be added when entering softDisabling."""
    self.state_machine.state = State.enabled
    self.events.add(make_event([ET.SOFT_DISABLE]))
    self.state_machine.update(self.events)

    assert ET.SOFT_DISABLE in self.state_machine.current_alert_types

  def test_soft_disable_alert_during_countdown(self):
    """SOFT_DISABLE alert should continue during countdown."""
    self.state_machine.state = State.softDisabling
    self.state_machine.soft_disable_timer = 100
    self.events.add(make_event([ET.SOFT_DISABLE]))
    self.state_machine.update(self.events)

    assert ET.SOFT_DISABLE in self.state_machine.current_alert_types

  def test_override_alert_types(self):
    """Override alert types should be added when overriding."""
    self.state_machine.state = State.enabled
    self.events.add(make_event([ET.OVERRIDE_LATERAL]))
    self.state_machine.update(self.events)

    assert ET.OVERRIDE_LATERAL in self.state_machine.current_alert_types
    assert ET.OVERRIDE_LONGITUDINAL in self.state_machine.current_alert_types

  def test_pre_enable_alert_type(self):
    """PRE_ENABLE alert should be added in preEnabled state."""
    self.state_machine.state = State.preEnabled
    self.events.add(make_event([ET.PRE_ENABLE]))
    self.state_machine.update(self.events)

    assert ET.PRE_ENABLE in self.state_machine.current_alert_types

  def test_enable_alert_type(self):
    """ENABLE alert should be added when enabling."""
    self.state_machine.state = State.disabled
    self.events.add(make_event([ET.ENABLE]))
    self.state_machine.update(self.events)

    assert ET.ENABLE in self.state_machine.current_alert_types

  def test_no_entry_alert_type(self):
    """NO_ENTRY alert should be added when blocked from enabling."""
    self.state_machine.state = State.disabled
    self.events.add(make_event([ET.ENABLE, ET.NO_ENTRY]))
    self.state_machine.update(self.events)

    assert ET.NO_ENTRY in self.state_machine.current_alert_types

  def test_warning_alert_in_active_states(self):
    """WARNING alert should be added in active states."""
    for state in ACTIVE_STATES:
      self.state_machine.state = state
      self.state_machine.soft_disable_timer = 100  # Keep softDisabling valid
      self.events.add(make_event([ET.SOFT_DISABLE]))  # Maintain softDisabling
      self.state_machine.update(self.events)

      assert ET.WARNING in self.state_machine.current_alert_types
      self.events.clear()


class TestStateMachineReturnValues:
  """Tests for enabled and active return values."""

  def setup_method(self):
    self.events = Events()
    self.state_machine = StateMachine()

  def test_disabled_returns_not_enabled_not_active(self):
    """Disabled state should return enabled=False, active=False."""
    self.state_machine.state = State.disabled
    enabled, active = self.state_machine.update(self.events)

    assert enabled is False
    assert active is False

  def test_pre_enabled_returns_enabled_not_active(self):
    """preEnabled should return enabled=True, active=False."""
    self.state_machine.state = State.preEnabled
    self.events.add(make_event([ET.PRE_ENABLE]))
    enabled, active = self.state_machine.update(self.events)

    assert enabled is True
    assert active is False

  def test_enabled_returns_enabled_and_active(self):
    """Enabled state should return enabled=True, active=True."""
    self.state_machine.state = State.enabled
    enabled, active = self.state_machine.update(self.events)

    assert enabled is True
    assert active is True

  def test_soft_disabling_returns_enabled_and_active(self):
    """softDisabling should return enabled=True, active=True."""
    self.state_machine.state = State.softDisabling
    self.state_machine.soft_disable_timer = 100
    self.events.add(make_event([ET.SOFT_DISABLE]))
    enabled, active = self.state_machine.update(self.events)

    assert enabled is True
    assert active is True

  def test_overriding_returns_enabled_and_active(self):
    """Overriding state should return enabled=True, active=True."""
    self.state_machine.state = State.overriding
    self.events.add(make_event([ET.OVERRIDE_LATERAL]))
    enabled, active = self.state_machine.update(self.events)

    assert enabled is True
    assert active is True


class TestStateMachineDisablePriority:
  """Tests for USER_DISABLE and IMMEDIATE_DISABLE priority."""

  def setup_method(self):
    self.events = Events()
    self.state_machine = StateMachine()

  def test_user_disable_priority_over_soft_disable(self):
    """USER_DISABLE should take priority over SOFT_DISABLE."""
    self.state_machine.state = State.enabled
    self.events.add(make_event([ET.USER_DISABLE, ET.SOFT_DISABLE]))
    self.state_machine.update(self.events)

    assert self.state_machine.state == State.disabled
    assert ET.USER_DISABLE in self.state_machine.current_alert_types

  def test_user_disable_priority_over_override(self):
    """USER_DISABLE should take priority over overrides."""
    self.state_machine.state = State.enabled
    self.events.add(make_event([ET.USER_DISABLE, ET.OVERRIDE_LATERAL]))
    self.state_machine.update(self.events)

    assert self.state_machine.state == State.disabled

  def test_immediate_disable_priority_over_soft_disable(self):
    """IMMEDIATE_DISABLE should take priority over SOFT_DISABLE."""
    self.state_machine.state = State.enabled
    self.events.add(make_event([ET.IMMEDIATE_DISABLE, ET.SOFT_DISABLE]))
    self.state_machine.update(self.events)

    assert self.state_machine.state == State.disabled
    assert ET.IMMEDIATE_DISABLE in self.state_machine.current_alert_types

  def test_user_disable_priority_over_immediate(self):
    """USER_DISABLE should take priority over IMMEDIATE_DISABLE."""
    self.state_machine.state = State.enabled
    self.events.add(make_event([ET.USER_DISABLE, ET.IMMEDIATE_DISABLE]))
    self.state_machine.update(self.events)

    assert self.state_machine.state == State.disabled
    # USER_DISABLE is checked first
    assert ET.USER_DISABLE in self.state_machine.current_alert_types

  def test_disable_works_from_all_non_disabled_states(self):
    """USER_DISABLE should work from any non-disabled state."""
    non_disabled_states = [State.enabled, State.softDisabling, State.preEnabled, State.overriding]
    for state in non_disabled_states:
      self.state_machine.state = state
      self.state_machine.soft_disable_timer = 100
      self.events.add(make_event([ET.USER_DISABLE]))
      self.state_machine.update(self.events)

      assert self.state_machine.state == State.disabled
      self.events.clear()
