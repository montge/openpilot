# dm-state-test-coverage Specification

## Purpose
TBD - created by archiving change migrate-dm-tests-new-schema. Update Purpose after archive.
## Requirements
### Requirement: Mock fixtures expose `alertLevel` as the DM state knob

The fork-only mock-SubMaster helpers in `selfdrive/controls/tests/test_controlsd.py` (`_set_driver_monitoring_state` and `_setup_default_sm`) SHALL accept an `alert_level` keyword of type `log.DriverMonitoringState.AlertLevel` (defaulting to `AlertLevel.none`) and SHALL set the published `driverMonitoringState.alertLevel` field accordingly. They SHALL NOT reference the deprecated `awarenessStatus` field.

#### Scenario: Default fixture publishes nominal alert level
- **GIVEN** a test instantiates `_setup_default_sm()` with no arguments
- **WHEN** the fixture publishes `driverMonitoringState`
- **THEN** the published `alertLevel` equals `log.DriverMonitoringState.AlertLevel.none`
- **AND** no `awarenessStatus` write occurs (the field is on `DriverMonitoringStateDEPRECATED`, not the live struct)

#### Scenario: Caller can request the terminal alert level
- **GIVEN** a test calls `_setup_default_sm(alert_level=AlertLevel.three)`
- **WHEN** the fixture publishes `driverMonitoringState`
- **THEN** the published `alertLevel` equals `AlertLevel.three`

#### Scenario: Helper accepts every enumerated AlertLevel without TypeError
- **GIVEN** the helper `_set_driver_monitoring_state`
- **WHEN** it is called once per member of `AlertLevel` (`none`, `one`, `two`, `three`)
- **THEN** each call completes without exception
- **AND** the corresponding published `alertLevel` matches the input

### Requirement: `forceDecel` tests cover terminal-alert and soft-disabling triggers

The test suite SHALL retain three `forceDecel` test cases exercising the rule defined in [selfdrive/controls/controlsd.py:196](selfdrive/controls/controlsd.py:196): one for the terminal DM alert, one for the soft-disabling state-machine state, and one for the nominal case where neither trigger fires. The tests SHALL derive the expected outcome from the rule `(alertLevel == AlertLevel.three) or (selfdriveState.state == softDisabling)` rather than hard-coding boolean expectations.

#### Scenario: Terminal DM alert forces deceleration
- **GIVEN** the mock SubMaster is configured with `alert_level=AlertLevel.three` and `state=State.enabled`
- **WHEN** `controls.state_control()` runs
- **THEN** `forceDecel` evaluates to `True`
- **AND** the test method name advertises the trigger as a terminal DM alert (not "negative awareness")

#### Scenario: Soft-disabling state forces deceleration
- **GIVEN** the mock SubMaster is configured with `alert_level=AlertLevel.none` and `state=State.softDisabling`
- **WHEN** `controls.state_control()` runs
- **THEN** `forceDecel` evaluates to `True`

#### Scenario: Nominal alert level and enabled state does not force deceleration
- **GIVEN** the mock SubMaster is configured with `alert_level=AlertLevel.none` and `state=State.enabled`
- **WHEN** `controls.state_control()` runs
- **THEN** `forceDecel` evaluates to `False`

#### Scenario: forceDecel formula constant tracks `controlsd.py`
- **GIVEN** the test module declares a single `TERMINAL_ALERT = log.DriverMonitoringState.AlertLevel.three` constant used by every forceDecel test
- **WHEN** the value of `TERMINAL_ALERT` is compared against the alert level the production rule in `controlsd.py:196` checks
- **THEN** the two are equal (else the test module diverges from the consumer and must be updated)

### Requirement: Default-state mocks for unrelated tests stay nominal

Test cases that need a populated `driverMonitoringState` only as scaffolding for an unrelated assertion (e.g., NaN/Inf cruise-logic checks at lines `~566`, `~847`, `~1216` of `test_controlsd.py`) SHALL set `alertLevel = AlertLevel.none` and SHALL NOT reference `awarenessStatus`. These sites SHALL NOT introduce nominal-state coverage assertions of their own — they exist to make adjacent tests runnable, not to verify DM behavior.

#### Scenario: Incidental DM-state mock at NaN cruise-logic test site
- **GIVEN** a test whose primary purpose is verifying cruise-logic behavior, not DM behavior
- **WHEN** it sets up its mock `driverMonitoringState`
- **THEN** it sets `alertLevel = AlertLevel.none`
- **AND** it does NOT write `awarenessStatus`
- **AND** it does NOT add new assertions about DM state (single responsibility)

### Requirement: AlertLevel ordinal regression guard

The test suite SHALL include one assertion verifying that `log.DriverMonitoringState.AlertLevel.three` retains ordinal value `3`. This guards the documented hazard in `cereal/log.capnp` ("ordinal must match the name to prevent bugs comparing against the raw ordinal value") and provides early detection if a future upstream merge perturbs the enum order.

#### Scenario: AlertLevel.three has ordinal 3
- **GIVEN** the cereal log schema as imported into the test module
- **WHEN** `int(log.DriverMonitoringState.AlertLevel.three)` is evaluated
- **THEN** the value is `3`

### Requirement: Coverage of `forceDecel` decision in `controlsd.py` does not regress

Line coverage of the `forceDecel` computation in [selfdrive/controls/controlsd.py:196](selfdrive/controls/controlsd.py:196) (or its post-merge equivalent line) SHALL remain at 100% after this migration. The redesigned `TestControlsForceDecel` class SHALL exercise both branches of the `or` (the `alertLevel` branch and the `softDisabling` branch) and the falsy fall-through.

#### Scenario: forceDecel line is covered by all three branches
- **GIVEN** the migrated `TestControlsForceDecel` test class
- **WHEN** the test class is run with `pytest --cov=openpilot.selfdrive.controls.controlsd`
- **THEN** the `forceDecel` assignment line shows as covered
- **AND** branch coverage (if measured) shows both the True-via-alertLevel and True-via-softDisabling outcomes exercised

### Requirement: No production code modifications

The change SHALL modify only test files. Production code under `cereal/`, `selfdrive/controls/` (excluding the `tests/` subdirectory), `selfdrive/monitoring/`, and `selfdrive/selfdrived/` SHALL NOT be touched.

#### Scenario: Diff contains only test-directory changes
- **GIVEN** the git diff produced by this change against `develop`
- **WHEN** the diff is filtered to file paths
- **THEN** every changed path is `selfdrive/controls/tests/test_controlsd.py`

