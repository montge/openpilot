## ADDED Requirements

### Requirement: Test file targets `selfdrive/monitoring/policy.py` and is named accordingly

The fork-only test file SHALL be located at `selfdrive/monitoring/tests/test_monitoring_policy.py` and SHALL import all driver-monitoring symbols from `openpilot.selfdrive.monitoring.policy`. It SHALL NOT import from the deleted `openpilot.selfdrive.monitoring.helpers` path, even via a shim.

#### Scenario: File path and name reflect the target module
- **GIVEN** a checkout of the fork at the post-merge HEAD
- **WHEN** `ls selfdrive/monitoring/tests/` is run
- **THEN** the file `test_monitoring_policy.py` is present
- **AND** the file `test_monitoring_helpers.py` is absent

#### Scenario: Imports resolve against the live module
- **GIVEN** the test file's import block
- **WHEN** Python loads the module
- **THEN** every imported name resolves under `openpilot.selfdrive.monitoring.policy` (no `…helpers…` path appears)

### Requirement: Surviving symbols are tested with their post-merge API

For each symbol that survived the upstream relocation (`DRIVER_MONITOR_SETTINGS`, `DriverPose`, `DriverBlink`, `DriverMonitoring`, `face_orientation_from_model`), the ported test class SHALL exercise the symbol using its current signature and attribute set (no `device_type` kwarg, no `roll` return from `face_orientation_*`, no references to deleted attributes like `dm.awareness`, `dm.wheelpos`, `dm.active_monitoring_mode`).

#### Scenario: `DRIVER_MONITOR_SETTINGS` constructed with no arguments
- **GIVEN** `TestDriverMonitorSettings` test cases
- **WHEN** they instantiate `DRIVER_MONITOR_SETTINGS`
- **THEN** they pass no positional or keyword arguments
- **AND** they assert on attribute names that exist on the current class (e.g. `_VISION_POLICY_ALERT_3_TIMEOUT`, `_FACE_THRESHOLD`, `_PHONE_THRESH`)

#### Scenario: `DriverPose` is tested without removed attributes
- **GIVEN** a `DriverPose(settings)` instance
- **WHEN** the test asserts on attributes
- **THEN** it asserts only on `pitch`, `yaw`, `pitch_offsetter`, `yaw_offsetter`, `calibrated`, `low_std`, `cfactor_pitch`, `cfactor_yaw`, `steer_yaw_offset`
- **AND** it does NOT assert on `roll`, `pitch_std`, `yaw_std`, `roll_std`, or the misspelled `pitch_offseter`/`yaw_offseter`

#### Scenario: `face_orientation_from_model` returns a 2-tuple
- **GIVEN** the renamed `TestFaceOrientationFromModel` test class
- **WHEN** it calls `face_orientation_from_model(orient_model, pos_model, rpy_calib)`
- **THEN** it unpacks the result as `pitch, yaw = …`, not `roll, pitch, yaw = …`

#### Scenario: `DriverMonitoring` init asserts on current state names
- **GIVEN** `TestDriverMonitoringInit`
- **WHEN** a `DriverMonitoring(rhd_saved=…, always_on=…)` is constructed
- **THEN** the test asserts on at minimum `dm.alert_level == AlertLevel.none`, `dm.active_policy == MonitoringPolicy.vision`, `dm.terminal_alert_cnt == 0`, `dm.face_detected is False`, `dm.driver_distracted is False`
- **AND** the test does NOT reference `dm.awareness`, `dm.awareness_active`, `dm.awareness_passive`, `dm.wheelpos`, `dm.phone`, `dm.active_monitoring_mode`, `dm.threshold_pre`, or `dm.step_change`

### Requirement: Distracted-state coverage uses the dict-keyed surface

The ported tests for distracted-state detection (`TestDriverMonitoringGetDistractedTypes`, `TestDriverMonitoringGetDistractedTypesCalibrated`) SHALL assert against `dm.distracted_types['pose']`, `dm.distracted_types['eye']`, `dm.distracted_types['phone']` rather than against the removed `DistractedType.DISTRACTED_*` bitfield constants. Note the rename `BLINK` → `'eye'`.

#### Scenario: Pose distraction sets the 'pose' key
- **GIVEN** a `DriverMonitoring` instance fed an input that triggers pose-distraction
- **WHEN** the post-update `dm.distracted_types` dict is read
- **THEN** `dm.distracted_types['pose']` is `True`

#### Scenario: Blink distraction sets the 'eye' key (not 'blink')
- **GIVEN** a `DriverMonitoring` instance fed an input with high blink probability
- **WHEN** the post-update `dm.distracted_types` dict is read
- **THEN** `dm.distracted_types['eye']` is `True`
- **AND** the test does not look up `'blink'`

#### Scenario: Phone distraction sets the 'phone' key
- **GIVEN** a `DriverMonitoring` instance fed an input with high phone probability
- **WHEN** the post-update `dm.distracted_types` dict is read
- **THEN** `dm.distracted_types['phone']` is `True`

### Requirement: `get_state_packet` field-presence assertion uses live-schema fields

The ported `TestDriverMonitoringGetStatePacket::test_get_state_packet_contains_expected_fields` SHALL assert presence of fields on the live `DriverMonitoringState` schema. At minimum it SHALL assert top-level fields `alertLevel`, `activePolicy`, and `isRHD`, and at least one populated sub-struct field (e.g. `visionPolicyState.faceDetected`). It SHALL NOT reference `awarenessStatus`.

#### Scenario: Packet exposes top-level live-schema fields
- **GIVEN** a `DriverMonitoring` instance after `get_state_packet()` is called
- **WHEN** the returned message's `driverMonitoringState` is inspected
- **THEN** `alertLevel`, `activePolicy`, and `isRHD` are accessible without `AttributeError`

#### Scenario: Packet exposes at least one sub-struct field
- **GIVEN** the same packet
- **WHEN** `driverMonitoringState.visionPolicyState.faceDetected` is read
- **THEN** the access succeeds (sub-struct is populated, not empty)

#### Scenario: Test does not reference removed fields
- **GIVEN** the body of `test_get_state_packet_contains_expected_fields`
- **WHEN** the file is grepped for `awarenessStatus`
- **THEN** no occurrence is present

### Requirement: Cereal enum imports use named members

The ported test file SHALL import `AlertLevel = log.DriverMonitoringState.AlertLevel` and `MonitoringPolicy = log.DriverMonitoringState.MonitoringPolicy` at module top, and SHALL use named members (e.g. `AlertLevel.none`, `MonitoringPolicy.vision`) for all comparisons. Raw integer ordinals SHALL NOT be used to compare enum values.

#### Scenario: Test references named enum members
- **GIVEN** any assertion comparing `dm.alert_level` or `dm.active_policy`
- **WHEN** the comparison value is examined
- **THEN** it is a named member like `AlertLevel.none` or `MonitoringPolicy.vision`
- **AND** it is not a raw integer

### Requirement: Fixture builders are duplicated, not imported, from `test_monitoring.py`

The ported file SHALL include local copies of the `make_msg` builder and `msg_NO_FACE_DETECTED` / `msg_ATTENTIVE` / `msg_DISTRACTED` (etc.) fixture constants. It SHALL NOT import them from `selfdrive/monitoring/test_monitoring.py`. A short attribution comment above the duplicated block SHALL note the source.

#### Scenario: Fixture import path
- **GIVEN** the test file's imports
- **WHEN** they are scanned for `from openpilot.selfdrive.monitoring.test_monitoring import`
- **THEN** no such line is present

#### Scenario: Attribution comment
- **GIVEN** the duplicated fixture block
- **WHEN** the comment immediately above it is read
- **THEN** it acknowledges the upstream-stock `selfdrive/monitoring/test_monitoring.py` as the source pattern

### Requirement: Coverage of `policy.py` does not regress versus the originals' coverage of `helpers.py`

After the port, line coverage of `selfdrive/monitoring/policy.py` produced by `test_monitoring_policy.py` SHALL be at least as high as the pre-merge line coverage of `selfdrive/monitoring/helpers.py` produced by the originals (or the `codecov.yml` monitoring component target of 85%, whichever is lower).

#### Scenario: Coverage measurement after port
- **GIVEN** the ported test file
- **WHEN** `pytest --cov=openpilot.selfdrive.monitoring.policy --cov-branch selfdrive/monitoring/tests/test_monitoring_policy.py` is run
- **THEN** the reported line coverage is ≥ the pre-merge baseline figure
- **AND** if the pre-merge baseline cannot be retrieved, line coverage is ≥ 85%

## REMOVED Requirements

### Requirement: `DistractedType` bitfield constants

**Reason**: The `DistractedType` class was removed by the upstream merge. The "is X distracted" question is now answered through `policy.DriverMonitoring.distracted_types: defaultdict[str, bool]` and is exhaustively covered by `TestDriverMonitoringGetDistractedTypes` / `TestDriverMonitoringGetDistractedTypesCalibrated`.

**Migration**: Tests asserting on `DistractedType.NOT_DISTRACTED == 0` and bitwise OR combinations are deleted. The named bitfield constants do not exist post-merge; their semantic content is replaced by the dict-keyed surface.

### Requirement: `DriverProb` class behavior

**Reason**: The `DriverProb` class was removed by the upstream merge. The probability-tracking logic is inlined as scalar attributes (`dm.phone_prob`) and an embedded `RunningStatFilter` for offset tracking; there is no top-level class to test.

**Migration**: `TestDriverProb` is deleted entirely. The single original test (`test_driver_prob_initialization`) was asserting attribute presence on a class that no longer exists.

### Requirement: `_reset_awareness` method behavior

**Reason**: The `_reset_awareness` method was removed by the upstream merge. The "awareness" public concept was a 0..1 float that no longer exists; the equivalent state-machine reset behavior is exercised through `_update_events` flows already covered by `TestDriverMonitoringUpdateEvents`.

**Migration**: `TestDriverMonitoringResetAwareness` is deleted. Tests should verify reset behavior indirectly via `dm.alert_level` after a reset-triggering event sequence.

### Requirement: `_set_timers(active_monitoring=…)` method behavior

**Reason**: The `_set_timers` method was removed by the upstream merge. The active/passive monitoring distinction is replaced by `MonitoringPolicy.vision` vs `MonitoringPolicy.wheeltouch`, exercised by `TestDriverMonitoringSetPolicy`.

**Migration**: `TestDriverMonitoringSetTimers` and `TestDriverMonitoringSetTimersEdgeCases` are deleted. Active/passive coverage is folded into the surviving policy tests.
