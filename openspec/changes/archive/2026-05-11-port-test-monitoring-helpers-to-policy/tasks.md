## 1. Pre-flight verification

- [x] 1.1 Confirm working from the merge worktree at `/Users/e/Development/GitHub/openpilot/.claude/worktrees/festive-lovelace-8096bd` (or `develop` after the merge has landed). All paths below are relative to that worktree.
- [x] 1.2 Verify the symbol-disposition table in [design.md](openspec/changes/port-test-monitoring-helpers-to-policy/design.md) still holds against current HEAD: run `grep -n "^class \|^def " selfdrive/monitoring/policy.py` and confirm `DRIVER_MONITOR_SETTINGS`, `DriverPose`, `DriverBlink`, `DriverMonitoring`, `face_orientation_from_model` are present, and `DriverProb`, `DistractedType` (top-level class), `_reset_awareness`, `_set_timers` are absent.
- [x] 1.3 Resolve Q1: read `selfdrive/monitoring/test_monitoring.py` end-to-end and record (in the PR description) which scenarios the upstream-stock test already covers — to avoid duplicating coverage in the fork test.
- [x] 1.4 Resolve Q2: `grep -rn "test_monitoring_helpers" .github/ codecov.yml selfdrive/monitoring/` and record any matches; if any reference the file path, update them in lockstep with the rename.
- [x] 1.5 Resolve Q3: try to retrieve pre-merge line coverage of `selfdrive/monitoring/helpers.py` from a green CI run on a commit containing both `4af466c7f` and `helpers.py`. If unavailable, record the fallback target (85% per `codecov.yml`) in the PR description.
- [x] 1.6 Verify the merge worktree's `.venv` has `parameterized` and pytest dependencies installed. If not, install via project-approved means and record in the PR description.

## 2. File rename (single commit, no content changes)

- [x] 2.1 `git mv selfdrive/monitoring/tests/test_monitoring_helpers.py selfdrive/monitoring/tests/test_monitoring_policy.py`
- [x] 2.2 Commit as `chore: rename test_monitoring_helpers.py to test_monitoring_policy.py` with a body explaining the rename precedes the content port (preserves git blame).

## 3. Imports and module-top scaffolding

- [x] 3.1 Replace the import block at the top of `test_monitoring_policy.py`:
  - Remove every reference to `openpilot.selfdrive.monitoring.helpers`.
  - Add `from cereal import log` and `import cereal.messaging as messaging`.
  - Add `from openpilot.selfdrive.monitoring.policy import (DRIVER_MONITOR_SETTINGS, DriverPose, DriverBlink, DriverMonitoring, face_orientation_from_model)`.
  - Add aliases: `AlertLevel = log.DriverMonitoringState.AlertLevel`, `MonitoringPolicy = log.DriverMonitoringState.MonitoringPolicy`, `EventName = log.OnroadEvent.EventName`.
  - Update the module docstring to reference `policy.py`, not `helpers.py`.
- [x] 3.2 Copy the `make_msg` builder and `msg_NO_FACE_DETECTED`/`msg_ATTENTIVE`/`msg_DISTRACTED`/`msg_ATTENTIVE_UNCERTAIN`/`msg_DISTRACTED_UNCERTAIN` fixture constants from `selfdrive/monitoring/test_monitoring.py` (no `tests/` subdir) verbatim, with a 2-line attribution comment above the block: `# Fixture builders mirrored from upstream-stock selfdrive/monitoring/test_monitoring.py # Duplicated rather than imported to insulate fork tests from upstream test renames.`
- [x] 3.3 `pytest --collect-only -q selfdrive/monitoring/tests/test_monitoring_policy.py` — expect collection failures only from the test bodies that still reference removed symbols. Module-level scaffolding should not error.

## 4. Delete classes for removed symbols

- [x] 4.1 Delete `class TestDistractedType:` (4 tests) in entirety. Add a 2-line comment block at the deletion point: `# TestDistractedType deleted: DistractedType class removed in upstream merge. # Distracted-state coverage now via TestDriverMonitoringGetDistractedTypes (see dict-keyed surface).`
- [x] 4.2 Delete `class TestDriverProb:` (1 test) in entirety with a similar comment.
- [x] 4.3 Delete `class TestDriverMonitoringResetAwareness:` (1 test) in entirety. Comment: `# TestDriverMonitoringResetAwareness deleted: _reset_awareness method removed; reset behavior now indirect via _update_events flows.`
- [x] 4.4 Delete `class TestDriverMonitoringSetTimers:` and `class TestDriverMonitoringSetTimersEdgeCases:` together. Comment: `# TestDriverMonitoringSetTimers* deleted: _set_timers method removed; active/passive replaced by MonitoringPolicy enum, covered by TestDriverMonitoringSetPolicy.`
- [x] 4.5 `pytest --collect-only -q selfdrive/monitoring/tests/test_monitoring_policy.py` — expect strictly fewer collection errors than after task 3.3.

## 5. Port surviving classes — atomic symbols

- [x] 5.1 `TestDriverMonitorSettings` (3 tests): remove `device_type='tici'` argument from every constructor call. Update the `expected_attrs` list to match the actual attribute set on `policy.DRIVER_MONITOR_SETTINGS` (use `dir()` to confirm). Drop assertions for any attribute that no longer exists; do not invent new ones.
- [x] 5.2 `TestDriverPose` (2 tests): remove `device_type='tici'` from the `setup_method`. In assertions, drop `pose.roll`, `pose.pitch_std`, `pose.yaw_std`, `pose.roll_std`. Rename `pose.pitch_offseter` → `pose.pitch_offsetter` and `pose.yaw_offseter` → `pose.yaw_offsetter`. Add assertions on the surviving new attributes (`cfactor_pitch`, `cfactor_yaw`, `steer_yaw_offset`).
- [x] 5.3 `TestDriverBlink` (1 test): straight rename of imports only — no body changes needed.
- [x] 5.4 `TestFaceOrientationFromNet` → rename class to `TestFaceOrientationFromModel`. Update all 3 call sites to `face_orientation_from_model`. Change all unpacking from `roll, pitch, yaw = …` to `pitch, yaw = …`. Drop `test_face_orientation_roll_unaffected_by_calib` entirely (no roll to assert on); add a 1-line comment `# Removed: roll is no longer returned by face_orientation_from_model` at the deletion point.
- [x] 5.5 Run `pytest selfdrive/monitoring/tests/test_monitoring_policy.py::TestDriverMonitorSettings selfdrive/monitoring/tests/test_monitoring_policy.py::TestDriverPose selfdrive/monitoring/tests/test_monitoring_policy.py::TestDriverBlink selfdrive/monitoring/tests/test_monitoring_policy.py::TestFaceOrientationFromModel -v`. All collected tests must pass.

## 6. Port surviving classes — `DriverMonitoring` init and direct-API tests

- [x] 6.1 `TestDriverMonitoringInit` (3 tests): remove every `mocker.patch('openpilot.selfdrive.monitoring.helpers.Params')` (the patch target no longer exists; if Params is needed, patch `openpilot.selfdrive.monitoring.policy.Params` instead). Replace assertions on removed attributes per the rules in [spec.md "DriverMonitoring init asserts on current state names"](openspec/changes/port-test-monitoring-helpers-to-policy/specs/monitoring-policy-test-coverage/spec.md). At minimum assert `dm.alert_level == AlertLevel.none`, `dm.active_policy == MonitoringPolicy.vision`, `dm.terminal_alert_cnt == 0`, `dm.face_detected is False`, `dm.driver_distracted is False`. Verify `rhd_saved` and `always_on` kwargs still work.
- [x] 6.2 `TestDriverMonitoringSetPolicy`: confirm tests assert against `dm.active_policy == MonitoringPolicy.vision` / `MonitoringPolicy.wheeltouch`. If they assert against integer constants or removed attributes, rewrite.
- [x] 6.3 `TestDriverMonitoringGetStatePacket`: in the `test_get_state_packet_contains_expected_fields` method (originally line ~340 of the broken file), replace `assert state.awarenessStatus is not None` with: `assert state.alertLevel is not None`, `assert state.activePolicy is not None`, `assert state.isRHD is not None`, `assert state.visionPolicyState.faceDetected is not None`. Drop any assertion on `state.faceDetected` or `state.isDistracted` at the top level (those moved into the sub-struct).

## 7. Port surviving classes — distraction detection

- [x] 7.1 `TestDriverMonitoringGetDistractedTypes` (3 tests): replace every `DistractedType.DISTRACTED_*` reference with the equivalent dict-key check. `DISTRACTED_POSE → dm.distracted_types['pose']`, `DISTRACTED_BLINK → dm.distracted_types['eye']` (note rename), `DISTRACTED_PHONE → dm.distracted_types['phone']`. Each test now asserts `assert dm.distracted_types['<key>'] is True` after the trigger sequence.
- [x] 7.2 `TestDriverMonitoringGetDistractedTypesCalibrated`: same renames as 7.1 plus any calibration-state plumbing that may need adjusting against the new `dm.pose.calibrated` flag.

## 8. Port surviving classes — sequence-driven tests

- [x] 8.1 `TestDriverMonitoringUpdateEvents` (5 tests): verify each test still hits a real method. The new class exposes `_update_events(driver_engaged, op_engaged, standstill, car_speed)` — confirm signature via `inspect.signature` on the running module before editing. Update any kwarg names that drifted.
- [x] 8.2 `TestDriverMonitoringUpdateStates`: same review as 8.1 against `_update_states(driver_data, cal_rpy, car_speed, engaged, standstill)` (per [test_monitoring.py:51](selfdrive/monitoring/test_monitoring.py:51) in the merge worktree).
- [x] 8.3 `TestDriverMonitoringUpdateEventsEdgeCases`: same review as 8.1.
- [x] 8.4 `TestDriverMonitoringDcamUncertainReset`: verify `dm.is_model_uncertain`, `_DCAM_UNCERTAIN_ALERT_COUNT`, `_DCAM_UNCERTAIN_RESET_COUNT` all still exist on the new class/settings. Adjust assertions if any of those moved.
- [x] 8.5 `TestDriverMonitoringTerminalAlerts`: verify `dm.terminal_alert_cnt`, `dm.terminal_time`, `_MAX_TERMINAL_ALERTS`, `_MAX_TERMINAL_DURATION` all still exist. Update if drifted.
- [x] 8.6 `TestDriverMonitoringRunStep`: verify the entry point's signature; if upstream renamed `run_step` to something else, rename the test class accordingly.

## 9. Full-suite verification

- [x] 9.1 `pytest selfdrive/monitoring/tests/test_monitoring_policy.py -v` — every test in the renamed file must collect AND pass.
- [x] 9.2 `pytest --cov=openpilot.selfdrive.monitoring.policy --cov-branch --cov-report=term-missing selfdrive/monitoring/tests/test_monitoring_policy.py` — record line and branch coverage. Compare against the pre-merge baseline from task 1.5; coverage must not regress.
- [x] 9.3 Run `pytest selfdrive/monitoring/test_monitoring.py -v` (upstream-stock test, no `tests/` subdir) and confirm it still passes — sanity check that the duplicated fixtures haven't drifted from their source.
- [x] 9.4 `scripts/lint/lint.sh ruff selfdrive/monitoring/tests/test_monitoring_policy.py` — fix any line-length or style issues.
- [x] 9.5 `scripts/lint/lint.sh` (full) — confirm no new failures attributable to this change.

## 10. OpenSpec validation and PR

- [x] 10.1 `openspec validate port-test-monitoring-helpers-to-policy` — must report clean.
- [x] 10.2 Stage and commit. Suggested commit message: `test(monitoring): port test_monitoring_helpers to policy.py`. Body should reference commit `4af466c7f` (origin) and the upstream merge that prompted the port; list the deleted test classes with rationale (mirror the REMOVED Requirements section of the spec).
- [x] 10.3 Push to `origin` only. Open a PR against `develop`. Per [CLAUDE.md](CLAUDE.md), wait for explicit user approval before push.
- [x] 10.4 Confirm CI passes — particularly the `codecov` patch-coverage gate. If patch coverage drops below the global 80% gate, investigate which lines of `policy.py` lost coverage and decide between re-porting a previously-deleted test or accepting the drop with a recorded rationale.

## 11. Archive the change

- [x] 11.1 After PR merges, run `openspec archive port-test-monitoring-helpers-to-policy` to fold `monitoring-policy-test-coverage` into `openspec/specs/` and move this change folder to `openspec/changes/archive/`.
