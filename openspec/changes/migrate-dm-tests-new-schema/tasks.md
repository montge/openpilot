## 1. Pre-flight verification (DONE — see notes)

- [x] 1.1 Confirm the upstream merge from `claude/festive-lovelace-8096bd` is on disk somewhere accessible. **Done:** worktree exists at `/Users/e/Development/GitHub/openpilot/.claude/worktrees/festive-lovelace-8096bd` at HEAD `6dbff92fb` (descendant of `e8b2e3291`). Implementation runs against that worktree; merge to `develop` may follow later.
- [x] 1.2 Schema parity: `cereal/log.capnp` in the merge worktree shows `DriverMonitoringState` with `alertLevel @5 :AlertLevel` and `activePolicy @6 :MonitoringPolicy`; `awarenessStatus @3` is on `DriverMonitoringStateDEPRECATED`. **Confirmed.**
- [x] 1.3 `forceDecel` rule: [selfdrive/controls/controlsd.py:196](selfdrive/controls/controlsd.py:196) in the merge worktree uses `(driverMonitoringState.alertLevel == log.DriverMonitoringState.AlertLevel.three) or (selfdriveState.state == State.softDisabling)`. **Confirmed.**
- [x] 1.4 Stray `awarenessStatus` refs: only the two upstream-owned writes in [selfdrive/test/process_replay/migration.py:510,519](selfdrive/test/process_replay/migration.py:510) (legitimate — they read `old.awarenessStatus` from `DriverMonitoringStateDEPRECATED` to translate old logs forward). No fork-side stragglers beyond `test_controlsd.py`. **Confirmed.**
- [x] 1.5 Scope check on `test_monitoring_helpers.py`: the file is comprehensively broken post-merge (imports a deleted module, several missing symbols). **Split out** to sibling change `port-test-monitoring-helpers-to-policy`; not in scope here.

## 2. Migrate `test_controlsd.py` — module-level setup

- [ ] 2.1 Add module-level alias `AlertLevel = log.DriverMonitoringState.AlertLevel` near the existing `State = log.SelfdriveState.OpenpilotState` alias on line 20. (The file already imports `log` from `cereal`.)
- [ ] 2.2 Inside `TestControlsForceDecel`, declare class-level constant `TERMINAL_ALERT = log.DriverMonitoringState.AlertLevel.three`.

## 3. Migrate `test_controlsd.py` — fixture and helper signatures

- [ ] 3.1 Rewrite `_set_driver_monitoring_state` (line ~269): change signature to `def _set_driver_monitoring_state(self, alert_level=None):`, default `alert_level` to `AlertLevel.none` inside the body, and replace `dms.awarenessStatus = awarenessStatus` with `dms.alertLevel = alert_level`.
- [ ] 3.2 Rewrite `_setup_default_sm` (line ~1299): change signature to `def _setup_default_sm(self, alert_level=None, state=State.enabled):`, default `alert_level` to `AlertLevel.none` inside the body, and replace `msg.driverMonitoringState.awarenessStatus = awarenessStatus` (line ~1334) with `msg.driverMonitoringState.alertLevel = alert_level`.
- [ ] 3.3 Run `pytest selfdrive/controls/tests/test_controlsd.py -x --co -q` (collect-only). Confirm no collection errors from the signature changes.

## 4. Migrate `test_controlsd.py` — forceDecel test cases

- [ ] 4.1 Rename `test_force_decel_on_negative_awareness` (line ~1342) to `test_force_decel_on_terminal_dm_alert`. Change call site to `self._setup_default_sm(alert_level=AlertLevel.three)`. Replace inline force_decel recompute with: `force_decel = bool((self.mock_sm['driverMonitoringState'].alertLevel == self.TERMINAL_ALERT) or (self.mock_sm['selfdriveState'].state == State.softDisabling))`. Update docstring.
- [ ] 4.2 Update `test_force_decel_on_soft_disabling` (line ~1353): the existing call already omits the alert kwarg so the default `AlertLevel.none` applies — no call-site change. Update inline `force_decel` recompute to use `alertLevel == self.TERMINAL_ALERT`.
- [ ] 4.3 Update `test_no_force_decel_under_normal_conditions` (line ~1363): change `self._setup_default_sm(awarenessStatus=1.0, state=State.enabled)` to `self._setup_default_sm(alert_level=AlertLevel.none, state=State.enabled)`. Update inline `force_decel` recompute as in 4.1.
- [ ] 4.4 Run `pytest selfdrive/controls/tests/test_controlsd.py::TestControlsForceDecel -v`. Confirm all three cases pass and that the True/False outcomes match expected polarity.

## 5. Migrate `test_controlsd.py` — incidental DM-state mocks

- [ ] 5.1 At line ~566, replace `msg.driverMonitoringState.awarenessStatus = 1.0` with `msg.driverMonitoringState.alertLevel = AlertLevel.none`.
- [ ] 5.2 At line ~847, same replacement as 5.1.
- [ ] 5.3 At line ~1216, same replacement as 5.1.
- [ ] 5.4 Run `pytest selfdrive/controls/tests/test_controlsd.py -x` and confirm no collateral breakage.

## 6. Add AlertLevel ordinal regression guard

- [ ] 6.1 Add a top-level test (outside any class) `test_alert_level_three_has_ordinal_three` asserting `int(log.DriverMonitoringState.AlertLevel.three) == 3`. Docstring references the `cereal/log.capnp` comment that documents the hazard.
- [ ] 6.2 Run the new test alone: `pytest selfdrive/controls/tests/test_controlsd.py::test_alert_level_three_has_ordinal_three -v`.

## 7. Coverage and full-suite verification

- [ ] 7.1 Run the migrated file: `pytest selfdrive/controls/tests/test_controlsd.py -v`. All previously-passing tests must still pass.
- [ ] 7.2 Run with coverage: `pytest --cov=openpilot.selfdrive.controls.controlsd --cov-report=term-missing selfdrive/controls/tests/test_controlsd.py`. Confirm the `forceDecel` line in `controlsd.py` shows as covered.
- [ ] 7.3 Run `scripts/lint/lint.sh ruff` over the modified file. Fix any line-length or style issues.

## 8. OpenSpec validation and PR

- [ ] 8.1 Run `openspec validate migrate-dm-tests-new-schema` and confirm clean.
- [ ] 8.2 Commit with a message summarizing the migration; reference commit `4af466c7f` (origin of the tests) and the upstream merge commit on the merge branch.
- [ ] 8.3 Push to `origin` (`montge/openpilot`); confirm CI green. Do NOT push to `upstream`.

## 9. Archive the change

- [ ] 9.1 After merge, run `openspec archive migrate-dm-tests-new-schema` to fold `dm-state-test-coverage` into `openspec/specs/` and move this change folder under `openspec/changes/archive/`.
