## Context

The fork-only test file `selfdrive/monitoring/tests/test_monitoring_helpers.py` was built in commit `4af466c7f` against a `selfdrive/monitoring/helpers.py` module that no longer exists post-merge. Triage on the merge worktree at `/Users/e/Development/GitHub/openpilot/.claude/worktrees/festive-lovelace-8096bd` reveals the breakage runs deeper than the import block:

**Symbol disposition:**

| Old symbol (imported from `helpers`) | New location | Disposition |
|---|---|---|
| `DRIVER_MONITOR_SETTINGS` | [policy.py:25](selfdrive/monitoring/policy.py:25) | Survived — but `__init__()` now takes no args (old test passed `device_type='tici'`). |
| `DriverPose` | [policy.py:83](selfdrive/monitoring/policy.py:83) | Survived — `__init__(settings)` unchanged shape; old test asserted `roll`, `yaw_std`, `pitch_std`, `roll_std` (none exist on the new class), `pitch_offseter` (typo — new attr is `pitch_offsetter`). |
| `DriverBlink` | [policy.py:97](selfdrive/monitoring/policy.py:97) | Survived — `__init__()` unchanged, has `left` and `right` attrs only. |
| `DriverMonitoring` | [policy.py:123](selfdrive/monitoring/policy.py:123) | Survived but **major API changes**: `__init__(rhd_saved=False, settings=None, always_on=False)` is similar, but no more `dm.awareness`, `dm.awareness_active`, `dm.awareness_passive`, `dm.wheelpos`, `dm.phone`, `dm.active_monitoring_mode`, `dm.threshold_pre`, no more `_reset_awareness()` or `_set_timers()` methods. State now lives in `dm.alert_level`, `dm.active_policy`, `dm.distracted_types` (dict), `dm.terminal_alert_cnt`, `dm.wheelpos_offsetter`, `dm.phone_prob` etc. |
| `face_orientation_from_net` | [policy.py:107](selfdrive/monitoring/policy.py:107) (renamed `face_orientation_from_model`) | Renamed + parameter rename (`*_desc` → `*_model`); also returns `(pitch, yaw)` now, NOT `(roll, pitch, yaw)` as the old function did. |
| `DistractedType` (class with INT bitfield constants `NOT_DISTRACTED`, `DISTRACTED_POSE`, `DISTRACTED_BLINK`, `DISTRACTED_PHONE`) | **Removed.** | Concept lives as `policy.DriverMonitoring.distracted_types: defaultdict[str, bool]` keyed on `'pose'`/`'eye'`/`'phone'` (note: `BLINK` → `eye`). Cereal struct `DriverMonitoringState.VisionPolicyState.DistractedTypes` mirrors the new keys. |
| `DriverProb` | **Removed entirely.** | No replacement class. The probability logic is now inlined as `dm.phone_prob: float`. |

**Test-class disposition (75 tests across 21 classes):**

| Class | Disposition |
|---|---|
| `TestDriverMonitorSettings` (3 tests) | Port — drop `device_type` kwarg; assert attribute presence with current names. |
| `TestDistractedType` (4 tests) | **Delete** — class is gone; assertions on bitfield constants no longer apply. The "is value X distracted" question is now answered by reading `dm.distracted_types['pose']` etc., already covered by `TestDriverMonitoringGetDistractedTypes`. |
| `TestDriverPose` (2 tests) | Port — fix attribute names (`pitch_offseter` → `pitch_offsetter`); drop `roll`, `*_std` assertions (don't exist). |
| `TestDriverProb` (1 test) | **Delete** — class is gone. |
| `TestDriverBlink` (1 test) | Port — straight rename only. |
| `TestFaceOrientationFromNet` (3 tests) | Port — rename to `TestFaceOrientationFromModel`, drop `roll` from return-value unpacking, update call signature. |
| `TestDriverMonitoringInit` (3 tests) | Port — replace assertions on removed attrs (`dm.wheelpos` → `dm.wheelpos_offsetter`, `dm.phone` → `dm.phone_prob`, drop `dm.awareness`); add assertions on new state (`dm.alert_level == AlertLevel.none`, `dm.active_policy == MonitoringPolicy.vision`). |
| `TestDriverMonitoringResetAwareness` (1 test) | **Delete** — `_reset_awareness()` no longer exists; awareness is no longer a public concept. The equivalent reset behavior is exercised through `_update_events` flows already covered by other classes. |
| `TestDriverMonitoringSetTimers` (2 tests) | **Delete** — `_set_timers()` no longer exists. The active/passive distinction is replaced by `MonitoringPolicy.vision` vs `MonitoringPolicy.wheeltouch`, exercised by `TestDriverMonitoringSetPolicy` (which survives). |
| `TestDriverMonitoringSetPolicy` | Port — verify against `dm.active_policy`. |
| `TestDriverMonitoringGetDistractedTypes` (3 tests) | Port — assert against `dm.distracted_types['pose']`/`['eye']`/`['phone']` instead of bitfield. Note `'blink'` → `'eye'` rename. |
| `TestDriverMonitoringGetStatePacket` | Port — replace the `state.awarenessStatus is not None` assertion with the new field set (`alertLevel`, `activePolicy`, `isRHD`); add `visionPolicyState.faceDetected` and `visionPolicyState.distractedTypes.{pose,eye,phone}` assertions per [policy.py:354-376](selfdrive/monitoring/policy.py:354). |
| `TestDriverMonitoringUpdateEvents` (5 tests) | Port — these test event-driven state transitions; verify each test still hits a real method on the new class (`_update_events` exists). |
| `TestDriverMonitoringSetTimersEdgeCases` | **Delete** — references gone method. |
| `TestDriverMonitoringGetDistractedTypesCalibrated` | Port with the same renames as `TestDriverMonitoringGetDistractedTypes`. |
| `TestDriverMonitoringUpdateStates` | Port — `_update_states` exists; verify field accesses still resolve. |
| `TestDriverMonitoringUpdateEventsEdgeCases` | Port — same caveats. |
| `TestDriverMonitoringDcamUncertainReset` | Port — `is_model_uncertain` and the dcam-uncertain counters still exist on the new class. |
| `TestDriverMonitoringTerminalAlerts` | Port — `terminal_alert_cnt` and `terminal_time` still exist; verify `_MAX_TERMINAL_ALERTS` settings name unchanged. |
| `TestDriverMonitoringRunStep` | Port — verify the step entry point still exists and has the same signature. |

Estimated outcome: ~10 test classes deleted (about 12-15 tests), ~11 ported (about 60 tests). Net ~60 tests preserved, vs. naive expectation of "1146 line file → 1146 line file."

## Goals / Non-Goals

**Goals:**
- Restore a runnable, fork-only test file that exercises `selfdrive/monitoring/policy.py` at coverage no lower than the originals exercised of the deleted `helpers.py`.
- Lean on `selfdrive/monitoring/test_monitoring.py` (upstream-stock, currently passing) for fixture patterns rather than reinventing them — its `make_msg(face_detected, distracted, model_uncertain)` builder and `_run_seq` runner are battle-tested and target the same module.
- Make every deletion explicit: the spec must enumerate which test classes are dropped and why, so a future reviewer can see the trade.
- Rename the file to `test_monitoring_policy.py` so its name reflects what it tests; keep it under `selfdrive/monitoring/tests/`.

**Non-Goals:**
- Porting tests for symbols that no longer exist (`DriverProb`, `DistractedType` constants, `_reset_awareness`, `_set_timers`). The behavior they were probing has either been refactored away or is now exercised through different surfaces already covered.
- Adding *new* coverage beyond what the originals had. The point is to preserve coverage intent, not extend it.
- Modifying any production code. Test-only.
- Touching the upstream-stock `test_monitoring.py` (no `tests/` subdir prefix). That is upstream-owned.

## Decisions

### Decision 1: Rename rather than overwrite

Move/rename `selfdrive/monitoring/tests/test_monitoring_helpers.py` → `selfdrive/monitoring/tests/test_monitoring_policy.py` in a single `git mv` step *before* editing. Rationale: preserves `git blame` continuity for the surviving tests, and the new name documents what the file actually targets after the port. CI configs (`codecov.yml`, `.github/workflows/`) need a grep — none should reference this specific filename today, but the migration plan re-checks.

### Decision 2: Accept ~10 test-class deletions, document each in REMOVED Requirements

Tests for `DriverProb`, `DistractedType` constants, `_reset_awareness`, and `_set_timers` are not portable — the symbols don't exist and the behavior either moved out of test-able boundaries (e.g. `_reset_awareness` was an implementation detail of internal awareness counting that no longer exists as a top-level concept) or is exhaustively covered elsewhere. Documenting each as a `REMOVED Requirements` entry in the spec keeps the audit trail clear.

**Alternative considered:** Try to find indirect surfaces to keep these tests alive (e.g. assert `dm.distracted_types == defaultdict(bool)` after a synthetic engaged sequence). Rejected — it would re-test what `TestDriverMonitoringGetDistractedTypes` already covers, just less directly.

### Decision 3: Reuse `test_monitoring.py`'s fixture builders, do not re-invent

The upstream stock test file `selfdrive/monitoring/test_monitoring.py` (note: no `tests/` subdir, sibling of `policy.py`) defines `make_msg(face_detected, distracted=False, model_uncertain=False)`, `msg_NO_FACE_DETECTED`, `msg_ATTENTIVE`, etc. These are exactly what the ported tests need. Either:
- (a) Import them: `from openpilot.selfdrive.monitoring.test_monitoring import make_msg, msg_ATTENTIVE, ...` — works but couples the fork test to upstream test internals; if upstream renames a fixture, fork breaks.
- (b) Duplicate them at the top of `test_monitoring_policy.py` — small duplication, fully fork-controlled.

Pick **(b)**. The fixtures are ~30 lines, low-churn upstream, and the duplication insulates the fork from upstream test refactors. The `tasks.md` step calls out copying them verbatim with attribution comment.

### Decision 4: Use cereal enum types, not raw integers, for new assertions

When asserting `alert_level == AlertLevel.none`, `active_policy == MonitoringPolicy.vision`, etc., reference the cereal enum imports at module top:
```python
AlertLevel = log.DriverMonitoringState.AlertLevel
MonitoringPolicy = log.DriverMonitoringState.MonitoringPolicy
```
Rationale: same reasoning as the sibling `migrate-dm-tests-new-schema` change — the schema's own comment warns against raw-ordinal comparison. Use the named enum members.

### Decision 5: Sub-struct field-presence assertions in `TestDriverMonitoringGetStatePacket`

Pre-flight on the merge worktree confirmed that [policy.py:354-376](selfdrive/monitoring/policy.py:354) populates these fields unconditionally in `get_state_packet()`:
- top-level: `lockout`, `alertLevel`, `activePolicy`, `isRHD`
- `visionPolicyState`: `awarenessStep`, `isDistracted`, `distractedTypes.pose`, `distractedTypes.eye`, `distractedTypes.phone`, `faceDetected`, `pose.*`

The ported `test_get_state_packet_contains_expected_fields` should assert presence on the top-level group AND at least one sub-struct field (`visionPolicyState.faceDetected`) so a regression that empties `visionPolicyState` would still trip a test.

**Alternative considered:** Assert every populated field. Rejected — over-specifies; a future field rename in the sub-struct would cause unrelated tests to flake.

### Decision 6: Don't import from the deleted `helpers` path even if a backward-compat shim is later added

If a fork maintainer ever adds a re-export shim like `selfdrive/monitoring/helpers.py` that does `from openpilot.selfdrive.monitoring.policy import *`, the ported test should still import from `policy` directly. Rationale: the test is documenting the module under test; using a shim hides the real surface and makes future deletions of the shim risky.

## Risks / Trade-offs

- **Risk:** Coverage regression. The originals had 75 tests across 21 classes; this plan keeps ~60 tests across ~11 classes. Even if line coverage is preserved, scenario coverage may not be.
  - **Mitigation:** Run `pytest --cov=openpilot.selfdrive.monitoring.policy --cov-branch` before/after and compare. Acceptance criterion: line coverage of `policy.py` after the port ≥ line coverage of `helpers.py` produced by the originals before the merge (need to retrieve the pre-merge coverage figure from the last green CI run on the original commit `4af466c7f`'s descendant).

- **Risk:** Triage missed a symbol that's actually still reachable through a less obvious path.
  - **Mitigation:** Pre-flight in `tasks.md` re-greps the merge worktree at HEAD and re-runs `pytest --collect-only` on the in-progress port; collection errors flag missing symbols before assertions fire.

- **Risk:** Upstream renames a fixture in `test_monitoring.py` after the duplication, causing fork drift.
  - **Mitigation:** Decision 3 picked (b) duplicate-don't-import precisely to insulate. Trade-off accepted: ~30 lines of duplicated fixture code.

- **Trade-off:** Rebuilt-as-rename instead of in-place edit means `git diff` between the old file and the new one is huge — every reviewer needs to read the spec's per-class disposition table to evaluate whether each deletion is right. Worth it for the resulting clarity.

- **Risk:** `face_orientation_from_model` returns 2 values (`pitch, yaw`), the old `face_orientation_from_net` returned 3 (`roll, pitch, yaw`). The two old tests asserting on `roll` cannot be ported as-is. The roll-doesn't-change-with-calibration test in particular is asserting a property of `_net` that may or may not hold for `_model`.
  - **Mitigation:** Drop `test_face_orientation_roll_unaffected_by_calib` (no `roll` to check) and reframe `test_face_orientation_at_center` and `test_face_orientation_applies_calibration` against the 2-tuple return.

## Migration Plan

1. **Block until merge.** Implementation requires the merge branch's `policy.py` on disk. Run from the merge worktree `/Users/e/Development/GitHub/openpilot/.claude/worktrees/festive-lovelace-8096bd` OR wait until the merge lands on `develop`.
2. **Pre-flight verification (tasks 1.x).** Re-grep for any straggler `helpers` imports and confirm the symbol disposition table in this design still holds against the worktree's HEAD.
3. **Rename file first.** `git mv selfdrive/monitoring/tests/test_monitoring_helpers.py selfdrive/monitoring/tests/test_monitoring_policy.py` — single commit, no content changes. This preserves `git blame` for the surviving tests across the rewrite.
4. **Edit imports + fixtures.** Replace import block, drop the three `mocker.patch('…helpers.Params')` references, copy `make_msg`/`msg_*` fixtures from `test_monitoring.py`.
5. **Walk class by class** in the order of the disposition table: port survivors, delete the others. Run `pytest selfdrive/monitoring/tests/test_monitoring_policy.py --collect-only -q` after each class to catch collection errors early.
6. **Run the full file:** `pytest selfdrive/monitoring/tests/test_monitoring_policy.py -v`. All collected tests must pass.
7. **Coverage check:** `pytest --cov=openpilot.selfdrive.monitoring.policy --cov-branch selfdrive/monitoring/tests/test_monitoring_policy.py`. Compare against pre-merge baseline.
8. **Lint:** `scripts/lint/lint.sh ruff selfdrive/monitoring/tests/test_monitoring_policy.py`.
9. **PR to develop.** Per [CLAUDE.md](CLAUDE.md), origin only; never upstream.
10. **Rollback:** Pure test-only — `git revert` is sufficient. The deleted `test_monitoring_helpers.py` was already broken at HEAD, so rollback restores the broken state, not a working state. That's acceptable: the merge breaks it, this change fixes it; reverting this change just re-breaks.

## Open Questions

- **Q1:** Does the upstream-stock `selfdrive/monitoring/test_monitoring.py` already cover the same scenarios as the porting candidates? If yes, we may be re-duplicating coverage. *Resolved during pre-flight by reading both files side by side and recording overlap in `tasks.md`.*

- **Q2:** Is the file rename (`test_monitoring_helpers.py` → `test_monitoring_policy.py`) referenced anywhere in CI configs or codecov targets? *Resolved by `grep -r "test_monitoring_helpers" .github/ codecov.yml selfdrive/monitoring/` during pre-flight.*

- **Q3:** What is the pre-merge coverage figure for `selfdrive/monitoring/helpers.py` produced by the originals? Needed to set the post-port acceptance threshold.
  - Source: last green CI run on a commit that contained both `4af466c7f` and `helpers.py`.
  - If no figure is recoverable, fall back to the global `codecov.yml` monitoring component target (85%) as the bar.
