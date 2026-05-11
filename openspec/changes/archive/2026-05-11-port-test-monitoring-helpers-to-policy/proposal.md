## Why

The fork-only test file `selfdrive/monitoring/tests/test_monitoring_helpers.py` (1,146 lines, 75 tests, added in commit `4af466c7f`) is comprehensively broken on the post-merge branch `claude/festive-lovelace-8096bd` (HEAD `6dbff92fb`, descendant of upstream merge `e8b2e3291`). Upstream relocated and refactored the driver-monitoring code:

- `selfdrive/monitoring/helpers.py` — **deleted**.
- `DRIVER_MONITOR_SETTINGS`, `DriverPose`, `DriverBlink`, `DriverMonitoring`, and `get_state_packet()` — **moved** to `selfdrive/monitoring/policy.py`.
- `DistractedType`, `DriverProb`, `face_orientation_from_net` — appear **removed entirely** (no replacements found by `grep` on the merge worktree).

Because the test file's top-of-module import block does `from openpilot.selfdrive.monitoring.helpers import (DRIVER_MONITOR_SETTINGS, DistractedType, DriverPose, DriverProb, DriverBlink, face_orientation_from_net, DriverMonitoring)`, the entire 1,146-line file fails at import time — collection errors, zero tests runnable. The previously planned one-line fix to `state.awarenessStatus is not None` at line 349 is one symptom among dozens.

This fork carries that comprehensive coverage on purpose (commit message: "test: add comprehensive test coverage for controls module … ~125 new test cases"). Losing 75 tests silently is unacceptable; rewriting them as a unit deserves its own change.

## What Changes

- **Replace the import block** to pull from `openpilot.selfdrive.monitoring.policy`.
- **Triage every test class against the new module layout.** For each class:
  - Tests that target a symbol still present (`DRIVER_MONITOR_SETTINGS`, `DriverPose`, `DriverBlink`, `DriverMonitoring`, `get_state_packet`) — port the imports and any field accesses; update assertions for new field names where applicable.
  - Tests that target a removed symbol (`DistractedType`, `DriverProb`, `face_orientation_from_net`) — investigate whether the behavior moved into a method/attribute on `policy.DriverMonitoring`, or was deleted along with the symbol. If deleted, remove the test class with a `## REMOVED Requirements` entry stating why. If absorbed, port to test the new path.
- **Replace the `awarenessStatus is not None` assertion** in `test_get_state_packet_contains_expected_fields` with assertions on the live-schema fields populated by [policy.py:354-376](selfdrive/monitoring/policy.py:354) — at minimum `alertLevel`, `activePolicy`, `isRHD`, plus `visionPolicyState.faceDetected` and `visionPolicyState.isDistracted` (the per-policy sub-state fields verified during pre-flight).
- **Rename the file to `test_monitoring_policy.py`** to match what it now tests. Update any path references (CI configs, codecov targets, README pointers) accordingly.
- Coverage target: per `codecov.yml`, monitoring is at the safety-critical 85% line. The ported tests must collectively exercise no fewer lines of `policy.py` than the originals exercised of the deleted `helpers.py`.

## Capabilities

### New Capabilities

- `monitoring-policy-test-coverage`: Defines what the ported test file must cover against the post-merge `selfdrive/monitoring/policy.py` module — required test classes, the `get_state_packet` field-presence contract, and the symbol-by-symbol disposition for the three removed identifiers (`DistractedType`, `DriverProb`, `face_orientation_from_net`).

### Modified Capabilities

*(none — `openspec/specs/test-coverage/` covers the algorithm-harness suite, not monitoring tests.)*

## Impact

- **Code touched:**
  - `selfdrive/monitoring/tests/test_monitoring_helpers.py` — renamed to `test_monitoring_policy.py`; large rewrite expected (~1,146 lines triaged class by class, possibly fewer after removals).
- **No production code changes.** Test-only.
- **Dependencies:**
  - Lands AFTER (or alongside) the upstream merge from branch `claude/festive-lovelace-8096bd` reaches `develop`.
  - Independent of the sibling change `migrate-dm-tests-new-schema` — the two touch different files and can ship separately.
- **CI/coverage:** `codecov.yml` monitoring component target (85%) must hold post-port. Port should preserve coverage of `policy.py` lines that correspond to what the originals covered in the old `helpers.py`.
- **Safety implication:** Driver monitoring is safety-critical per [CLAUDE.md](CLAUDE.md). Losing test coverage of `DriverMonitoring` event handling, awareness reset, and state-packet construction would be a real regression. Pre-flight (in `tasks.md`) must explicitly enumerate which of the original 75 tests can be preserved versus deleted, with rationale recorded.
- **Effort estimate:** Day-scale, not hour-scale. The triage is the bulk of the work — actual edits per test are usually small once the disposition is decided.
