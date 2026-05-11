## Context

The fork carries ~125 fork-only test cases added in commit `4af466c7f` to harden the upstream `selfdrive/controls` and `selfdrive/monitoring` modules well past their stock coverage. Two of those files — [selfdrive/controls/tests/test_controlsd.py](selfdrive/controls/tests/test_controlsd.py) and [selfdrive/monitoring/tests/test_monitoring_helpers.py](selfdrive/monitoring/tests/test_monitoring_helpers.py) — encode assumptions about the `cereal.log.DriverMonitoringState` schema that upstream has since rewritten.

Concretely, on `develop` today the schema exposes `awarenessStatus :Float32` at field `@3` and `forceDecel` is computed in [selfdrive/controls/controlsd.py](selfdrive/controls/controlsd.py) as `(driverMonitoringState.awarenessStatus < 0.0) or (selfdriveState.state == softDisabling)`. After the upstream merge staged on branch `claude/festive-lovelace-8096bd` (commit `e8b2e3291`) lands, the live struct is replaced wholesale: the old fields move into `DriverMonitoringStateDEPRECATED` and the new struct exposes an `AlertLevel` enum (`none`/`one`/`two`/`three`), a `MonitoringPolicy` enum (`wheeltouch`/`vision`), per-policy sub-states, lockout flags, and RHD calibration. The `forceDecel` rule becomes `alertLevel == AlertLevel.three`. The semantics are preserved (terminal DM alert ⇒ decelerate) but the trigger is now a discrete enum, not a continuous float crossing zero.

Stakeholders: this fork's test suite (CI gate), and any future fork maintainer reading these tests to understand what `forceDecel` is supposed to do. There is no production-code dependency — `controlsd.py` and `monitoring/helpers.py` come from upstream and already use the new schema in the merge commit.

Constraints:
- The redesign cannot land until the upstream merge is on `develop`; the new struct doesn't exist on disk in this worktree yet. The proposal/design/specs phases proceed now; the `tasks` phase blocks on the merge.
- `forceDecel` is safety-critical; weakening its coverage is unacceptable.
- Coverage targets in `codecov.yml` (controls 80%, monitoring 85%) must hold.
- Tests run under pytest with `OpenpilotPrefix` isolation per [conftest.py](conftest.py).

## Goals / Non-Goals

**Goals:**
- Preserve every behavioral assertion the original tests made, expressed against the new schema.
- Make the test-side `forceDecel` mapping derive from `controlsd.py`'s rule (or from a single shared constant) so future enum reorders fail loudly in tests rather than silently disabling decel coverage.
- Keep the public test class names and method names stable where the scenario semantics survive the rewrite, so coverage diffs are reviewable and `git blame` continuity is maintained on unchanged tests around them.
- Lay down a `dm-state-test-coverage` capability spec that survives this change and can be reused the next time upstream reshapes DM state.

**Non-Goals:**
- Modifying any production code (`selfdrive/controls/controlsd.py`, `selfdrive/monitoring/helpers.py`, `selfdrive/monitoring/policy.py`, `cereal/log.capnp`). All of those are upstream-owned at the merge commit.
- Migrating other fork-only tests added in `4af466c7f` (test_longcontrol, test_longitudinal_planner, test_drive_helpers, test_latcontrol, test_ldw) — those four don't reference the DM schema and stay untouched.
- Adding *new* coverage beyond what the originals provided. This is a port, not a feature add. Any net-new assertions are limited to the one regression guard described under Decision 4.
- Implementing the rewrite itself. This change defines the contract; `tasks` will execute it after the merge lands.

## Decisions

### Decision 1: Map `awarenessStatus` semantics to `AlertLevel`, not to `VisionPolicyState.awarenessPercent`

The new struct has both `alertLevel` (enum) and `visionPolicyState.awarenessPercent` (Int8). The old `awarenessStatus` was a Float32 the original tests treated as a "drive-level alert intensity" — they used `< 0` to mean "terminal alert" and `1.0` to mean "nominal". The new `controlsd.py:196` consumes `alertLevel` directly for `forceDecel`. Therefore the test fixture parameter and the scenarios should target `alertLevel`, not the percent.

**Rationale:** The percent is policy-internal plumbing (and is `Int8`-clamped to `[0,100]`); `alertLevel` is the enum the consumer actually reads. Tying tests to `alertLevel` keeps them pinned to the consumer contract, not the producer's intermediate state.

**Alternatives considered:**
- Set `visionPolicyState.awarenessPercent = 0` and rely on the producer to translate to `AlertLevel.three`. *Rejected:* the producer (`DriverMonitoring._update_events`) is not in the path of these tests; they mock the published struct directly and wouldn't trigger the translation. Setting the percent without setting the alert level would yield `AlertLevel.none` on the wire and no `forceDecel`.
- Keep using a float-shaped knob and translate inside the helper. *Rejected:* hides the schema change behind a fork-local API and makes future schema audits harder.

### Decision 2: New fixture parameter is `alert_level: log.DriverMonitoringState.AlertLevel = AlertLevel.none`

Replace `awarenessStatus: float = 1.0` in `_set_driver_monitoring_state` and `_setup_default_sm` with `alert_level: AlertLevel = AlertLevel.none`. Import once at module top: `AlertLevel = log.DriverMonitoringState.AlertLevel`.

**Rationale:** Snake_case keyword matches Python convention (the original `awarenessStatus` was camelCase only because it mirrored the capnp field name; we're not bound to that anymore since the field name itself moved). Default of `none` mirrors the original default of `1.0` (nominal).

**Alternatives considered:**
- Keep the camelCase `alertLevel=`. *Rejected:* readability — `alert_level=AlertLevel.three` reads cleanly; `alertLevel=AlertLevel.three` looks like a typo.
- Take a `forceDecel: bool` directly and have the helper pick a level. *Rejected:* couples the test fixture to current `forceDecel` rules; we want the test to *exercise* the rule, not bypass it.

### Decision 3: Derive expected `forceDecel` from a single in-test constant matching `controlsd.py`

Add at the top of the `TestControlsForceDecel` class:
```python
TERMINAL_ALERT = log.DriverMonitoringState.AlertLevel.three
```
And replace every inline `bool((... .awarenessStatus < 0.0) or ...)` recompute with:
```python
expected = bool((self.mock_sm['driverMonitoringState'].alertLevel == TERMINAL_ALERT) or
                (self.mock_sm['selfdriveState'].state == State.softDisabling))
```

**Rationale:** The originals duplicated the `forceDecel` formula inline three times. Centralizing the constant means a future enum reorder (e.g., adding an `AlertLevel.four`) only needs one update, and the tests still reflect what `controlsd.py` actually does. We deliberately do NOT import the formula from `controlsd.py` itself — the point of the test is to catch divergence, not paper over it.

**Alternatives considered:**
- Hard-code `AlertLevel.three` inline. *Rejected:* harder to grep and to update.
- Import `forceDecel` calculation from `controlsd`. *Rejected:* tautological — the test would always match the production code.

### Decision 4: Add one regression-guard test for the `three == terminal` invariant

The new `cereal/log.capnp` carries this comment on the enum:
```
# ordinal must match the name to prevent bugs
# comparing against the raw ordinal value
```
This is exactly the bug class that would silently disable forceDecel: someone reorders the enum and `AlertLevel.three` becomes ordinal 2 (or some other value) while `controlsd.py`'s string-name comparison still reads "three". Add a single new test:
```python
def test_alert_level_three_is_ordinal_three():
    """Regression guard: the AlertLevel enum's `three` member must keep ordinal 3.
    controlsd.forceDecel relies on the name; if the ordinal drifts the wire-level
    comparison can break across producer/consumer version mismatches."""
    assert int(log.DriverMonitoringState.AlertLevel.three) == 3
```

**Rationale:** Cheap (one assertion), targets a documented hazard called out in the schema itself, and gives the fork an early warning if a future upstream merge perturbs the enum. This is the only net-new assertion in this change.

**Alternatives considered:**
- Skip it. *Rejected:* the schema literally documents this as a hazard; cheap insurance.
- Assert all four ordinals. *Rejected:* over-specifies — the fork only depends on `three` for `forceDecel`. Asserting `none`, `one`, `two` adds churn for any benign rename.

### Decision 5: Replace `state.awarenessStatus is not None` with assertions over `alertLevel`, `activePolicy`, and `isRHD`

The original `test_get_state_packet_contains_expected_fields` asserts on three fields: `awarenessStatus`, `faceDetected`, `isDistracted`. After the merge, `faceDetected` and `isDistracted` move into `visionPolicyState` (sub-struct), and `awarenessStatus` is gone. Picking the new top-level scalars `alertLevel`, `activePolicy`, and `isRHD` keeps the test's intent ("`get_state_packet` populates the published struct, not just defaults") and keeps it readable.

**Rationale:** Top-level fields fail loudly if the producer regresses to an empty struct; sub-struct fields would require `.visionPolicyState.faceDetected` chains that the verify phase needs to confirm exist on the actual `get_state_packet()` output post-merge.

**Alternatives considered:**
- Reach into `visionPolicyState.faceDetected` to preserve symmetry with the original. *Rejected without verification:* depends on whether `helpers.get_state_packet()` populates that sub-struct unconditionally or only when `activePolicy == vision`. Marked as an Open Question; verify in implementation.
- Drop the test entirely. *Rejected:* it's the only test we have asserting `get_state_packet()` returns a populated message, not just defaults. That contract still matters.

### Decision 6: Keep test class and method names where semantics are preserved; rename only where they would lie

- `test_force_decel_on_negative_awareness` → `test_force_decel_on_terminal_dm_alert` (the trigger is no longer "negative awareness")
- `test_force_decel_on_soft_disabling` → unchanged (state machine trigger is unchanged)
- `test_no_force_decel_under_normal_conditions` → unchanged
- `test_get_state_packet_contains_expected_fields` → unchanged (the *intent* is unchanged; only the field list moves)
- Helper `_set_driver_monitoring_state` → unchanged name, signature changes
- Helper `_setup_default_sm` → unchanged name, signature changes

**Rationale:** Renaming for renaming's sake breaks `git blame` continuity. Rename only when the old name is now actively misleading.

## Risks / Trade-offs

- **Risk:** The merge from `claude/festive-lovelace-8096bd` doesn't land cleanly on `develop`, or lands with further schema tweaks beyond what `e8b2e3291` shows.
  - **Mitigation:** Pin the design and spec to the specific commit (`e8b2e3291`); the verify phase re-checks the schema on `develop` HEAD before tasks run. If the schema differs, treat it as a new change rather than retrofitting this one.

- **Risk:** Coverage drops because the new tests stub fewer fields than the new struct exposes.
  - **Mitigation:** The default-mock fixture (`_setup_default_sm`) only needs to populate fields that `controlsd` actually reads (`alertLevel`); the rest stay zero/default. Coverage of `controlsd.py:196` is what matters and that line reads exactly one field. Verify with `pytest --cov=openpilot.selfdrive.controls.controlsd` after porting.

- **Risk:** Silent semantic regression — the new tests "pass" but no longer exercise the terminal-alert branch.
  - **Mitigation:** Decision 4's enum-ordinal regression guard plus the explicit `assert force_decel is True` on the terminal-alert case. The verify phase MUST run `pytest selfdrive/controls/tests/test_controlsd.py::TestControlsForceDecel -v` with both `alert_level=AlertLevel.three` and `alert_level=AlertLevel.none` paths and confirm both assertion polarities fire.

- **Risk:** Mocks don't reach `forceDecel` because some sibling field on `DriverMonitoringState` (e.g., `lockout`, `alwaysOn`) gates `controlsd` behavior we don't yet appreciate.
  - **Mitigation:** Re-grep `controlsd.py` (post-merge) for every `driverMonitoringState.<field>` access. The current grep on `e8b2e3291` shows only `alertLevel` is consumed there, but verify again on `develop` HEAD.

- **Trade-off:** We're not migrating to the new sub-states (`visionPolicyState`, `wheeltouchPolicyState`). The original tests didn't cover those and we're not extending scope. Future work item if/when fork-side coverage of those sub-states becomes valuable.

## Migration Plan

1. **Block on merge.** Do not begin the `tasks` phase until commit `e8b2e3291` (or its successor on `claude/festive-lovelace-8096bd`) is merged into `develop`.
2. **Verify schema parity.** On `develop` HEAD, confirm `cereal/log.capnp` defines `DriverMonitoringState.AlertLevel { none, one, two, three }` and that `selfdrive/controls/controlsd.py` uses `alertLevel == AlertLevel.three` for `forceDecel`. If either drifted, halt and reopen design.
3. **Branch off develop.** Create a feature branch `feature/migrate-dm-tests-new-schema` from `develop` (per the gitflow conventions in [CLAUDE.md](CLAUDE.md)).
4. **Execute tasks.** Apply the rewrites in the order the `tasks.md` artifact specifies, running `pytest <file>` after each file's rewrite.
5. **Coverage check.** Run `pytest --cov=openpilot.selfdrive.controls.controlsd --cov=openpilot.selfdrive.monitoring.helpers selfdrive/controls/tests/test_controlsd.py selfdrive/monitoring/tests/test_monitoring_helpers.py` and confirm coverage of the lines previously covered by the migrated tests has not dropped.
6. **PR to develop only.** Per [CLAUDE.md](CLAUDE.md), `origin = montge/openpilot` is the only push target; never push to `upstream`.
7. **Rollback.** Pure test-file change — `git revert` of the merge commit is fully sufficient; no production state to roll back.

## Open Questions

- **Q1:** Does `helpers.DriverMonitoring.get_state_packet()` populate `visionPolicyState.faceDetected` unconditionally on a fresh instance, or only when `activePolicy == vision` and a face has been seen? *Answered during verify by reading post-merge `helpers.py` and reproducing locally.* Resolution informs whether Decision 5's assertion list extends into the sub-struct.

- **Q2:** Is there a fork-local mock-message helper anywhere else in `selfdrive/controls/tests/` or `selfdrive/monitoring/tests/` that also writes `awarenessStatus`? `grep -r "awarenessStatus" selfdrive/` on `develop` HEAD post-merge will be the source of truth — anything beyond the two files in this proposal expands scope.

- **Q3:** Does `codecov.yml`'s patch coverage gate (80%) treat test-only diffs as a no-op, or does it require the patched test files to themselves be covered? *Verify via a draft PR before merging.*
