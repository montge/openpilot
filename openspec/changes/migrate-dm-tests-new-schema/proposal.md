## Why

The fork-only test file `selfdrive/controls/tests/test_controlsd.py` added in commit `4af466c7f` ("test: add comprehensive test coverage for controls module") will break once the upstream merge at commit `e8b2e3291` (staged on branch `claude/festive-lovelace-8096bd`) lands on `develop`. Upstream redesigned `cereal.log.DriverMonitoringState` — the old `awarenessStatus` Float32 (and its peers) moved into a new `DriverMonitoringStateDEPRECATED` struct, and the live struct now exposes an `alertLevel` enum (`none`/`one`/`two`/`three`) plus an `activePolicy` enum (`wheeltouch`/`vision`) along with policy-specific sub-states. Upstream's `forceDecel` rule is now `alertLevel == AlertLevel.three`, not `awarenessStatus < 0.0`.

These tests are the only coverage we have for `controlsd._set_driver_monitoring_state` mocking, the `_setup_default_sm` fixture, and three `forceDecel` scenarios (terminal DM alert, soft-disabling, nominal). A naive find-and-replace would silently weaken what they verify, so the redesign needs to be planned, not patched.

**Scope note (post pre-flight):** The original briefing also named `selfdrive/monitoring/tests/test_monitoring_helpers.py:349`. Pre-flight (task 1.5) revealed that file is comprehensively broken post-merge — `selfdrive/monitoring/helpers.py` no longer exists; the symbols moved into `selfdrive/monitoring/policy.py` and several imported symbols (`DistractedType`, `DriverProb`, `face_orientation_from_net`) appear to have been removed entirely. Porting that 1,146-line / 75-test file is a module-relocation problem categorically different from the `awarenessStatus` field rename, and is split off into a sibling change `port-test-monitoring-helpers-to-policy`.

## What Changes

- **BREAKING (fork-only test API):** Replace the `awarenessStatus=<float>` keyword in `_set_driver_monitoring_state` and `_setup_default_sm` (`selfdrive/controls/tests/test_controlsd.py`) with an `alert_level=<AlertLevel>` keyword that sets `driverMonitoringState.alertLevel`. Default to `AlertLevel.none` (nominal).
- Rewrite the three `forceDecel` test cases (`test_force_decel_on_negative_awareness`, `test_force_decel_on_soft_disabling`, `test_no_force_decel_under_normal_conditions`) so the trigger condition mirrors the new `controlsd.py:196` rule (`alertLevel == AlertLevel.three`). Rename the negative-awareness case to `test_force_decel_on_terminal_dm_alert` to match what it now actually exercises.
- Update the three incidental mock writes (test_controlsd.py:566, 847, 1216) that currently set `awarenessStatus = 1.0` to instead set `alertLevel = AlertLevel.none` so default-state mocks remain nominal.
- Add a regression-guard test asserting that `int(log.DriverMonitoringState.AlertLevel.three) == 3`, since the `cereal/log.capnp` schema explicitly documents enum-ordinal drift as a hazard for the name-based comparison `controlsd.py` uses.

## Capabilities

### New Capabilities
- `dm-state-test-coverage`: Defines the test-level contract this fork holds against `cereal.log.DriverMonitoringState` — which fields the mock fixtures must populate, which `forceDecel` transitions must be exercised, and which `get_state_packet` fields must be asserted. Owning this as a capability gives us a stable target the next time upstream reshapes the DM schema.

### Modified Capabilities
*(none — `openspec/specs/test-coverage/` covers algorithm harness modules only; no existing spec covers controlsd or DM helper tests.)*

## Impact

- **Code touched:**
  - `selfdrive/controls/tests/test_controlsd.py` only (~10 sites + 1 new regression guard)
- **Out of scope (split to sibling change):** `selfdrive/monitoring/tests/test_monitoring_helpers.py` — see `port-test-monitoring-helpers-to-policy`.
- **No production code changes.** This is test-only.
- **Dependencies:** Lands AFTER the upstream merge from branch `claude/festive-lovelace-8096bd` (commit `e8b2e3291`) reaches `develop`, OR is implemented directly on that merge branch ahead of the merge to `develop`. The new schema is required on disk for the rewrite to compile.
- **CI/coverage:** Controls coverage target in `codecov.yml` (80%) must hold post-migration; the redesigned tests should exercise the same control-flow branches as the originals.
- **Safety implication:** `forceDecel` is a safety-critical signal. The redesigned tests must verify the new `alertLevel == three` trigger fires under exactly the same operational scenarios the old tests verified for `awarenessStatus < 0` — losing this coverage would let a regression in DM-driven decel slip past CI.
- **No spec deletion / no breaking change to public APIs.** The `awarenessStatus` keyword in the two private test helpers is fork-only and not consumed elsewhere.
