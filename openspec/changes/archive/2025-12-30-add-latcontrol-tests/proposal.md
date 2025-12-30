# Proposal: add-latcontrol-tests

## Summary
Add comprehensive unit tests for lateral control implementations (`LatControlAngle`, `LatControlPID`, `LatControlTorque`) to improve test coverage from 0% to 90%+ for these safety-critical control modules.

## Motivation
The lateral control implementations are safety-critical components that directly affect vehicle steering behavior. Currently:
- `latcontrol_angle.py` (24 lines) - **0% coverage**
- `latcontrol_pid.py` (32 lines) - **0% coverage**
- `latcontrol_torque.py` (77 lines) - **0% coverage**

The base class `latcontrol.py` has 95% coverage with existing tests, but the concrete implementations have no dedicated test coverage. Per project conventions, safety-critical modules (controls) require 95% coverage.

## Scope
- Add test files for each lateral control implementation
- Follow existing patterns from `test_latcontrol.py` and `test_longcontrol.py`
- Use pytest-mock's `mocker` fixture (per TID251 compliance)
- Test initialization, update logic, and edge cases

## Out of Scope
- Changes to production code
- Tests for `lat_mpc.py` or `long_mpc.py` (separate effort)
- Integration tests with actual vehicle models

## Risks
- **Low**: These are additive test-only changes
- Mock complexity for `LatControlTorque` due to dependencies on `torque_from_lateral_accel` and `lateral_accel_from_torque` functions

## Success Criteria
- All three lateral control modules have dedicated test files
- Coverage for `selfdrive/controls/lib/latcontrol_*.py` reaches 90%+
- All tests pass in CI
- No changes to production code required
