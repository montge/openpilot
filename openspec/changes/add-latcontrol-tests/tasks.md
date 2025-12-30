# Tasks: add-latcontrol-tests

## Implementation Order

Tasks are ordered by complexity (simplest first) to enable incremental progress.

### Phase 1: LatControlAngle (simplest, 24 lines)

- [x] Create `test_latcontrol_angle.py` with mock helpers
- [x] Add tests for `__init__` (sat_check_min_speed override, use_steer_limited_by_safety)
- [x] Add tests for `update` when inactive (returns current steering angle)
- [x] Add tests for `update` when active (calculates desired angle from curvature)
- [x] Add tests for saturation detection (Tesla vs non-Tesla behavior)
- [x] Verify coverage >= 90% for `latcontrol_angle.py` - **100% achieved**

### Phase 2: LatControlPID (medium, 32 lines)

- [x] Create `test_latcontrol_pid.py` with mock helpers
- [x] Add tests for `__init__` (PID controller setup, ff_factor, steer_feedforward)
- [x] Add tests for `update` when inactive (returns zero torque)
- [x] Add tests for `update` when active (PID calculation, feedforward)
- [x] Add tests for integrator freeze conditions (steer limited, steering pressed, low speed)
- [x] Add tests for saturation detection
- [x] Verify coverage >= 90% for `latcontrol_pid.py` - **100% achieved**

### Phase 3: LatControlTorque (complex, 77 lines)

- [x] Create `test_latcontrol_torque.py` with mock helpers
- [x] Add tests for `__init__` (torque_params, PID setup, buffers, filters)
- [x] Add tests for `update_live_torque_params` (dynamic parameter updates)
- [x] Add tests for `update_limits` (limit recalculation)
- [x] Add tests for `update` when inactive (returns zero torque)
- [x] Add tests for `update` when active (lateral accel calculation, jerk filtering)
- [x] Add tests for roll compensation
- [x] Add tests for friction compensation
- [x] Add tests for integrator freeze conditions
- [x] Verify coverage >= 90% for `latcontrol_torque.py` - **100% achieved**

### Phase 4: Validation

- [x] Run full test suite for `selfdrive/controls/lib/tests/` - **231 tests passed**
- [x] Generate coverage report and verify all three modules >= 90%
- [x] Verify no ruff/mypy errors

## Summary

All tasks completed successfully:
- 66 new tests added across 3 test files
- 100% coverage achieved for all three lateral control modules
- All ruff checks pass
- Total controls/lib coverage increased from 46% to 59%

## Dependencies

- All tasks can be parallelized within each phase
- Phase 2 and 3 can start once Phase 1 patterns are established
- Phase 4 depends on all previous phases

## Notes

- Follow patterns from existing `test_latcontrol.py` for mock helpers
- Use `mocker.MagicMock()` per pytest-mock conventions (TID251)
- Mock VM (VehicleModel), params, and car-specific functions
