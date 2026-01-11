## 1. Replace Banned unittest.mock Imports

- [ ] 1.1 Replace `MagicMock` in `adapters.py` with pytest fixtures or `pytest-mock`
- [ ] 1.2 Replace `patch` in `deterministic.py` with `pytest-mock` mocker fixture
- [ ] 1.3 Replace `patch` in `test_gpu.py` with `pytest-mock` mocker fixture

## 2. Add strict=True to zip() Calls

- [ ] 2.1 Fix `zip()` in `runner.py:51` (Scenario.with_ground_truth)
- [ ] 2.2 Fix `zip()` in `runner.py:205` (ScenarioRunner.compare)

## 3. Fix Exception Chaining

- [ ] 3.1 Fix `raise ImportError` in `scenario_schema.py:194`
- [ ] 3.2 Fix `raise ImportError` in `scenarios.py:53`
- [ ] 3.3 Fix `raise ImportError` in `scenarios.py:123`
- [ ] 3.4 Fix `raise ImportError` in `scenarios.py:217`

## 4. Remove Unused Variables

- [ ] 4.1 Remove `lane_width` in `scenario_generator.py:196`
- [ ] 4.2 Remove `phase` in `scenario_generator.py:216`
- [ ] 4.3 Remove `stop_distance` in `scenario_generator.py:369`
- [ ] 4.4 Remove/rename `ctx` in `test_deterministic.py:204`
- [ ] 4.5 Remove/rename `ctx` in `test_deterministic.py:257`
- [ ] 4.6 Remove `initial_x` and `initial_y` in `test_vehicle_dynamics.py:339-340`

## 5. Fix Unused Loop Variables

- [ ] 5.1 Rename `name` to `_name` in `test_scenarios.py:325`
- [ ] 5.2 Rename `name` to `_name` in `test_vehicle_dynamics.py:322`

## 6. Remove Unused Imports

- [ ] 6.1 Remove unused `pandas` import in `scenarios.py:119`

## 7. Verify

- [ ] 7.1 Run `ruff check` and confirm all issues resolved
- [ ] 7.2 Run tests to ensure no regressions
