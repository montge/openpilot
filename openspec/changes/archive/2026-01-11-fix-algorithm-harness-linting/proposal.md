# Change: Fix Algorithm Harness Linting Issues

## Why

The algorithm harness code merged from Claude Code web has 19 linting violations that need to be fixed to pass pre-commit hooks and maintain code quality standards.

## What Changes

- **Replace banned `unittest.mock` imports with pytest equivalents**:
  - `adapters.py`: Replace `MagicMock` with pytest fixtures
  - `deterministic.py`: Replace `patch` with `pytest-mock`
  - `test_gpu.py`: Replace `patch` with `pytest-mock`

- **Add `strict=True` to `zip()` calls**:
  - `runner.py`: 2 instances

- **Fix exception chaining** (use `raise ... from err`):
  - `scenario_schema.py`: 1 instance
  - `scenarios.py`: 3 instances

- **Remove unused variables**:
  - `scenario_generator.py`: `lane_width`, `phase`, `stop_distance`
  - `test_deterministic.py`: `ctx` (2 instances)
  - `test_vehicle_dynamics.py`: `initial_x`, `initial_y`

- **Fix unused loop variables**:
  - `test_scenarios.py`: Rename `name` to `_name`
  - `test_vehicle_dynamics.py`: Rename `name` to `_name`

- **Remove unused import**:
  - `scenarios.py`: Remove unused `pandas` import

## Impact

- Affected specs: algorithm-test-harness
- Affected code: `selfdrive/controls/lib/tests/algorithm_harness/`, `system/hardware/nvidia/`
- No functional changes - linting fixes only
