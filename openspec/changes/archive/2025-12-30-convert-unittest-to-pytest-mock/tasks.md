# Tasks: Convert unittest to pytest-mock

## Overview

Convert test files from `unittest.mock` to `pytest-mock` fixture patterns to comply with ruff TID251 rule.

## Status: COMPLETED

All TID251 violations have been resolved. The conversion was completed in 3 commits:

1. `5d8d9360c` - refactor(tests): convert unittest.mock to pytest-mock (53 files)
2. `2711b4759` - fix(tests): use numpy bool-compatible assertions in test_latcontrol
3. `f3e84becc` - refactor(tests): convert test_selfdrived.py to pytest-mock

## Phase 1: Common Module (14 files)

- [x] **1.1** Convert `common/tests/test_api.py`
- [x] **1.2** Convert `common/tests/test_git.py`
- [x] **1.3** Convert `common/tests/test_gps.py`
- [x] **1.4** Convert `common/tests/test_logging_extra.py`
- [x] **1.5** Convert `common/tests/test_prefix.py`
- [x] **1.6** Convert `common/tests/test_realtime.py`
- [x] **1.7** Convert `common/tests/test_spinner.py`
- [x] **1.8** Convert `common/tests/test_stat_live.py`
- [x] **1.9** Convert `common/tests/test_swaglog.py`
- [x] **1.10** Convert `common/tests/test_time_helpers.py`
- [x] **1.11** Convert `common/tests/test_timeout.py`
- [x] **1.12** Convert `common/tests/test_util.py`
- [x] **1.13** Convert `common/tests/test_utils.py`
- [x] **1.14** Verify: `ruff check common/tests/ --select TID251` ✓

## Phase 2: Selfdrive - Car (3 files)

- [x] **2.1** Convert `selfdrive/car/tests/test_card.py`
- [x] **2.2** Convert `selfdrive/car/tests/test_cruise.py`
- [x] **2.3** Convert `selfdrive/car/tests/test_models.py`
- [x] **2.4** Verify: `ruff check selfdrive/car/tests/ --select TID251` ✓

## Phase 3: Selfdrive - Controls (10 files)

- [x] **3.1** Convert `selfdrive/controls/lib/tests/test_desire_helper.py`
- [x] **3.2** Convert `selfdrive/controls/lib/tests/test_drive_helpers.py`
- [x] **3.3** Convert `selfdrive/controls/lib/tests/test_latcontrol.py`
- [x] **3.4** Convert `selfdrive/controls/lib/tests/test_ldw.py`
- [x] **3.5** Convert `selfdrive/controls/lib/tests/test_longcontrol.py`
- [x] **3.6** Convert `selfdrive/controls/lib/tests/test_longitudinal_planner.py`
- [x] **3.7** Convert `selfdrive/controls/tests/test_controlsd.py`
- [x] **3.8** Convert `selfdrive/controls/tests/test_desire_helper.py`
- [x] **3.9** Convert `selfdrive/controls/tests/test_ldw.py`
- [x] **3.10** Convert `selfdrive/controls/tests/test_longitudinal_planner.py`
- [x] **3.11** Convert `selfdrive/controls/tests/test_radard.py`
- [x] **3.12** Verify: `ruff check selfdrive/controls/ --select TID251` ✓

## Phase 4: Selfdrive - Other (10 files)

- [x] **4.1** Convert `selfdrive/locationd/test/test_constants.py`
- [x] **4.2** Convert `selfdrive/locationd/test/test_helpers.py`
- [x] **4.3** Convert `selfdrive/locationd/test/test_helpers_util.py`
- [x] **4.4** Convert `selfdrive/locationd/test/test_paramsd.py`
- [x] **4.5** Convert `selfdrive/locationd/test/test_torqued.py`
- [x] **4.6** Convert `selfdrive/monitoring/tests/test_monitoring_helpers.py`
- [x] **4.7** Convert `selfdrive/selfdrived/tests/test_events.py`
- [x] **4.8** Convert `selfdrive/selfdrived/tests/test_helpers.py`
- [x] **4.9** Convert `selfdrive/selfdrived/tests/test_selfdrived.py`
- [x] **4.10** Verify: `ruff check selfdrive/ --select TID251` ✓

## Phase 5: System Module (9 files)

- [x] **5.1** Convert `system/hardware/tests/test_base.py`
- [x] **5.2** Convert `system/hardware/tests/test_hw.py`
- [x] **5.3** Convert `system/loggerd/tests/test_config.py`
- [x] **5.4** Convert `system/loggerd/tests/test_xattr_cache.py`
- [x] **5.5** Convert `system/manager/test/test_manager_integration.py`
- [x] **5.6** Convert `system/manager/test/test_process.py`
- [x] **5.7** Convert `system/tests/test_statsd.py`
- [x] **5.8** Convert `system/tests/test_version.py`
- [x] **5.9** Convert `system/ui/lib/tests/test_emoji.py`
- [x] **5.10** Verify: `ruff check system/ --select TID251` ✓

## Phase 6: Tools & Cereal (7 files)

- [x] **6.1** Convert `cereal/tests/test_services.py`
- [x] **6.2** Convert `tools/car_porting/test_car_model.py`
- [x] **6.3** Convert `tools/lib/tests/test_api.py`
- [x] **6.4** Convert `tools/lib/tests/test_auth_config.py`
- [x] **6.5** Convert `tools/lib/tests/test_cache.py`
- [x] **6.6** Convert `tools/lib/tests/test_route.py`
- [x] **6.7** Convert `tools/lib/tests/test_sanitizer.py`
- [x] **6.8** Convert `tools/lib/tests/test_url_file.py`
- [x] **6.9** Verify: `ruff check tools/ cereal/ --select TID251` ✓

## Phase 7: Final Validation

- [x] **7.1** Run full ruff check: `ruff check . --select TID251` ✓
- [x] **7.2** Run pre-commit: `pre-commit run --all-files` ✓
- [x] **7.3** Run affected tests: All tests pass ✓
- [x] **7.4** Verify no functional test changes ✓

## Conversion Pattern Used

For each file:
1. Remove `from unittest.mock import ...` and `import unittest`
2. Add `mocker` parameter to test functions that need mocking
3. Replace `@patch('x')` decorators with `mocker.patch('x')` calls
4. Replace `with patch('x') as m:` with `m = mocker.patch('x')`
5. Replace `MagicMock()` with `mocker.MagicMock()`
6. Run `ruff check <file> --select TID251` to verify
7. Run `pytest <file>` to verify tests pass
