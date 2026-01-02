# Tasks: Add Module Test Coverage

## 1. Priority 1: Largest Coverage Gaps (50-65%)

### 1.1 tools.lib (50% → 71% achieved, target 90%)
- [x] 1.1.1 Identify uncovered files in tools/lib/
- [x] 1.1.2 Add tests for route.py edge cases (68% → 98%)
- [x] 1.1.3 Add tests for github_utils.py (0% → 100%)
- [x] 1.1.4 Add tests for auth.py (0% → 91%)
- [x] 1.1.5 Add tests for bootlog.py (0% → 100%)
- [x] 1.1.6 Add tests for file_sources.py (49% → 100%)
- [x] 1.1.7 Add tests for filereader.py (68% → 93%)
- [x] 1.1.8 Add tests for url_file.py (56% → 99%)
- [ ] 1.1.9 Add tests for framereader.py (requires video files)
- [ ] 1.1.10 Add tests for vidindex.py (requires video files)
- [ ] 1.1.11 Verify 90% coverage achieved

### 1.2 system.athena (58% → 61%)
- [x] 1.2.1 Identify uncovered files in system/athena/
- [ ] 1.2.2 Add tests for athenad.py API handlers
- [x] 1.2.3 Add tests for registration.py (76% → 80%)
- [x] 1.2.4 Add tests for manage_athenad.py (0% → 92%)
- [ ] 1.2.5 Verify 90% coverage achieved

### 1.3 system.manager (63% → 90%)
- [x] 1.3.1 Add tests for helpers.py write_onroad_params
- [ ] 1.3.2 Add tests for process.py lifecycle methods
- [ ] 1.3.3 Add tests for manager.py startup/shutdown
- [x] 1.3.4 Add tests for process_config.py (65% → 100%)
- [ ] 1.3.5 Verify 90% coverage achieved

## 2. Priority 2: Medium Coverage Gaps (65-80%)

### 2.1 selfdrive.monitoring (67% → 98% achieved!)
- [x] 2.1.1 Identify uncovered files in selfdrive/monitoring/
- [x] 2.1.2 Add tests for helpers.py (67% → 98%)
- [ ] 2.1.3 Add tests for dmonitoringd.py (0% → 90%)
- [x] 2.1.4 Verify 90% coverage achieved

### 2.2 selfdrive.controls (78% → 90%)
- [ ] 2.2.1 Identify uncovered files in selfdrive/controls/
- [ ] 2.2.2 Add tests for controlsd.py edge cases
- [ ] 2.2.3 Add tests for radard.py
- [ ] 2.2.4 Add tests for plannerd.py
- [ ] 2.2.5 Verify 90% coverage achieved

## 3. Priority 3: Small Coverage Gaps (80-90%)

### 3.1 selfdrive.selfdrived (80% → 91% achieved!)
- [x] 3.1.1 Identify uncovered branches in selfdrive/selfdrived/
- [x] 3.1.2 Add tests for events.py (85% → 91%)
- [ ] 3.1.3 Add tests for selfdrived.py state transitions
- [x] 3.1.4 Verify 90% coverage achieved

### 3.2 system.loggerd (82% → 82%)
- [x] 3.2.1 Identify uncovered files in system/loggerd/
- [x] 3.2.2 Add tests for uploader.py helper functions (new file)
- [x] 3.2.3 deleter.py already at 99%
- [ ] 3.2.4 Verify 90% coverage achieved (uploader.py has network code)

## 4. Validation
- [ ] 4.1 Run full coverage report for all target modules
- [ ] 4.2 Ensure no regressions in existing tests
- [ ] 4.3 Update coverage baselines if needed

## Progress Summary

### Session 1 (Initial Implementation)
- **tools.lib**: 50% → 68% (+18%)
  - route.py: 68% → 98% (added Route, Segment, SegmentRange tests)
  - github_utils.py: 0% → 100% (new test file)
  - auth.py: 0% → 91% (new test file)
  - bootlog.py: 0% → 100% (new test file)
  - file_sources.py: 49% → 100% (new test file)
- **system.manager**: Added helpers.py tests
- **New test files created**: 6
- **New tests added**: 131

### Session 2 (Continued Implementation)
- **system.manager**: process_config.py: 65% → 100% (+35%)
  - Added 36 tests covering all helper functions
- **selfdrive.monitoring**: helpers.py: 67% → 98% (+31%)
  - Added 38 tests covering _set_timers, _get_distracted_types, _update_states, run_step
- **New tests added**: 74

### Session 3 (Athena Coverage)
- **system.athena**: manage_athenad.py: 0% → 92% (+92%)
  - Added 3 tests for main function and constants
- **system.athena**: registration.py: 76% → 80% (+4%)
  - Added 4 tests for is_registered_device and edge cases
- **selfdrive.controls.lib**: Already at 95%+ for most files
  - MPC libraries (lat_mpc.py, long_mpc.py) require acados solver
- **system.loggerd**: Added uploader helper tests
  - get_directory_sort, listdir_by_creation, clear_locks, FakeRequest/Response
- **New tests added**: 21

### Session 4 (Selfdrived & Locationd Coverage)
- **selfdrive.selfdrived**: events.py: 85% → 91% (+6%)
  - Added 13 tests for mici device paths, callbacks, and edge cases
  - Tests: create_alerts with callback_args=None, NoEntryAlert/StartupAlert on mici
  - Tests: soft_disable_alert, user_soft_disable_alert callbacks
  - Tests: startup_master_alert with REPLAY env var
- **selfdrive.locationd**: torqued.py: 57% → 61% (+4%)
  - Added 7 tests for LinAlgError handling, torque tuning, get_msg options
- **system.hardware**: power_monitoring.py: 90% → 97% (+7%)
  - Added 3 tests for exception handling and edge cases
- **tools.lib**: filereader.py: 68% → 93% (+25%)
  - Added 14 tests for resolve_name, file_exists, DiskFile, FileReader, internal_source_available
- **tools.lib**: url_file.py: 56% → 99% (+43%)
  - Added 12 tests for MaxRetryError, multipart responses, read_aux, get_length caching, read caching
- **cereal.messaging**: FrequencyTracker: 18% → 35% (+17%)
  - Added 15 tests for init, record_recv_time, valid property
- **cereal.messaging**: Helper functions: 35% → 43% (+8%)
  - Added 17 tests for new_message, log_from_bytes, pub_sock, sub_sock
- **New tests added**: 81
