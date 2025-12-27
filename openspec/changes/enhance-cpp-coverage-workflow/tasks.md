# Tasks: Enhance C++ Coverage Workflow

## Overview

Expand C++ coverage from 2 to 6 test directories, raise threshold to 80%, and add component-level reporting.

## Phase 1: Expand Test Coverage

- [x] **1.1** Add `selfdrive/pandad/tests/` to coverage build
  - Update scons `--coverage` targets
  - Add binary to llvm-cov report generation
  - Verify tests pass in CI environment

- [x] **1.2** Add `tools/cabana/tests/` to coverage build
  - Requires Qt5 dependencies (already in workflow)
  - Update scons targets and llvm-cov binaries

- [x] **1.3** Add `tools/replay/tests/` to coverage build
  - Requires ffmpeg dependencies (already in workflow)
  - Update scons targets and llvm-cov binaries

- [x] **1.4** Add `system/camerad/test/` to coverage build
  - Check for additional dependencies (OpenCL, camera libs)
  - Update scons targets and llvm-cov binaries

## Phase 2: Raise Coverage Threshold

- [x] **2.1** Update threshold from 65% to 80%
  - Modify `cpp-coverage.yml` threshold check
  - Verify current coverage meets new threshold
  - If not, document gap and create follow-up for test additions

## Phase 3: Component-Level Reporting

- [x] **3.1** Add separate Codecov flags for C++ components
  - `cpp-core`: common/, system/, selfdrive/pandad/
  - `cpp-tools`: tools/cabana/, tools/replay/
  - Update codecov-action upload step with flags

- [x] **3.2** Update `codecov.yml` with C++ component targets
  - Add `cpp-core` and `cpp-tools` flag configurations
  - Set per-component thresholds (80% each)

## Phase 4: Validation

- [ ] **4.1** Run workflow on develop branch
  - Verify all tests execute successfully
  - Verify coverage report includes all components
  - Verify Codecov receives component-level data

- [ ] **4.2** Test PR coverage behavior
  - Create test PR with C++ changes
  - Verify coverage diff is calculated correctly
  - Verify threshold enforcement works

## Dependencies

- Phase 2 depends on Phase 1 (need coverage data before raising threshold)
- Phase 3 can run in parallel with Phase 2
- Phase 4 depends on all previous phases

## Validation Criteria

- All 6 test directories appear in coverage report
- Total C++ coverage >= 80%
- Codecov dashboard shows C++ component breakdown
- PRs with C++ coverage regression are blocked

## Implementation Notes

### Files Modified
- `.github/workflows/cpp-coverage.yml`: Extended to build and run all 6 C++ test directories, raised threshold to 80%, added component-level Codecov uploads
- `codecov.yml`: Added `cpp`, `cpp-core`, `cpp-tools` flags and component definitions with 80% targets

### Test Binaries Added
- `selfdrive/pandad/tests/test_pandad_usbprotocol`
- `tools/cabana/tests/test_cabana`
- `tools/replay/tests/test_replay`
- `system/camerad/test/test_ae_gray`
