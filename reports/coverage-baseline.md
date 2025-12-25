# Coverage Baseline Report

**Date**: 2025-12-25
**Branch**: feature/test-ci-pipelines (develop)

## Executive Summary

| Metric | Python | C++ | Target | Gap |
|--------|--------|-----|--------|-----|
| Line Coverage | **33%** | TBD | 90% | -57% |
| Branch Coverage | TBD | TBD | 80% | TBD |
| Files Analyzed | ~404 | ~86 | - | - |
| Lines Covered | 9,189 / 24,777 | TBD | - | - |

**Key Finding**: Current Python coverage is 33%, significantly below the 90% target. Safety-critical modules average 50% coverage.

## Safety-Critical Modules

Based on `docs/SAFETY.md`, the following modules are safety-critical:

| Module | Purpose | Priority | Current | Target |
|--------|---------|----------|---------|--------|
| `selfdrive/monitoring/helpers.py` | Driver monitoring | **Critical** | 87% | 95% |
| `selfdrive/selfdrived/helpers.py` | Actuation checks | **Critical** | 36% | 95% |
| `selfdrive/selfdrived/selfdrived.py` | State machine | **Critical** | 7% | 95% |
| `selfdrive/controls/controlsd.py` | Vehicle control | **High** | 15% | 90% |
| `selfdrive/pandad/pandad.py` | CAN bus | **High** | 8% | 90% |
| `opendbc/safety/` | Safety model | **Critical** | (submodule) | 95% |

**Safety-Critical Average: 50%** (target: 95%)

### Urgent Gaps

| File | Coverage | Gap | Risk |
|------|----------|-----|------|
| `selfdrived.py` | 7% | -88% | State machine largely untested |
| `pandad.py` | 8% | -82% | CAN communication untested |
| `controlsd.py` | 15% | -75% | Vehicle control untested |
| `helpers.py` (selfdrived) | 36% | -59% | Actuation checks partially tested |

### Coverage Targets by Priority

| Priority | Line Coverage | Branch Coverage |
|----------|---------------|-----------------|
| Critical | 95% | 90% |
| High | 90% | 85% |
| Standard | 80% | 70% |

## Python Coverage Details

### Per-Module Breakdown

| Module | Statements | Covered | Coverage | Target |
|--------|------------|---------|----------|--------|
| selfdrive/ | 13,826 | 5,267 | **34%** | 90% |
| system/ | 9,775 | 3,157 | **27%** | 90% |
| common/ | 1,176 | 765 | **61%** | 90% |
| **TOTAL** | **24,777** | **9,189** | **33%** | **90%** |

### Least Covered Critical Files

| File | Statements | Coverage | Notes |
|------|------------|----------|-------|
| `system/ui/widgets/label.py` | 472 | 8% | UI component |
| `selfdrive/selfdrived/selfdrived.py` | 358 | 7% | **Critical** |
| `selfdrive/pandad/pandad.py` | 122 | 8% | **Critical** |
| `selfdrive/controls/controlsd.py` | 158 | 15% | **Critical** |
| `system/ui/lib/wrap_text.py` | 66 | 6% | UI helper |

## C++ Coverage Details

C++ coverage requires build instrumentation. Current status:
- [ ] llvm-cov instrumentation added to SConstruct
- [ ] C++ test execution with coverage
- [ ] Report generation

## Test Statistics

From baseline run (2025-12-25):
- **Passed**: 462
- **Failed**: 2 (infrastructure issues, not code bugs)
- **Skipped**: 113 (tici/slow markers)
- **Duration**: ~3 minutes
- **Coverage collected**: Yes

### Failed Tests (Infrastructure)
- `system/loggerd/tests/test_loggerd.py::TestLoggerd::test_init_data_values` - timeout
- `system/athena/tests/test_athenad.py::TestAthenadMethods::test_upload_handler_timeout` - race condition

## Roadmap to 90% Coverage

### Phase 1: Quick Wins (33% → 50%)
Focus on files with many statements but low coverage:
1. Add tests for `system/` UI components (currently 27%)
2. Improve `common/` utility coverage (currently 61%)

### Phase 2: Safety-Critical (50% → 70%)
Priority testing for safety modules:
1. `selfdrive/selfdrived/selfdrived.py` (7% → 80%)
2. `selfdrive/controls/controlsd.py` (15% → 80%)
3. `selfdrive/pandad/pandad.py` (8% → 80%)

### Phase 3: Full Coverage (70% → 90%)
1. Add integration tests for remaining modules
2. Enable branch coverage enforcement
3. Per-module coverage gates in CI

## Next Steps

1. [x] Measure Python coverage baseline
2. [ ] Add llvm-cov instrumentation for C++
3. [ ] Set initial fail_under at 33% (prevent regression)
4. [ ] Prioritize tests for safety-critical modules
5. [ ] Configure per-module coverage in CI
