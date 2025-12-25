# Tasks: Add Code Quality Gates

## Phase 1: Establish Baseline

- [x] **1.1** Measure current Python coverage baseline
  - Run `pytest --cov` on full test suite
  - Record line coverage, branch coverage, per-module breakdown
  - Document in `reports/coverage-baseline.md`
  - **Result: 33% line coverage (9,189/24,777 statements)**

- [ ] **1.2** Measure current C++ coverage baseline
  - Configure scons for coverage instrumentation
  - Run C++ tests with llvm-cov
  - Record baseline metrics

- [x] **1.3** Document safety-critical modules
  - Identify highest-priority paths (controls/, pandad/, safety/)
  - Set per-module coverage targets
  - **Result: Safety-critical modules at 50% avg (selfdrived 7%, pandad 8%, controlsd 15%)**

## Phase 2: Python Coverage Gates

- [x] **2.1** Enable pytest-cov fail_under in pyproject.toml
  - Start at current baseline (prevents regression)
  - Configure per-module thresholds for critical paths
  - **Result: fail_under = 33 in pyproject.toml**

- [x] **2.2** Update Codecov workflow
  - Add coverage threshold checks
  - Configure PR comments with coverage delta
  - **Result: Added coverage check step, codecov.yml config**

- [x] **2.3** Create coverage ratchet script
  - Script to update thresholds as coverage improves
  - Prevents accidental regression
  - **Result: scripts/ratchet-coverage.sh**

## Phase 3: C++ Coverage

- [x] **3.1** Add llvm-cov instrumentation to SConstruct
  - Enable -fprofile-instr-generate -fcoverage-mapping
  - Create coverage build target
  - **Result: Added --coverage option to scons**

- [x] **3.2** Create C++ coverage script
  - Build with coverage, run tests, generate report
  - Export to lcov format for Codecov
  - **Result: scripts/cpp-coverage.sh**

- [ ] **3.3** Create C++ coverage workflow
  - Add to CI pipeline
  - Upload to Codecov

- [ ] **3.4** Set C++ coverage thresholds
  - Start with baseline, ratchet up over time

## Phase 4: MISRA CI Integration

- [x] **4.1** Add cppcheck-misra to CI workflow
  - Run on PRs, report new violations only (differential)
  - Initially non-blocking (warnings)
  - **Result: .github/workflows/misra.yml**

- [ ] **4.2** Add clang-tidy-automotive to CI workflow
  - Configure as optional/advisory check
  - Report findings without blocking
  - **Note: Requires custom clang-tidy-automotive build (commented out)**

- [ ] **4.3** Create MISRA baseline file
  - Record current violation counts
  - Track improvement over time

## Phase 5: Quality Gates

- [ ] **5.1** Enable coverage gates (blocking)
  - Python: fail if coverage drops
  - C++: fail if coverage drops

- [ ] **5.2** Enable MISRA gates (optional)
  - cppcheck-misra: fail on new high-severity violations
  - clang-tidy-automotive: advisory only

- [ ] **5.3** Configure SonarCloud quality gates
  - Set coverage thresholds in SonarCloud
  - Configure PR decoration

## Validation

- [ ] All coverage workflows run successfully
- [ ] Coverage reports appear in Codecov
- [ ] MISRA reports generated without blocking PRs
- [ ] SonarCloud shows quality metrics
- [ ] PR with coverage regression is blocked
