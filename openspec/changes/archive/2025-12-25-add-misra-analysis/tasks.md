# Tasks: Add MISRA Static Analysis

## Phase 1: Baseline MISRA Analysis with cppcheck

- [x] **1.1** Create cppcheck configuration for MISRA analysis
  - Configure MISRA C:2012 addon
  - Set up suppressions for third-party code
  - Script: `scripts/lint/cppcheck-misra.sh`

- [x] **1.2** Run cppcheck MISRA analysis on core directories
  - Target: selfdrive/, system/, common/
  - Exclude: third_party/, *_repo/, generated files
  - Output: `reports/cppcheck-misra-report.txt`
  - Result: **3,255 findings**

- [x] **1.3** Categorize and document cppcheck findings
  - Top rules: 15.5 (533), 8.4 (422), 12.1 (175), 10.4 (173), 11.9 (170)
  - Summary in `reports/misra-comparison-report.md`

## Phase 2: MISRA Analysis with clang-tidy-automotive

- [x] **2.1** Verify clang-tidy-automotive build is functional
  - Version: LLVM 20.1.8
  - Available: 172 automotive checks (65 C:2025, 46 C++:2023)

- [x] **2.2** Create MISRA-focused `.clang-tidy` configuration
  - Script: `scripts/lint/clang-tidy-misra.sh`
  - Enabled: automotive-* (excluding automotive-cpp23-req-8.3.1 due to bug)

- [x] **2.3** Run MISRA analysis on same directories as baseline
  - Output: `reports/clang-tidy-misra-report.txt`
  - Result: **59,791 findings** (in openpilot source files)

- [x] **2.4** Generate comparison report
  - Report: `reports/misra-comparison-report.md`
  - Bug found: automotive-cpp23-req-8.3.1 crash (Issue #5)

## Phase 3: Integration and Documentation

- [x] **3.1** Create analysis scripts for reproducibility
  - `scripts/lint/cppcheck-misra.sh` - cppcheck MISRA C:2012
  - `scripts/lint/clang-tidy-misra.sh` - clang-tidy-automotive MISRA C:2025/C++:2023

- [x] **3.2** Document findings and recommendations
  - Comparison report: `reports/misra-comparison-report.md`
  - Tool issues: `reports/clang-tidy-automotive-issues.md`
  - GitHub issue created: montge/clang-tidy-automotive#5

- [ ] **3.3** (Optional) Create GitHub Actions workflow
  - Run MISRA checks on PRs
  - Report new violations
  - Track trend over time

## Validation
- [x] Both analyses complete without errors
- [x] Reports are generated and readable
- [x] Comparison clearly shows differences between toolsets
