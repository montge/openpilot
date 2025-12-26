# Tasks: Add Coverage Quality Gates

## 1. Verification
- [x] 1.1 Verify pre-commit is installed and hooks work locally
- [x] 1.2 Run full build with `scons -u -j$(nproc)` to verify compilation
- [x] 1.3 Run full lint including mypy to check current state

## 2. Coverage Threshold Configuration
- [x] 2.1 Create codecov.yml with threshold configuration (90% overall, 80% per-component)
- [x] 2.2 Update .github/workflows/codecov.yml to enforce thresholds
- [x] 2.3 Configure project-level and component-level targets

## 3. Pre-commit Standardization
- [x] 3.1 Document pre-commit installation in README or CONTRIBUTING (added to openspec/project.md)
- [x] 3.2 Verify all hooks run successfully on current codebase
- [ ] 3.3 Add coverage check to pre-push hooks (optional)

## 4. Documentation
- [x] 4.1 Update openspec/project.md with project conventions
- [x] 4.2 Document coverage requirements and quality gates
- [x] 4.3 Add developer onboarding instructions for quality tools

## 5. Cleanup
- [x] 5.1 Review MISRA baseline violations - determined generated code should be excluded (already in config)
- [x] 5.2 Merge PR #1 (SonarCloud configuration)
- [x] 5.3 Merge PR #2 (OpenSpec configuration)

## 6. Validation
- [x] 6.1 Run full test suite with coverage (Core: 31%, System: 34%, Tools: 28%)
- [ ] 6.2 Verify Codecov thresholds are enforced on a test PR
- [ ] 6.3 Confirm SonarCloud quality gate passes

## Coverage Results Summary
| Shard | Tests | Coverage | Notes |
|-------|-------|----------|-------|
| Core (controls, car, selfdrived, etc.) | 90 passed, 7 errors | 31% | Missing deps fixed mid-run |
| System | 80 passed, 5 failed, 4 errors | 34% | Some environment-specific failures |
| Tools | 13 passed, 2 failed, 4 errors | 28% | Simulator deps missing |

Current coverage is ~30%, well below the 90% target. This establishes the baseline for improvement.
