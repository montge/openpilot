# Change: Add Code Quality Gates

## Why

openpilot is safety-critical automotive software. Currently there are no enforced coverage thresholds or quality gates in CI. Establishing measurable quality targets (90% line coverage, 80% branch coverage) with CI enforcement will:

1. Prevent quality regression as new code is added
2. Identify untested safety-critical paths
3. Establish a foundation for contributing bug fixes to upstream
4. Align with automotive industry practices (ISO 26262, MISRA)

## What Changes

- **Coverage Infrastructure**: Configure pytest-cov (Python) and llvm-cov/gcov (C++) with unified reporting
- **Coverage Thresholds**: Enforce 90% line coverage, 80% branch coverage via CI gates
- **MISRA Integration**: Add MISRA analysis (cppcheck + clang-tidy-automotive) to CI
- **Quality Dashboard**: SonarCloud integration for tracking trends
- **Incremental Approach**: Start with current baseline, ratchet up thresholds over time

## Impact

- Affected specs: `misra-analysis` (modify), `code-quality` (new)
- Affected code:
  - `.github/workflows/` - CI workflow updates
  - `pyproject.toml` - coverage configuration
  - `scripts/lint/` - MISRA scripts
  - Test files across `selfdrive/`, `system/`, `common/`
- Languages: Python (~404 files), C++ (~86 files)

## Scope

Bug fixes discovered through improved coverage will be contributed to upstream (commaai/openpilot). The full quality infrastructure stays in the fork.
