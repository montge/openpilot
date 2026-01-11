# Change: Add Test Coverage Requirements for Algorithm Harness

## Why

The algorithm test harness is safety-adjacent code that influences how control algorithms are tested and validated. High test coverage ensures:

1. **Reliability**: Framework code is thoroughly tested before use in algorithm validation
2. **Regression Prevention**: Changes don't break existing functionality
3. **Documentation**: Tests serve as executable documentation of expected behavior
4. **Confidence**: Users can trust harness results for algorithm comparison

Target: **90%+ code coverage** for all algorithm harness modules.

## What Changes

- **Add Coverage Configuration** (`.coveragerc` or `pyproject.toml`):
  - Configure coverage.py for algorithm harness
  - Set minimum coverage thresholds
  - Exclude test files from coverage calculation

- **Add Coverage CI Job** (`.github/workflows/`):
  - Run coverage on PR for algorithm harness changes
  - Fail PR if coverage drops below threshold
  - Generate and upload coverage reports

- **Add Missing Tests**:
  - Ensure all modules have >90% coverage
  - Add edge case tests
  - Add integration tests

- **Add Coverage Badges**:
  - Display coverage status in README
  - Track coverage trends over time

## Impact

- Affected specs: New `test-coverage` capability
- Affected code:
  - `selfdrive/controls/lib/tests/algorithm_harness/` - All modules
  - `.github/workflows/` - New CI job
  - `pyproject.toml` or `.coveragerc` - Coverage configuration
- Dependencies: coverage.py, pytest-cov
- **No runtime impact** (testing infrastructure only)
