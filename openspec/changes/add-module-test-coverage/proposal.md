# Change: Add Module Test Coverage

## Why
Several core Python modules have test coverage below the 90% target established in the code-quality spec. This creates risk for regressions and makes refactoring difficult. Systematic coverage improvements will strengthen code quality and enable safer changes.

## What Changes
- Add unit tests to modules currently below 90% line coverage
- Prioritize by coverage gap and module criticality
- Target modules:
  - `tools.lib` (50% → 90%)
  - `system.athena` (58% → 90%)
  - `system.manager` (63% → 90%)
  - `selfdrive.monitoring` (67% → 90%)
  - `selfdrive.controls` (78% → 90%)
  - `selfdrive.selfdrived` (80% → 90%)
  - `system.loggerd` (82% → 90%)

## Impact
- Affected specs: code-quality
- Affected code: Test files in each module's `tests/` directory
- No changes to production code behavior
