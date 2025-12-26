## MODIFIED Requirements

### Requirement: Python Coverage Enforcement
The CI pipeline SHALL enforce minimum Python code coverage thresholds with strict quality gates.

#### Scenario: PR with sufficient coverage passes
- **GIVEN** a pull request with Python changes
- **WHEN** the coverage workflow runs
- **THEN** the PR passes if project coverage >= 90%
- **AND** each component has coverage >= 80%

#### Scenario: PR with coverage regression fails
- **GIVEN** a pull request that reduces coverage below the threshold
- **WHEN** the coverage workflow runs
- **THEN** the PR fails with a clear message indicating the coverage gap
- **AND** the PR cannot be merged until coverage is improved

#### Scenario: Coverage thresholds are configurable per component
- **GIVEN** the codecov.yml configuration file
- **WHEN** a developer reviews coverage requirements
- **THEN** they can see the 90% project target
- **AND** they can see 80% minimum per-component targets
- **AND** critical modules (controls, car, selfdrived) have higher targets

## ADDED Requirements

### Requirement: Pre-commit Hook Standardization
All developers SHALL use pre-commit hooks for consistent code quality checks.

#### Scenario: Developer installs pre-commit hooks
- **GIVEN** a developer clones the repository
- **WHEN** they run `pre-commit install`
- **THEN** hooks are installed for commit-time checks
- **AND** ruff, mypy, codespell, and openspec-validate run on commits

#### Scenario: Pre-commit catches issues before commit
- **GIVEN** a developer has pre-commit installed
- **WHEN** they commit code with linting issues
- **THEN** the commit is blocked with clear error messages
- **AND** auto-fixable issues (ruff format) are fixed automatically

#### Scenario: Developer can skip slow checks when needed
- **GIVEN** a developer needs to make a quick commit
- **WHEN** they run `SKIP=mypy git commit -m "message"`
- **THEN** mypy is skipped for that commit
- **AND** CI will still run the full checks

### Requirement: Coverage Quality Dashboard
The project SHALL provide visibility into coverage metrics and trends.

#### Scenario: Developer views coverage on PR
- **GIVEN** a pull request is opened
- **WHEN** coverage checks complete
- **THEN** Codecov posts a comment with coverage diff
- **AND** the comment shows which files changed coverage
- **AND** overall coverage percentage is visible

#### Scenario: Coverage trends are tracked
- **GIVEN** the project has Codecov configured
- **WHEN** a developer visits the Codecov dashboard
- **THEN** they see coverage trends over time
- **AND** they can identify modules needing more tests
