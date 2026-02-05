# code-quality Specification

## Purpose
Enforce code quality standards including test coverage thresholds, linting rules, and static analysis across the openpilot codebase.
## Requirements
### Requirement: Python Coverage Enforcement
The CI pipeline SHALL enforce minimum Python code coverage thresholds.

#### Scenario: PR with sufficient coverage passes
- **GIVEN** a pull request with Python changes
- **WHEN** the coverage workflow runs
- **THEN** the PR passes if line coverage >= 90% and branch coverage >= 80%

#### Scenario: PR with coverage regression fails
- **GIVEN** a pull request that reduces coverage below the threshold
- **WHEN** the coverage workflow runs
- **THEN** the PR fails with a clear message indicating the coverage gap

### Requirement: C++ Coverage Reporting
The CI pipeline SHALL generate comprehensive C++ code coverage reports for all test directories.

#### Scenario: C++ coverage report generated for all components
- **GIVEN** a pull request with C++ changes
- **WHEN** the coverage workflow runs
- **THEN** llvm-cov generates coverage for all C++ test directories:
  - common/tests/
  - system/loggerd/tests/
  - system/camerad/test/
  - selfdrive/pandad/tests/
  - tools/cabana/tests/
  - tools/replay/tests/
- **AND** the report is uploaded to Codecov with component flags

#### Scenario: C++ coverage threshold enforced
- **GIVEN** a pull request with C++ changes
- **WHEN** C++ coverage drops below the baseline threshold (15%)
- **THEN** the PR fails with a coverage report
- **AND** the failure message indicates which components are below threshold
- **NOTE** Target threshold is 80%; baseline set to prevent regression while tests are added

#### Scenario: Component-level C++ coverage visible
- **GIVEN** the C++ coverage workflow has completed
- **WHEN** a developer views the Codecov dashboard
- **THEN** they see separate coverage metrics for:
  - cpp-core (common, system, selfdrive/pandad)
  - cpp-tools (tools/cabana, tools/replay)
- **AND** each component shows its own coverage percentage

### Requirement: MISRA Analysis in CI
The CI pipeline SHALL run MISRA static analysis on C/C++ code.

#### Scenario: MISRA analysis runs on PR
- **GIVEN** a pull request with C/C++ changes
- **WHEN** the MISRA workflow runs
- **THEN** cppcheck-misra analyzes changed files
- **AND** clang-tidy-automotive analyzes changed files (Not yet enabled in CI - planned)
- **AND** new violations are reported in the PR

#### Scenario: Differential MISRA reporting
- **GIVEN** a pull request introducing new MISRA violations
- **WHEN** the MISRA workflow runs
- **THEN** only new violations (not existing baseline) are reported
- **AND** the PR is marked with warnings (not blocked by default)

### Requirement: Quality Dashboard
The project SHALL provide a quality dashboard showing coverage and analysis trends.

#### Scenario: Developer views quality metrics
- **GIVEN** a developer wants to check quality metrics
- **WHEN** they visit SonarCloud or Codecov dashboards
- **THEN** they see current coverage percentages
- **AND** they see coverage trend over time
- **AND** they see MISRA violation counts
- **NOTE** MISRA findings are uploaded as GitHub Actions artifacts, not to SonarCloud/Codecov dashboards.

### Requirement: pytest-mock for Test Mocking
All test files SHALL use pytest-mock's `mocker` fixture instead of `unittest.mock`.

#### Scenario: Test file uses pytest-mock patterns
- **GIVEN** a test file that requires mocking
- **WHEN** the file is checked by ruff with TID251 rule
- **THEN** no `unittest` or `unittest.mock` imports are present
- **AND** mocking is done via the `mocker` fixture parameter

#### Scenario: Pre-commit hooks pass for test files
- **GIVEN** a developer modifies a test file
- **WHEN** they run `pre-commit run` or commit
- **THEN** the ruff TID251 check passes
- **AND** no manual `--no-verify` bypass is needed

### Requirement: Lateral Control Implementation Tests
All lateral control implementations SHALL have dedicated unit tests with >= 90% coverage.

#### Scenario: LatControlAngle has comprehensive tests
- **GIVEN** the `latcontrol_angle.py` module
- **WHEN** pytest runs with coverage
- **THEN** line coverage is >= 90%
- **AND** tests cover initialization, active/inactive behavior, and saturation detection

#### Scenario: LatControlPID has comprehensive tests
- **GIVEN** the `latcontrol_pid.py` module
- **WHEN** pytest runs with coverage
- **THEN** line coverage is >= 90%
- **AND** tests cover initialization, PID calculation, feedforward, and integrator freeze

#### Scenario: LatControlTorque has comprehensive tests
- **GIVEN** the `latcontrol_torque.py` module
- **WHEN** pytest runs with coverage
- **THEN** line coverage is >= 90%
- **AND** tests cover initialization, torque calculation, jerk filtering, and friction compensation

### Requirement: Module-Level Coverage Targets
Each Python module SHALL maintain minimum test coverage thresholds appropriate to its criticality.

#### Scenario: Core module meets coverage threshold
- **GIVEN** a core Python module (selfdrive/, system/, common/)
- **WHEN** pytest runs with coverage measurement
- **THEN** line coverage is >= 80%
- **AND** branch coverage is >= 80%
- **NOTE** The enforced minimum in codecov.yml is 80%. The aspirational target is 90%.

#### Scenario: Tools module meets coverage threshold
- **GIVEN** a tools module (tools/lib/, tools/replay/)
- **WHEN** pytest runs with coverage measurement
- **THEN** line coverage is >= 80%
- **AND** branch coverage is >= 80%
- **NOTE** The enforced minimum in codecov.yml is 80%. The aspirational target is 90%.

#### Scenario: Coverage report identifies gaps
- **GIVEN** a module with coverage below threshold
- **WHEN** the developer runs coverage analysis
- **THEN** uncovered lines and branches are clearly identified
- **AND** the report provides actionable guidance for improvement

#### Scenario: Safety-critical modules have higher coverage
- **GIVEN** a safety-critical module (selfdrive/controls/, selfdrive/monitoring/)
- **WHEN** pytest runs with coverage measurement
- **THEN** line coverage is >= 85%
- **AND** all safety-related code paths have explicit test coverage
- **NOTE** The enforced minimum in codecov.yml is 85% for selfdrive/monitoring/. The aspirational target is 95%.
