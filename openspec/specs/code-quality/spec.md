# code-quality Specification

## Purpose
TBD - created by archiving change add-code-quality-gates. Update Purpose after archive.
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
- **AND** clang-tidy-automotive analyzes changed files
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
