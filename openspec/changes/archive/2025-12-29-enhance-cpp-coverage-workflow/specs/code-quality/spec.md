# code-quality Spec Delta

## MODIFIED Requirements

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
