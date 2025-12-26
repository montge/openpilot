## MODIFIED Requirements

### Requirement: Reproducible Analysis Scripts
All analysis configurations and scripts SHALL be version controlled, reproducible, and integrated into CI.

#### Scenario: CI runs analysis on PR
- **GIVEN** a pull request is opened
- **WHEN** the MISRA analysis workflow runs
- **THEN** it produces consistent results across runs
- **AND** new violations are reported in the PR
- **AND** a baseline comparison shows only new issues

#### Scenario: Baseline tracking
- **GIVEN** the MISRA analysis has run
- **WHEN** the baseline file is updated
- **THEN** subsequent runs only report new violations
- **AND** the total violation count is tracked over time
