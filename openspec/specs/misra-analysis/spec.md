# misra-analysis Specification

## Purpose
TBD - created by archiving change add-misra-analysis. Update Purpose after archive.
## Requirements
### Requirement: Baseline MISRA Analysis Configuration
The project SHALL provide a cppcheck configuration for MISRA C:2012 baseline analysis.

#### Scenario: Developer runs baseline MISRA analysis
Given the developer has cppcheck installed with the MISRA addon
When they run `scripts/lint/cppcheck-misra.sh`
Then cppcheck analyzes selfdrive/, system/, and common/ directories
And outputs findings to `cppcheck-misra-report.txt`
And excludes third_party, submodules, and generated code

### Requirement: MISRA Analysis Configuration
The project SHALL provide a configuration for MISRA C:2025 and C++:2023 analysis using clang-tidy-automotive.

#### Scenario: Developer runs MISRA analysis
Given the developer has clang-tidy-automotive built at a known path
When they run `scripts/lint/clang-tidy-misra.sh`
Then the automotive fork analyzes selfdrive/, system/, and common/ directories
And outputs findings to `misra-report.txt`
And uses automotive-* and automotive-cpp23-* checks

### Requirement: Analysis Comparison Report
The project SHALL provide tooling to compare baseline and MISRA analysis results.

#### Scenario: Developer generates comparison report
Given both baseline-report.txt and misra-report.txt exist
When they run `scripts/lint/compare-analysis.sh`
Then a comparison report is generated showing:
- Findings unique to baseline analysis
- Findings unique to MISRA analysis
- Common findings between both
- Summary statistics

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
