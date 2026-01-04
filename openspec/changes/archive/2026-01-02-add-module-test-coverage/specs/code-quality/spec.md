## ADDED Requirements

### Requirement: Module-Level Coverage Targets
Each Python module SHALL maintain minimum test coverage thresholds appropriate to its criticality.

#### Scenario: Core module meets coverage threshold
- **GIVEN** a core Python module (selfdrive/, system/, common/)
- **WHEN** pytest runs with coverage measurement
- **THEN** line coverage is >= 90%
- **AND** branch coverage is >= 80%

#### Scenario: Tools module meets coverage threshold
- **GIVEN** a tools module (tools/lib/, tools/replay/)
- **WHEN** pytest runs with coverage measurement
- **THEN** line coverage is >= 90%
- **AND** branch coverage is >= 80%

#### Scenario: Coverage report identifies gaps
- **GIVEN** a module with coverage below threshold
- **WHEN** the developer runs coverage analysis
- **THEN** uncovered lines and branches are clearly identified
- **AND** the report provides actionable guidance for improvement

#### Scenario: Safety-critical modules have higher coverage
- **GIVEN** a safety-critical module (selfdrive/controls/, selfdrive/monitoring/)
- **WHEN** pytest runs with coverage measurement
- **THEN** line coverage is >= 95%
- **AND** all safety-related code paths have explicit test coverage
