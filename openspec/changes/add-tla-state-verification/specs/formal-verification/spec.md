# formal-verification Specification

## Purpose
Define formal verification requirements for safety-critical openpilot code using mathematically rigorous tools.

## ADDED Requirements

### Requirement: TLA+ State Machine Verification
The CI pipeline SHALL run TLA+ model checking on the selfdrived state machine specification.

#### Scenario: TLC verifies disable commands always honored
- **GIVEN** the SelfDrived.tla specification
- **WHEN** TLC runs with the DisableAlwaysHonored invariant
- **THEN** TLC proves: USER_DISABLE always transitions to disabled
- **AND** TLC proves: IMMEDIATE_DISABLE always transitions to disabled
- **AND** no counterexample is found

#### Scenario: TLC verifies NO_ENTRY blocks engagement
- **GIVEN** the SelfDrived.tla specification with NO_ENTRY event
- **WHEN** TLC runs with the NoEntryBlocks invariant
- **THEN** TLC proves: state cannot become enabled while NO_ENTRY is active
- **AND** no path exists from disabled to enabled with NO_ENTRY

#### Scenario: TLC verifies soft disable progress
- **GIVEN** the SelfDrived.tla specification in softDisabling state
- **WHEN** TLC checks the SoftDisableProgress temporal property
- **THEN** TLC proves: softDisabling eventually leads to disabled or enabled
- **AND** no infinite loop in softDisabling state

#### Scenario: TLC runs on state machine changes
- **GIVEN** a PR modifying selfdrive/selfdrived/state.py or verification/tlaplus/
- **WHEN** the CI pipeline runs
- **THEN** TLC verification executes on the specification
- **AND** PR fails if any invariant is violated
