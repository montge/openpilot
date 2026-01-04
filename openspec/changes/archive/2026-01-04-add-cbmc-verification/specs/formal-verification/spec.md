# formal-verification Specification

## Purpose
Define formal verification requirements for safety-critical openpilot code using mathematically rigorous tools.

## ADDED Requirements

### Requirement: CBMC Bounded Model Checking
The CI pipeline SHALL run CBMC verification on safety-critical C code.

#### Scenario: CBMC verifies safety TX hook invariants
- **GIVEN** the opendbc/safety/safety.h code
- **WHEN** CBMC runs with the controls_allowed harness
- **THEN** CBMC proves: `controls_allowed = false` implies TX blocked
- **AND** CBMC proves: `relay_malfunction = true` implies TX blocked

#### Scenario: CBMC verifies torque bounds
- **GIVEN** the opendbc/safety/lateral.h code
- **WHEN** CBMC runs with the torque bounds harness
- **THEN** CBMC proves: accepted torque is within `[-max_torque, +max_torque]`
- **AND** CBMC proves: torque rate change within limits

#### Scenario: CBMC verifies no integer overflow
- **GIVEN** C code with bit-packing operations (shifts, masks)
- **WHEN** CBMC runs with `--signed-overflow-check`
- **THEN** no signed integer overflow is possible for valid CAN message inputs

#### Scenario: CBMC runs on safety code changes
- **GIVEN** a PR modifying files in opendbc/safety/
- **WHEN** the CI pipeline runs
- **THEN** CBMC verification executes on affected harnesses
- **AND** PR fails if any CBMC property is violated
