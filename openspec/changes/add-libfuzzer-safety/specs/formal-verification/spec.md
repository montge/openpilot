# formal-verification Specification Delta

## ADDED Requirements

### Requirement: libFuzzer Continuous Fuzzing
The CI pipeline SHALL run libFuzzer-based fuzzing on safety-critical C code.

#### Scenario: Fuzz harness exercises safety TX hook
- **GIVEN** the fuzz_safety_tx_hook.c harness
- **WHEN** libFuzzer generates random CAN messages
- **THEN** no crashes, memory errors, or undefined behavior occurs
- **AND** ASAN/UBSAN sanitizers detect no violations

#### Scenario: Fuzz harness exercises torque validation
- **GIVEN** the fuzz_torque_cmd.c harness
- **WHEN** libFuzzer generates random torque commands
- **THEN** bounds checking rejects out-of-range values
- **AND** no integer overflow or underflow occurs

#### Scenario: Fuzzing runs on safety code changes
- **GIVEN** a PR modifying files in opendbc/safety/
- **WHEN** the CI pipeline runs
- **THEN** fuzzing executes for minimum corpus iteration
- **AND** PR fails if any sanitizer violation is detected
