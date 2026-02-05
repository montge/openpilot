# formal-verification Specification

## Purpose
Define formal verification requirements for safety-critical openpilot code using mathematically rigorous tools that can prove properties hold for all inputs, not just tested cases.
## Requirements

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
- **WHEN** TLC runs with the NoEntryBlocksEngagement invariant
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

### Requirement: SPIN Protocol Verification
The CI pipeline SHALL run SPIN model checking on the msgq messaging protocol.

#### Scenario: SPIN verifies no message loss for valid readers
- **GIVEN** the msgq_protocol.pml Promela model
- **WHEN** SPIN runs with the `no_stale_reads` LTL property
- **THEN** SPIN proves: valid readers receive all published messages
- **AND** no counterexample shows a stale read

#### Scenario: SPIN verifies reader eviction correctness
- **GIVEN** the msgq_protocol.pml Promela model with slow readers
- **WHEN** SPIN runs with the `reader0_progress` property
- **THEN** SPIN proves: readers make progress and are not starved
- **AND** eviction happens before data is overwritten

#### Scenario: SPIN detects known race conditions
- **GIVEN** the msgq_protocol.pml model with race condition scenarios
- **WHEN** SPIN runs exhaustive state exploration
- **THEN** SPIN finds: new-reader-during-wraparound race (known issue)
- **AND** counterexample shows the problematic interleaving

#### Scenario: SPIN runs on protocol changes
- **GIVEN** a PR modifying msgq_repo/msgq/msgq.cc or verification/spin/
- **WHEN** the CI pipeline runs
- **THEN** SPIN verification executes on the protocol model
- **AND** PR fails if safety properties are violated

### Requirement: libFuzzer Continuous Fuzzing
The CI pipeline SHALL run libFuzzer-based fuzzing on safety-critical C code.

#### Scenario: Fuzz harness exercises safety TX hook
- **GIVEN** the fuzz_safety_tx_hook.c harness
- **WHEN** libFuzzer generates random CAN messages
- **THEN** no crashes, memory errors, or undefined behavior occurs
- **AND** ASAN/UBSAN sanitizers detect no violations

#### Scenario: Fuzz harness exercises torque validation
- **GIVEN** the fuzz_safety_tx_hook.c harness
- **WHEN** libFuzzer generates random torque commands
- **THEN** bounds checking rejects out-of-range values
- **AND** no integer overflow or underflow occurs

#### Scenario: Fuzzing runs on safety code changes
- **GIVEN** a PR modifying files in opendbc/safety/
- **WHEN** the CI pipeline runs
- **THEN** fuzzing executes for minimum corpus iteration
- **AND** PR fails if any sanitizer violation is detected
