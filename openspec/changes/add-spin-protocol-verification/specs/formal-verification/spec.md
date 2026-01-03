# formal-verification Specification

## Purpose
Define formal verification requirements for safety-critical openpilot code using mathematically rigorous tools.

## ADDED Requirements

### Requirement: SPIN Protocol Verification
The CI pipeline SHALL run SPIN model checking on the msgq messaging protocol.

#### Scenario: SPIN verifies no message loss for valid readers
- **GIVEN** the msgq_protocol.pml Promela model
- **WHEN** SPIN runs with the no_message_loss LTL property
- **THEN** SPIN proves: valid readers receive all published messages
- **AND** no counterexample shows a missed message

#### Scenario: SPIN verifies reader eviction correctness
- **GIVEN** the msgq_protocol.pml Promela model with slow readers
- **WHEN** SPIN runs with the eviction_respected property
- **THEN** SPIN proves: evicted readers stop accessing the buffer
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
