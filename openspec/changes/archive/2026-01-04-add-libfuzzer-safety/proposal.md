# Proposal: add-libfuzzer-safety

## Why

CBMC verifies properties hold for bounded inputs, but libFuzzer provides continuous runtime fuzzing that can discover edge cases and crashes that bounded model checking might miss. Fuzzing complements formal verification by exploring real execution paths with mutated inputs.

## What Changes

Add libFuzzer-based fuzzing targets for the safety-critical C code in opendbc/safety/. This creates fuzz harnesses that exercise:
- CAN message parsing and safety TX hook
- Torque command validation
- Relay malfunction detection
- State machine transitions

The fuzzing runs in CI using OSS-Fuzz integration or GitHub's code scanning.

## Scope

- Create fuzz harnesses in verification/fuzz/
- Add CI workflow for continuous fuzzing
- Integrate with existing CBMC verification

## Dependencies

- Clang compiler with libFuzzer support
- Existing safety code in opendbc/safety/
