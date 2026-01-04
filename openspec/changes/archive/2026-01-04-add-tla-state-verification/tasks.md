# Tasks: add-tla-state-verification

## Setup
- [x] Create verification/tlaplus/ directory structure
- [x] Add TLA+ tools download to CI
- [x] Create SelfDrived.cfg model configuration

## Core Specification
- [x] Write TLA+ module for state machine states
- [x] Define event types matching events.py
- [x] Model state transition logic from state.py
- [x] Add soft_disable_timer countdown logic
- [x] Model initialization sequence

## Safety Invariants
- [x] TypeInvariant: all variables have valid types
- [x] DisableAlwaysHonored: USER_DISABLE/IMMEDIATE_DISABLE reach disabled
- [x] NoEntryBlocks: NO_ENTRY prevents engagement
- [x] TimerConsistency: timer > 0 only in softDisabling

## Temporal Properties
- [x] SoftDisableProgress: softDisabling eventually resolves
- [x] AlertBeforeDisable: driver alerted before system disables
- [x] NoDeadlock: system never gets stuck
- [x] EventuallyStable: system reaches disabled or enabled

## CI Integration
- [x] Create `.github/workflows/tlaplus.yml` workflow
- [x] Configure path filters for tlaplus/ and state.py changes
- [x] Add step to download tla2tools.jar
- [x] Run TLC with invariants and properties

## Validation
- [x] TLC passes all invariants with no counterexamples
- [x] TLC verifies safety properties (temporal properties deferred due to non-deterministic events)
- [ ] Manually inject known bugs and verify TLC catches them (future enhancement)
- [ ] Document state space size and verification time (future enhancement)

## Documentation
- [x] Write TLA_VERIFICATION.md with usage instructions
- [x] Add comments to TLA+ spec explaining each property
- [x] Document how to extend spec for new states/events
