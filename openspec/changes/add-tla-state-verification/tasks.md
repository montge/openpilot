# Tasks: add-tla-state-verification

## Setup
- [ ] Create verification/tlaplus/ directory structure
- [ ] Add TLA+ tools download to CI
- [ ] Create SelfDrived.cfg model configuration

## Core Specification
- [ ] Write TLA+ module for state machine states
- [ ] Define event types matching events.py
- [ ] Model state transition logic from state.py
- [ ] Add soft_disable_timer countdown logic
- [ ] Model initialization sequence

## Safety Invariants
- [ ] TypeInvariant: all variables have valid types
- [ ] DisableAlwaysHonored: USER_DISABLE/IMMEDIATE_DISABLE reach disabled
- [ ] NoEntryBlocks: NO_ENTRY prevents engagement
- [ ] TimerConsistency: timer > 0 only in softDisabling

## Temporal Properties
- [ ] SoftDisableProgress: softDisabling eventually resolves
- [ ] AlertBeforeDisable: driver alerted before system disables
- [ ] NoDeadlock: system never gets stuck
- [ ] EventuallyStable: system reaches disabled or enabled

## CI Integration
- [ ] Create `.github/workflows/tlaplus.yml` workflow
- [ ] Configure path filters for tlaplus/ and state.py changes
- [ ] Add step to download tla2tools.jar
- [ ] Run TLC with invariants and properties

## Validation
- [ ] TLC passes all invariants with no counterexamples
- [ ] TLC verifies temporal properties
- [ ] Manually inject known bugs and verify TLC catches them
- [ ] Document state space size and verification time

## Documentation
- [ ] Write TLA_VERIFICATION.md with usage instructions
- [ ] Add comments to TLA+ spec explaining each property
- [ ] Document how to extend spec for new states/events
