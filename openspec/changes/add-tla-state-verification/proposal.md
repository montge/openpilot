# add-tla-state-verification

## Summary
Add TLA+ specifications to formally verify the selfdrived state machine. TLA+ excels at verifying temporal properties like "driver is always alerted before system disengages" and proving state machine invariants hold across all possible event sequences.

## Motivation
The selfdrived state machine (state.py) controls vehicle engagement/disengagement:
- 5 states: disabled, preEnabled, enabled, softDisabling, overriding
- Complex event priority ordering (USER_DISABLE > IMMEDIATE_DISABLE > SOFT_DISABLE > ...)
- 3-second soft-disable timer for driver alerting
- Safety-critical: must never allow unintended engagement or silent disengagement

Current verification: Unit tests cover known scenarios but cannot prove properties hold for ALL event sequences.

TLA+ can prove:
- Disable commands ALWAYS honored from ANY state
- Driver ALWAYS alerted before system disengages (temporal property)
- Timer correctly counts down and triggers transition
- NO_ENTRY blocks ALL paths to engagement

## Scope

### In Scope
- TLA+ specification of selfdrived state machine
- Model checking with TLC for invariant verification
- Temporal property verification (liveness, safety)
- CI integration to run TLC on specification changes

### Out of Scope
- Panda safety code (separate CBMC proposal)
- Real-time timing verification (TLA+ doesn't model wall-clock time)
- Full selfdrived.py behavior (focus on state machine core)
- Messaging layer (separate SPIN proposal)

## Acceptance Criteria
- [ ] TLA+ spec models all 5 states and transitions
- [ ] TLC proves: USER_DISABLE/IMMEDIATE_DISABLE always reach disabled
- [ ] TLC proves: NO_ENTRY blocks engagement (never reach enabled with NO_ENTRY)
- [ ] TLC proves: softDisabling always leads to disabled OR enabled (no deadlock)
- [ ] CI workflow runs TLC on spec changes

## Dependencies
- TLA+ Toolbox or tla2tools.jar
- Java runtime for TLC model checker
- selfdrive/selfdrived/state.py (reference implementation)

## References
- [TLA+ Language Manual](https://lamport.azurewebsites.net/tla/tla.html)
- [Learn TLA+](https://learntla.com/)
- selfdrive/selfdrived/state.py (99 lines, state machine core)
- selfdrive/selfdrived/events.py (event type definitions)
