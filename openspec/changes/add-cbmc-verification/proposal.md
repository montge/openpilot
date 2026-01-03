# add-cbmc-verification

## Summary
Add CBMC (C Bounded Model Checker) to verify safety-critical properties in the opendbc safety code. CBMC can prove absence of runtime errors and verify safety invariants in C code without requiring new specification languages.

## Motivation
The opendbc/safety/ code enforces critical vehicle safety constraints:
- `controls_allowed` state machine (prevents unauthorized actuation)
- Torque/angle limits per ISO 11270/15622
- Relay malfunction detection
- Message validation (checksums, counters, timeouts)

Currently verified by: MISRA static analysis + unit tests. Neither can prove invariants hold for all inputs.

CBMC can prove:
- No integer overflow in torque calculations
- `controls_allowed = false` blocks all TX messages
- Safety limits are never exceeded
- State machine invariants hold

## Scope

### In Scope
- CBMC verification of `safety.h`, `lateral.h`, `longitudinal.h`
- Verification of 2-3 manufacturer modes (Toyota, Honda) as representative samples
- CI integration to run CBMC on safety code changes
- Property annotations using CBMC `__CPROVER_assert()`

### Out of Scope
- All 22 manufacturer modes (can be added incrementally)
- VisionIPC or messaging code (separate proposal)
- Real-time timing verification
- Python code verification

## Acceptance Criteria
- [ ] CBMC proves no integer overflow in `steer_torque_cmd_checks()`
- [ ] CBMC proves `controls_allowed = false` implies no TX message allowed
- [ ] CBMC proves torque limits respected for bounded inputs
- [ ] CI workflow runs CBMC on PR changes to opendbc/safety/
- [ ] Documentation for adding new CBMC properties

## Dependencies
- CBMC installation (apt-get install cbmc or brew install cbmc)
- opendbc safety code structure

## References
- [CBMC Documentation](https://www.cprover.org/cbmc/)
- opendbc/safety/*.h (~6,090 lines of safety-critical C code)
- docs/SAFETY.md (current safety approach)
