# Tasks: add-cbmc-verification

## Setup
- [x] Create verification/cbmc/ directory structure
- [x] Add CBMC installation to CI dependencies
- [x] Create basic Makefile for running CBMC locally

## Core Harnesses
- [x] Write harness for `safety_tx_hook()` controls_allowed invariant
- [x] Write harness for `safety_tx_hook()` relay_malfunction invariant
- [x] Write harness for `steer_torque_cmd_checks()` bounds verification
- [ ] Write harness for `steer_angle_cmd_checks()` bounds verification (deferred - similar pattern to torque)
- [x] Write harness for `longitudinal_accel_checks()` bounds verification

## Overflow Verification
- [ ] Add overflow checks for torque bit-packing in Toyota (future enhancement)
- [ ] Add overflow checks for torque bit-packing in Honda (future enhancement)
- [ ] Verify `to_signed()` helper handles all bit widths correctly (future enhancement)

## State Machine Verification
- [ ] Model `controls_allowed` state transitions (future enhancement)
- [ ] Verify monotonicity of `relay_malfunction` flag (future enhancement)
- [ ] Verify message counter/checksum validation logic (future enhancement)

## CI Integration
- [x] Create `.github/workflows/cbmc.yml` workflow
- [ ] Add CBMC to pre-commit hooks (optional, may be slow) - skipped, CI sufficient
- [x] Configure path filters for opendbc/safety/ changes only

## Documentation
- [x] Write CBMC_VERIFICATION.md with usage instructions
- [x] Document how to add new properties
- [x] Add examples for manufacturer mode verification

## Validation
- [x] All core harnesses pass syntax checking (gcc -fsyntax-only)
- [x] CI workflow created and configured (runs on opendbc/safety/ changes)
- [x] Core harnesses validated locally

## Files Created
- `verification/cbmc/stubs.h` - Hardware stubs for model checking
- `verification/cbmc/harness_controls_allowed.c` - P1 verification
- `verification/cbmc/harness_relay_malfunction.c` - P2 verification
- `verification/cbmc/harness_torque_bounds.c` - P3 verification
- `verification/cbmc/harness_longitudinal.c` - P4 verification
- `verification/cbmc/Makefile` - Build and run verifications
- `verification/cbmc/CBMC_VERIFICATION.md` - Documentation
- `.github/workflows/cbmc.yml` - CI workflow
