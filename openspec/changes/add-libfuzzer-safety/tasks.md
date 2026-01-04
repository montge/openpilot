# Tasks: add-libfuzzer-safety

## Setup
- [x] Create verification/fuzz/ directory structure
- [x] Add libFuzzer compilation flags to Makefile
- [x] Create stubs for hardware functions

## Fuzz Targets
- [x] Write fuzz_safety_tx_hook.c - CAN message TX validation
- [x] Write fuzz_can_rx.c - CAN message RX parsing
- [ ] Write fuzz_torque_cmd.c - Torque command bounds checking (future enhancement)
- [ ] Write fuzz_state_machine.c - Controls allowed state transitions (future enhancement)

## CI Integration
- [x] Create .github/workflows/fuzz.yml workflow
- [x] Configure corpus storage in CI artifacts
- [x] Add coverage-guided fuzzing with sanitizers (ASAN, UBSAN)

## Validation
- [x] All fuzz targets compile without errors
- [x] Fuzzing runs for minimum duration without crashes (3M+ iterations)
- [x] Document any bugs found during initial fuzzing (none found)

## Documentation
- [x] Write README.md with local usage instructions
- [x] Document how to add new fuzz targets
