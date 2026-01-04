# libFuzzer Safety Code Fuzzing

This directory contains libFuzzer-based fuzz harnesses for testing the safety-critical
C code in openpilot. Fuzzing complements the CBMC bounded model checking by testing
real execution paths with sanitizers.

## Quick Start

```bash
# Build all fuzz targets
make all

# Run quick fuzzing session (10s per target)
make run

# Build corpus over time (60s per target)
make corpus

# Clean build artifacts
make clean
```

## Fuzz Targets

### fuzz_safety_tx_hook

Tests the CAN TX safety validation logic:
- Message address allowlist checking
- Controls_allowed enforcement
- Relay malfunction blocking
- Torque bounds validation

### fuzz_can_rx

Tests CAN RX message parsing:
- Counter validation and rollover
- CRC/checksum calculation
- Signal extraction (signed/unsigned)
- Speed value parsing

## Sanitizers

All targets are built with:
- **AddressSanitizer (ASAN)**: Detects memory errors (buffer overflows, use-after-free)
- **UndefinedBehaviorSanitizer (UBSAN)**: Detects undefined behavior (signed overflow, null dereference)

## Adding New Fuzz Targets

1. Create a new `fuzz_*.c` file with:
   ```c
   int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
     // Parse data into test inputs
     // Call function under test
     // Check invariants (use __builtin_trap() on violation)
     return 0;
   }
   ```

2. Add the target to the Makefile

3. Create seed corpus files for better coverage

## CI Integration

The GitHub Actions workflow runs fuzzing on every push/PR that modifies:
- `verification/fuzz/**`
- `opendbc/safety/**`

Crashes are uploaded as artifacts for analysis.

## Requirements

- Clang with libFuzzer support (clang-14+)
- LLVM tools for corpus management
