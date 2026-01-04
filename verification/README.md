# Formal Verification & Security

This directory contains formal verification tools and security testing infrastructure
for openpilot's safety-critical code.

## Overview

| Tool | Target | Technique | CI Workflow |
|------|--------|-----------|-------------|
| **CBMC** | C safety code | Bounded model checking | `cbmc.yml` |
| **TLA+** | State machine | Temporal logic | `tlaplus.yml` |
| **SPIN** | msgq protocol | Model checking | `spin.yml` |
| **libFuzzer** | C safety code | Coverage-guided fuzzing | `fuzz.yml` |

## Directory Structure

```
verification/
├── cbmc/           # CBMC bounded model checking
│   ├── harness_*.c # Verification harnesses
│   └── Makefile    # Local build/run
├── fuzz/           # libFuzzer continuous fuzzing
│   ├── fuzz_*.c    # Fuzz targets
│   └── Makefile    # Local build/run
├── spin/           # SPIN protocol verification
│   ├── *.pml       # Promela models
│   └── Makefile    # Local build/run
└── tlaplus/        # TLA+ state verification
    ├── *.tla       # TLA+ specifications
    └── *.cfg       # TLC configurations
```

## Quick Start

### CBMC (C Bounded Model Checking)
```bash
cd verification/cbmc
make all      # Run all harnesses
make syntax   # Syntax check only
```

### TLA+ (State Machine Verification)
```bash
cd verification/tlaplus
# Download tla2tools.jar first
java -jar tla2tools.jar -config SelfDrived_Safety.cfg SelfDrived.tla
```

### SPIN (Protocol Verification)
```bash
cd verification/spin
make all      # Compile and verify
make run      # Run verification
```

### libFuzzer (Continuous Fuzzing)
```bash
cd verification/fuzz
make all      # Build fuzz targets
make run      # Run quick fuzzing (10s each)
```

## What Each Tool Verifies

### CBMC
- `controls_allowed = false` blocks torque commands
- `relay_malfunction = true` blocks all TX
- Torque values within bounds
- No integer overflow in bit-packing

### TLA+
- Type invariants hold in all states
- State machine never enters undefined state
- Timer never exceeds maximum
- (Action properties verified structurally)

### SPIN
- No message loss for valid readers
- Reader eviction correctness
- Known race conditions detected
- No deadlock in protocol

### libFuzzer
- No crashes with random CAN messages
- No ASAN violations (memory safety)
- No UBSAN violations (undefined behavior)
- Invariants hold under fuzzing

## CI Integration

All tools run automatically on relevant code changes:

- **CBMC**: Runs on `opendbc/safety/**` changes
- **TLA+**: Runs on `selfdrive/selfdrived/state.py` or `verification/tlaplus/**`
- **SPIN**: Runs on `msgq/**` or `verification/spin/**`
- **Fuzzing**: Runs on `opendbc/safety/**` or `verification/fuzz/**`

## Adding New Verification

See individual README files in each subdirectory for details on extending
the verification suite.

## Security Scanning

In addition to formal verification, the project uses:

- **CodeQL**: Semantic code analysis (Python, C/C++)
- **Dependabot**: Automated dependency updates
- **SonarCloud**: Code quality and security scanning
- **MISRA**: C code compliance checking
