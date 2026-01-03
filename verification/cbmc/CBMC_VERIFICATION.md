# CBMC Safety Verification

This directory contains CBMC (C Bounded Model Checker) harnesses for verifying safety-critical properties in the opendbc safety code.

## Overview

CBMC performs bounded model checking on C code, proving that properties hold for all inputs within the specified bounds. Unlike testing, CBMC explores all possible execution paths.

## Verified Properties

| Property | File | Description |
|----------|------|-------------|
| P1 | `harness_controls_allowed.c` | `controls_allowed=false` blocks non-zero torque |
| P2 | `harness_relay_malfunction.c` | `relay_malfunction=true` blocks all TX messages |
| P3 | `harness_torque_bounds.c` | Accepted torque is within `[-max, +max]` limits |
| P4 | `harness_longitudinal.c` | Accel commands respect min/max limits |

## Requirements

Install CBMC:

```bash
# Ubuntu/Debian
sudo apt-get install cbmc

# macOS
brew install cbmc

# Verify installation
cbmc --version
```

## Usage

### Run All Verifications

```bash
cd verification/cbmc
make all
```

### Run Individual Verifications

```bash
make controls      # Verify controls_allowed property
make relay         # Verify relay_malfunction property
make torque        # Verify torque bounds property
make longitudinal  # Verify longitudinal accel property
```

### Quick Check (Development)

```bash
make quick  # Faster verification without overflow checks
```

## Adding New Properties

### 1. Create a Harness File

```c
// harness_my_property.c
#include "stubs.h"
#include "opendbc/safety/declarations.h"

/* Provide required external definitions */
bool controls_allowed = false;
// ... other globals

void verify_my_property(void) {
    // Set up preconditions with __CPROVER_assume()
    __CPROVER_assume(some_condition);

    // Call the function under test
    bool result = function_to_verify(...);

    // Assert the property with __CPROVER_assert()
    __CPROVER_assert(expected_condition, "Property description");
}

int main(void) {
    verify_my_property();
    return 0;
}
```

### 2. Add to Makefile

```makefile
my_property: check-cbmc harness_my_property.c
	$(CBMC) $(CBMC_FLAGS) $(INCLUDE_DIRS) harness_my_property.c
```

### 3. Add to CI Workflow

Update `.github/workflows/cbmc.yml` to include the new harness.

## CBMC Options

The Makefile uses these verification options:

| Option | Description |
|--------|-------------|
| `--bounds-check` | Check array bounds |
| `--pointer-check` | Check pointer safety |
| `--signed-overflow-check` | Check signed integer overflow |
| `--unsigned-overflow-check` | Check unsigned integer overflow |
| `--conversion-check` | Check type conversion safety |
| `--unwind 10` | Unroll loops up to 10 iterations |
| `--unwinding-assertions` | Assert loop bounds are sufficient |

## Interpreting Results

### Success

```
VERIFICATION SUCCESSFUL
```

All properties hold for all inputs within the bounds.

### Failure

```
VERIFICATION FAILED
Counterexample:
  ...
```

CBMC found an input that violates the property. The counterexample shows the specific values that cause the violation.

### Inconclusive

```
unwinding assertion loop 0: FAILURE
```

The loop bound (`--unwind N`) may be too small. Increase it or add `--unwinding-assertions` to ensure completeness.

## Architecture

```
verification/cbmc/
├── stubs.h                      # Hardware stubs for model checking
├── harness_controls_allowed.c   # P1: controls_allowed blocks torque
├── harness_relay_malfunction.c  # P2: relay_malfunction blocks TX
├── harness_torque_bounds.c      # P3: torque bounds verification
├── harness_longitudinal.c       # P4: longitudinal accel bounds
├── Makefile                     # Build and run verifications
└── CBMC_VERIFICATION.md         # This file
```

## References

- [CBMC Documentation](https://www.cprover.org/cbmc/)
- [CBMC User Manual](https://www.cprover.org/cprover-manual/)
- [opendbc Safety Code](../../opendbc_repo/opendbc/safety/)
