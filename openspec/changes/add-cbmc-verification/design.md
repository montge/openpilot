# CBMC Verification Design

## Architecture Overview

```
opendbc/safety/
├── safety.h           ← Core safety coordinator (534 lines)
├── lateral.h          ← Steering limits (353 lines)
├── longitudinal.h     ← Accel/brake limits (35 lines)
├── declarations.h     ← Data structures
├── helpers.h          ← Utility functions
└── modes/
    ├── toyota.h       ← Toyota safety mode (429 lines)
    ├── honda.h        ← Honda safety mode (436 lines)
    └── ...            ← 20 more modes
```

## Key Properties to Verify

### P1: Controls Authority Invariant
```c
// In safety.h tx_hook
__CPROVER_assert(
  !controls_allowed => !tx_allowed,
  "TX blocked when controls not allowed"
);
```

### P2: Relay Malfunction Blocks TX
```c
__CPROVER_assert(
  relay_malfunction => !tx_allowed,
  "TX blocked on relay malfunction"
);
```

### P3: Torque Bounds
```c
// In lateral.h steer_torque_cmd_checks
__CPROVER_assert(
  desired_torque >= -max_torque && desired_torque <= max_torque,
  "Torque within absolute limits"
);
```

### P4: No Integer Overflow
```c
// For bit-packed calculations like:
// int torque = (msg->data[5] << 8) | msg->data[6];
__CPROVER_assert(
  __CPROVER_overflow_result_plus(a, b) == false,
  "No overflow in torque calculation"
);
```

### P5: Rate Limiting
```c
__CPROVER_assert(
  abs(desired_torque - last_torque) <= max_rate,
  "Torque rate within limits"
);
```

## CBMC Harness Structure

```c
// verification/cbmc/safety_harness.c

#include "opendbc/safety/safety.h"

void verify_controls_blocked_invariant(void) {
  // Nondet inputs
  CANPacket_t msg;
  __CPROVER_havoc_object(&msg);

  // Precondition: controls not allowed
  controls_allowed = false;

  // Action: attempt to send
  int result = safety_tx_hook(&msg);

  // Postcondition: TX must be blocked
  __CPROVER_assert(result == 0, "TX blocked when controls disabled");
}

void verify_torque_bounds(void) {
  int16_t desired_torque;
  __CPROVER_assume(desired_torque >= -32768 && desired_torque <= 32767);

  bool violation = steer_torque_cmd_checks(desired_torque, ...);

  // If no violation, torque must be within limits
  __CPROVER_assert(
    violation || (desired_torque >= -limits.max_torque &&
                  desired_torque <= limits.max_torque),
    "Accepted torque within bounds"
  );
}
```

## CI Integration

```yaml
# .github/workflows/cbmc.yml
name: CBMC Verification

on:
  pull_request:
    paths:
      - 'opendbc_repo/opendbc/safety/**'

jobs:
  cbmc:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: true

      - name: Install CBMC
        run: |
          sudo apt-get update
          sudo apt-get install -y cbmc

      - name: Verify safety properties
        run: |
          cbmc verification/cbmc/safety_harness.c \
            --unwind 10 \
            --bounds-check \
            --pointer-check \
            --signed-overflow-check \
            -I opendbc_repo/opendbc/safety
```

## Incremental Approach

### Phase 1: Core Safety (This Proposal)
- `safety.h` TX/RX hooks
- `lateral.h` torque checks
- `longitudinal.h` accel checks

### Phase 2: Toyota Mode
- Full TX hook verification
- CAN message parsing
- State machine transitions

### Phase 3: Honda Mode
- Different state machine logic
- Multi-CAN bus handling

### Phase 4: Generalization
- Extract common patterns
- Template for new modes
- Automated property generation

## Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| Full unwinding | Complete proof | Slow for loops |
| Bounded checking | Fast | Not complete proof |
| Incremental | Manageable | More harnesses |

## Estimated Verification Time
- Core safety.h: ~30 seconds
- With Toyota mode: ~2-3 minutes
- Full verification: ~10 minutes in CI
