# TLA+ Verification for openpilot selfdrived State Machine

This directory contains TLA+ specifications for formally verifying the openpilot
selfdrived state machine. TLA+ is a formal specification language that enables
mathematical proof of system properties.

## Overview

The `SelfDrived.tla` specification models the 5-state machine from
`selfdrive/selfdrived/state.py` that controls vehicle engagement and
disengagement. The TLC model checker exhaustively verifies that safety
properties hold across ALL possible event sequences.

### Why TLA+?

Unit tests verify known scenarios, but cannot prove properties hold for every
possible sequence of events. TLA+ model checking explores the entire state
space, proving that:

- Disable commands are ALWAYS honored from ANY state
- NO_ENTRY ALWAYS blocks engagement
- The soft disable timer correctly counts down
- The system NEVER deadlocks

## Quick Start

### Prerequisites

- Java 11 or later (Java 17 recommended)
- Make (optional, for convenience)

### Running the Model Checker

Using Make:
```bash
cd verification/tlaplus
make check
```

Without Make:
```bash
cd verification/tlaplus

# Download TLA+ tools (one-time)
curl -fsSL -o tla2tools.jar \
  https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar

# Run TLC
java -XX:+UseParallelGC -Xmx4g -jar tla2tools.jar \
  -workers auto -deadlock -cleanup \
  -config SelfDrived.cfg SelfDrived.tla
```

### Expected Output

A successful run produces output like:
```
TLC2 Version 2.18 of ...
Running breadth-first search Model-Checking with fp ...
Computed initial states: 1 distinct state.
Model checking completed. No error has been found.
  Estimates of the progress ...
  State space: ... distinct states, ... states left on queue.
The depth of the complete state graph search is ...
```

If TLC finds a property violation, it outputs a counterexample trace showing
the sequence of states and events that violate the property.

## Files

| File | Description |
|------|-------------|
| `SelfDrived.tla` | TLA+ specification of the state machine |
| `SelfDrived.cfg` | TLC configuration (constants, invariants, properties) |
| `Makefile` | Build automation for running TLC |
| `TLA_VERIFICATION.md` | This documentation |

## State Machine Model

### States

The specification models 5 states matching the Python implementation:

| State | Description |
|-------|-------------|
| `disabled` | System not engaged, driver in full control |
| `preEnabled` | Waiting for pre-conditions (e.g., brake release) |
| `enabled` | System actively controlling vehicle |
| `softDisabling` | Graceful disable with 3-second countdown |
| `overriding` | User temporarily overriding controls |

### Events

| Event | Description |
|-------|-------------|
| `ENABLE` | Request to enable controls |
| `PRE_ENABLE` | Waiting for pre-enable conditions |
| `NO_ENTRY` | Block engagement (camera malfunction, etc.) |
| `SOFT_DISABLE` | Graceful disable with warning |
| `IMMEDIATE_DISABLE` | Immediate disable (serious issue) |
| `USER_DISABLE` | User-initiated disable (brake, cancel) |
| `OVERRIDE_LATERAL` | User overriding steering |
| `OVERRIDE_LONGITUDINAL` | User overriding accel/brake |

### Transition Logic

The state machine follows these priority rules:

1. **USER_DISABLE/IMMEDIATE_DISABLE**: Always transition to disabled (highest priority)
2. **State-specific**: Each state handles events according to its logic
3. **NO_ENTRY**: Blocks engagement from disabled state

## Verified Properties

### Safety Invariants

These properties are checked in every reachable state:

| Property | Description |
|----------|-------------|
| `TypeInvariant` | All variables have valid types |
| `StateValid` | State is always one of the 5 defined states |
| `TimerBounded` | Soft disable timer never exceeds maximum |

### Temporal Properties

These properties verify behavior over time (require fairness):

| Property | Description |
|----------|-------------|
| `SoftDisableProgress` | softDisabling eventually leads to disabled or enabled |
| `PreEnableProgress` | preEnabled eventually leads to enabled or disabled |
| `OverrideProgress` | overriding eventually resolves |

### Additional Checked Properties

| Property | Description |
|----------|-------------|
| `DisableAlwaysHonored` | USER_DISABLE/IMMEDIATE_DISABLE always reach disabled |
| `NoEntryBlocksEngagement` | NO_ENTRY prevents engagement from disabled |
| `DisableResetsTimer` | Disable commands reset the soft disable timer |

## Configuration

### SOFT_DISABLE_TIME

The timer countdown is configurable in `SelfDrived.cfg`:

```tla
CONSTANTS
  SOFT_DISABLE_TIME = 5
```

| Value | State Space | Verification Time |
|-------|-------------|-------------------|
| 5 | Small | Seconds |
| 10 | Medium | ~1 minute |
| 300 (production) | Very large | Hours/impractical |

The small value (5) is sufficient to verify timer logic correctness.

## Extending the Specification

### Adding New States

1. Add the state to the `States` set
2. Update `ActiveStates` and `EnabledStates` if applicable
3. Add a `TransitionFromNewState` operator
4. Update the `CASE` in `Update`
5. Run TLC to verify no properties are violated

### Adding New Events

1. Add the event to `EventTypes`
2. Update relevant transition operators
3. Add any new invariants/properties
4. Run TLC

### Adding New Properties

Add invariants to `SelfDrived.cfg`:
```tla
INVARIANTS
  TypeInvariant
  StateValid
  YourNewInvariant
```

Add temporal properties:
```tla
PROPERTIES
  SoftDisableProgress
  YourNewTemporalProperty
```

## CI Integration

The TLA+ verification runs automatically via GitHub Actions when:

- Files in `verification/tlaplus/` change
- `selfdrive/selfdrived/state.py` changes
- `selfdrive/selfdrived/events.py` changes

See `.github/workflows/tlaplus.yml` for the workflow definition.

## Troubleshooting

### TLC reports "deadlock reached"

The specification includes deadlock checking. If TLC reports deadlock:
1. Check the counterexample trace
2. Verify the `Next` relation covers all cases
3. Ensure no state has no enabled transitions

### TLC reports property violation

1. Read the counterexample trace carefully
2. The trace shows the sequence of states and events
3. Either fix the specification or update the property

### TLC runs out of memory

Increase Java heap:
```bash
java -Xmx8g -jar tla2tools.jar ...
```

Or reduce SOFT_DISABLE_TIME to shrink state space.

### TLC takes too long

Use safety-only checking:
```bash
make check-fast
```

This skips temporal properties which require exploring the full state graph.

## References

- [TLA+ Language Manual](https://lamport.azurewebsites.net/tla/tla.html)
- [Learn TLA+](https://learntla.com/)
- [TLC Model Checker](https://lamport.azurewebsites.net/tla/tlc.html)
- [selfdrive/selfdrived/state.py](../../selfdrive/selfdrived/state.py) - Reference implementation
- [selfdrive/selfdrived/events.py](../../selfdrive/selfdrived/events.py) - Event definitions
