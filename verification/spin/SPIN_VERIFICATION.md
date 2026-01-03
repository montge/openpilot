# SPIN Protocol Verification for msgq

This directory contains SPIN model checker files for verifying the lock-free
msgq messaging protocol used in openpilot.

## Overview

The msgq system (`msgq_repo/msgq/msgq.cc`) implements a lock-free shared memory
ring buffer for inter-process communication. Key features:

- Shared memory ring buffer with atomic 64-bit packed pointers
- Cycle counters (32-bit) for wraparound detection
- Support for up to 15 concurrent readers per queue
- UID-based reader tracking and eviction
- Conflate mode for dropping intermediate messages

SPIN (Simple Promela Interpreter) exhaustively explores all possible thread
interleavings to find:
- Race conditions
- Deadlocks
- Protocol violations (e.g., lost messages, stale reads)
- Liveness issues (e.g., starvation)

## Files

| File | Description |
|------|-------------|
| `msgq_protocol.pml` | Promela model of the msgq ring buffer protocol |
| `Makefile` | Build and run verification targets |
| `SPIN_VERIFICATION.md` | This documentation file |

## Prerequisites

Install SPIN on Ubuntu/Debian:

```bash
sudo apt-get install spin
```

Verify installation:

```bash
spin -V
```

## Quick Start

```bash
cd verification/spin

# Run full verification (safety + liveness properties)
make verify

# Run safety checks only (faster)
make safety

# Run random simulation (for debugging)
make simulate

# Clean generated files
make clean
```

## Promela Model Structure

### Shared State

The model abstracts the `msgq_header_t` structure:

```promela
/* Ring buffer (stores message IDs) */
byte buffer[BUFFER_SIZE];

/* Write pointer: cycle counter + offset */
byte write_offset = 0;
byte write_cycles = 0;

/* Per-reader state */
byte read_offsets[NUM_READERS];
byte read_cycles_arr[NUM_READERS];
bool read_valids[NUM_READERS];
```

### Processes

1. **Publisher** - Models `msgq_msg_send()`:
   - Writes messages to ring buffer
   - Invalidates slow readers on wraparound
   - Updates write pointer atomically

2. **Subscriber** - Models `msgq_init_subscriber()` and `msgq_msg_recv()`:
   - Registers with atomic CAS
   - Reads messages and advances read pointer
   - Handles eviction by resetting position

3. **LateSubscriber** - Tests race condition:
   - Registers during publisher's invalidation loop
   - May escape invalidation (known issue)

### Key Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_READERS` | 2 | Number of concurrent readers (reduced from 15 for tractability) |
| `BUFFER_SIZE` | 4 | Ring buffer size in message slots |
| `MAX_CYCLE` | 3 | Cycle counter modulus |
| `MAX_MESSAGES` | 6 | Total messages to publish |

## Properties Verified

### Safety Properties

1. **no_stale_reads** - Valid readers never read uninitialized buffer slots
2. **all_messages_written** - Publisher writes all MAX_MESSAGES when done

### Liveness Properties

1. **publisher_progress** - Publisher eventually completes
2. **reader0_progress** - Valid readers eventually make progress

### Checked Inline

- Eviction race detection (read becomes invalid during read operation)
- Message gap detection (reader misses messages due to slow reading)

## Running Specific Verifications

### Verify a specific LTL property:

```bash
make ltl PROP=publisher_progress
```

### Increase search depth for larger state space:

```bash
make verify MAX_DEPTH=500000
```

### Run exhaustive verification (no partial order reduction):

```bash
make exhaustive
```

### Use bitstate hashing for very large state spaces:

```bash
make bitstate
```

## Interpreting Results

### Successful Verification

```
(Spin Version 6.5.2 -- ...)

Full statespace search for:
  never claim         + (all)
  assertion violations  + (if within scope of claim)
  ...

State-vector 48 byte, depth reached 1234, errors: 0
```

Key indicators:
- `errors: 0` - No property violations found
- `depth reached` - Maximum search depth explored

### Counterexample Found

```
pan:1: assertion violated ...
spin: trail ends after 42 steps
```

View the counterexample:

```bash
make trail
```

The trail shows the sequence of steps leading to the violation.

## Model Abstractions

The Promela model makes several abstractions from the real implementation:

| Aspect | Real Implementation | Promela Model |
|--------|---------------------|---------------|
| Pointer packing | 64-bit (32:32 split) | Separate byte variables |
| Atomicity | Hardware atomics + memory barriers | `atomic {}` blocks |
| Buffer | Byte array with messages | Message ID integers |
| UID | 64-bit random + TID | Simple byte counter |
| Signals | SIGUSR2 | Not modeled (polling assumed) |
| Memory layout | Shared mmap | Global variables |

## Known Issues Being Verified

### Issue 1: New Reader During Wraparound

A new reader may register while the publisher is iterating through
`num_readers` to invalidate slow readers. The new reader escapes
invalidation because it wasn't counted when the loop started.

```c
// msgq.cc lines 259-267
for (uint64_t i = 0; i < num_readers; i++){  // num_readers captured
  // New reader registers here - not in our copy of num_readers
  if ((read_pointer > write_pointer) && ...)
    *q->read_valids[i] = false;
}
```

The `LateSubscriber` process in the model tests this race.

### Issue 2: Conflate Mode Tight Loop

In conflate mode, if the publisher writes faster than the reader can
consume, the reader may restart repeatedly without making progress.
This is a liveness concern rather than safety.

## CI Integration

The `.github/workflows/spin.yml` workflow:

1. Triggers on changes to msgq files or verification models
2. Installs SPIN on Ubuntu runner
3. Generates verifier from Promela model
4. Runs safety verification (fast)
5. Runs full verification with LTL properties
6. Reports counterexamples if found

## Extending the Model

### Adding New Properties

Add LTL formulas at the end of `msgq_protocol.pml`:

```promela
ltl my_property {
  [](<>(messages_written > 0) -> <>(messages_read[0] > 0))
}
```

### Increasing Concurrency

Edit the model parameters (may cause state space explosion):

```promela
#define NUM_READERS 3
#define BUFFER_SIZE 8
```

Consider using bitstate hashing for larger configurations:

```bash
make bitstate MAX_DEPTH=1000000
```

### Adding Race Condition Tests

Create new processes that interleave with publisher/subscriber:

```promela
proctype RaceTest() {
  /* Wait for specific condition */
  do
  :: condition -> break
  :: else -> skip
  od;

  /* Perform racy operation */
  atomic { ... }
}
```

## Performance Tuning

| Technique | When to Use | Command |
|-----------|-------------|---------|
| Safety only | Skip liveness checks | `make safety` |
| State compression | Memory-bound | `make compressed` |
| Bitstate hashing | Very large state space | `make bitstate` |
| Increase hash | Memory available | `MAX_DEPTH=... HASH_SIZE=28 make verify` |

## References

- [SPIN Documentation](http://spinroot.com/spin/whatispin.html)
- [Promela Language Reference](http://spinroot.com/spin/Man/promela.html)
- [Model Checking Book](http://spinroot.com/spin/Doc/Book_extras/)
- msgq implementation: `msgq_repo/msgq/msgq.cc` (477 lines)
- msgq header: `msgq_repo/msgq/msgq.h`
