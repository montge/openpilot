# add-spin-protocol-verification

## Summary
Add SPIN model checking to verify the lock-free msgq messaging protocol. SPIN excels at finding race conditions, deadlocks, and protocol violations in concurrent systems - exactly the issues identified in the shared memory ring buffer implementation.

## Motivation
The msgq messaging system (msgq_repo/msgq/) uses lock-free shared memory with:
- Atomic operations for 15 concurrent readers per service
- 64-bit packed cycle counters for wraparound detection
- UID-based reader eviction on process death
- Conflate mode for dropping intermediate messages

**Known issues identified during analysis:**
1. Race condition: New reader registration during wraparound invalidation
2. Reader eviction window during concurrent registration
3. Conflate mode tight loop potential
4. UID wraparound with TID reuse

Current verification: Unit tests cover happy paths but cannot explore all interleavings.

SPIN can:
- Exhaustively explore all thread interleavings
- Find subtle race conditions
- Verify no message loss (except in conflate mode)
- Prove reader eviction correctness

## Scope

### In Scope
- Promela model of msgq ring buffer protocol
- Verification of reader/writer synchronization
- Verification of cycle counter wraparound logic
- Verification of UID-based eviction

### Out of Scope
- Full C code verification (use CBMC for that)
- ZMQ fallback implementation
- VisionIPC (different protocol)
- Performance analysis

## Acceptance Criteria
- [ ] Promela model captures ring buffer read/write protocol
- [ ] SPIN proves no lost messages for single reader (non-conflate)
- [ ] SPIN proves reader eviction is correct
- [ ] SPIN finds the known wraparound race condition
- [ ] CI workflow runs SPIN on protocol changes

## Dependencies
- SPIN model checker (apt-get install spin)
- gcc for compiling pan.c verifier
- msgq_repo/msgq/msgq.cc (reference implementation)

## References
- [SPIN Documentation](http://spinroot.com/spin/whatispin.html)
- msgq_repo/msgq/msgq.cc (477 lines, ring buffer implementation)
- msgq_repo/msgq/msgq.h (data structures)
