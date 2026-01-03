# Tasks: add-spin-protocol-verification

## Setup
- [ ] Create verification/spin/ directory structure
- [ ] Add SPIN installation to CI dependencies
- [ ] Create Makefile for local verification

## Core Model
- [ ] Model ring buffer data structures in Promela
- [ ] Model Publisher process (write + invalidate)
- [ ] Model Subscriber process (register + read)
- [ ] Model atomic operations as atomic blocks
- [ ] Model cycle counter wraparound logic

## Race Condition Models
- [ ] Model new-reader-during-wraparound race
- [ ] Model eviction-during-registration race
- [ ] Model conflate mode tight loop scenario
- [ ] Model UID reuse scenario

## Safety Properties
- [ ] no_message_loss: valid readers get all messages
- [ ] eviction_respected: evicted readers stop reading
- [ ] cycle_prevents_stale: cycle counter prevents stale reads
- [ ] no_buffer_corruption: no concurrent write to same slot

## Liveness Properties
- [ ] publisher_progress: publisher eventually completes
- [ ] reader_progress: valid readers eventually read
- [ ] no_deadlock: system never gets stuck

## CI Integration
- [ ] Create `.github/workflows/spin.yml` workflow
- [ ] Configure path filters for msgq/ changes
- [ ] Generate and compile pan.c verifier
- [ ] Run verification with bounded state space

## Validation
- [ ] SPIN finds the known wraparound race
- [ ] SPIN proves no message loss for bounded runs
- [ ] Document state space explosion limits
- [ ] Compare SPIN findings with existing unit tests

## Documentation
- [ ] Write SPIN_VERIFICATION.md with usage instructions
- [ ] Document Promela model assumptions
- [ ] Add comments explaining each LTL property
