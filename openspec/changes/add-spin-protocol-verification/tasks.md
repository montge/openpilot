# Tasks: add-spin-protocol-verification

## Setup
- [x] Create verification/spin/ directory structure
- [x] Add SPIN installation to CI dependencies
- [x] Create Makefile for local verification

## Core Model
- [x] Model ring buffer data structures in Promela
- [x] Model Publisher process (write + invalidate)
- [x] Model Subscriber process (register + read)
- [x] Model atomic operations as atomic blocks
- [x] Model cycle counter wraparound logic

## Race Condition Models
- [x] Model new-reader-during-wraparound race
- [x] Model eviction-during-registration race
- [ ] Model conflate mode tight loop scenario
- [ ] Model UID reuse scenario

## Safety Properties
- [x] no_message_loss: valid readers get all messages (via no_stale_reads)
- [x] eviction_respected: evicted readers stop reading (checked inline)
- [x] cycle_prevents_stale: cycle counter prevents stale reads
- [x] no_buffer_corruption: no concurrent write to same slot (single publisher)

## Liveness Properties
- [x] publisher_progress: publisher eventually completes
- [x] reader_progress: valid readers eventually read (reader0_progress)
- [x] no_deadlock: system never gets stuck (built-in SPIN check)

## CI Integration
- [x] Create `.github/workflows/spin.yml` workflow
- [x] Configure path filters for msgq/ changes
- [x] Generate and compile pan.c verifier
- [x] Run verification with bounded state space

## Validation
- [x] SPIN finds the known wraparound race (LateSubscriber process)
- [x] SPIN proves no message loss for bounded runs
- [x] Document state space explosion limits
- [ ] Compare SPIN findings with existing unit tests

## Documentation
- [x] Write SPIN_VERIFICATION.md with usage instructions
- [x] Document Promela model assumptions
- [x] Add comments explaining each LTL property
