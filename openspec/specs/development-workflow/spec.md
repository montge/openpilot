# development-workflow Specification

## Purpose
Define the upstream synchronization workflow for keeping the fork in sync with commaai/openpilot.
## Requirements
### Requirement: Upstream Synchronization

The develop branch SHALL be kept reasonably synchronized with upstream/master to ensure compatibility and reduce merge conflicts.

#### Scenario: Periodic upstream sync
- **WHEN** upstream has diverged significantly
- **THEN** a sync operation should be performed before new development work

#### Scenario: Sync verification
- **WHEN** an upstream sync is completed
- **THEN** the build succeeds, linter passes, and tests pass

#### Scenario: Post-sync validation
- **WHEN** an upstream sync merge is completed
- **THEN** formal verification checks (CBMC, SPIN, TLA+, libfuzzer) SHALL be run
- **AND** build verification SHALL succeed
- **AND** lint checks SHALL pass

> **Note:** Formal verification checks run as part of standard CI on push, not as a separate post-sync step.
