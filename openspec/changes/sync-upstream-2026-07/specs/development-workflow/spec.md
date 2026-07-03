## MODIFIED Requirements

### Requirement: Upstream Synchronization

The develop branch SHALL be kept reasonably synchronized with upstream/master to ensure compatibility and reduce merge conflicts.

#### Scenario: Periodic upstream sync
- **WHEN** upstream has significant changes (>20 commits behind)
- **THEN** a sync operation should be performed before new development work

#### Scenario: Sync verification
- **WHEN** an upstream sync is completed
- **THEN** all existing custom code and openspec changes remain functional
- **AND** fork-owned files are relocated to match any upstream layout changes

#### Scenario: Post-sync validation
- **WHEN** an upstream sync merge is completed
- **THEN** build verification SHALL succeed and lint checks SHALL pass
- **AND** fork-relevant tests SHALL pass locally, with Linux CI on a sync PR as the authoritative check for platform-dependent upstream tests

#### Scenario: Sync lands via pull request
- **WHEN** a sync touches a large surface (layout changes, API migrations)
- **THEN** it SHALL land through a PR against develop on the fork so Linux CI validates before merge
- **AND** pushes of upstream-synced branches use --no-verify to skip the LFS pre-push hook (LFS pushurl targets commaai's GitLab; the fork introduces no LFS objects)
