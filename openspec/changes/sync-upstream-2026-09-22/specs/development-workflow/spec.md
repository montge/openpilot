## ADDED Requirements

### Requirement: Upstream Sync Triage

An upstream sync SHALL start from a triage report that ranks the incoming upstream commits by the fork-side work they are likely to need, produced by `openpilot/tools/upstream_sync` before the merge.

#### Scenario: Triage before merging
- **WHEN** a sync of develop with upstream/master is planned
- **THEN** the triage report is generated for `develop..upstream/master` and saved with the sync's openspec change
- **AND** every commit in its resolve and adapt buckets is resolved, adapted, or recorded as needing no fork change before the sync lands

#### Scenario: Facts from git, judgments from TypeSafe, policy in code
- **WHEN** the triage runs
- **THEN** exact facts (predicted conflicts, fork-edited files, fork references, fork importers) come from git
- **AND** semantic judgments come from typed TypeSafe questions whose raw answers are kept in `triage.json`
- **AND** bucket placement is decided by explicit weights and thresholds in code, not by the model

#### Scenario: Safety-relevant upstream changes
- **WHEN** the triage judges a commit safety-relevant and fork code imports a module it changes
- **THEN** the commit is placed in adapt so that the fork's own checks of that code are re-run against upstream's change
- **AND** upstream's safety behavior is taken as written, never weakened by a fork change

#### Scenario: Calibration after each sync
- **WHEN** a sync lands
- **THEN** the commits that actually needed fork work are compared with the report, and misses are recorded in the change's tasks (and fixed in the tool when a general rule exists)
