# Agent Instructions

This project uses **OpenSpec (OPSX)** for spec-driven development. The workflow is managed
through Claude Code skills and slash commands.

## Available Commands

| Command | Purpose |
|---------|---------|
| `/opsx:new` | Start a new change (proposal + artifacts) |
| `/opsx:ff` | Fast-forward: generate all artifacts at once |
| `/opsx:continue` | Create the next artifact for a change |
| `/opsx:apply` | Implement tasks from a change |
| `/opsx:verify` | Verify implementation matches artifacts |
| `/opsx:archive` | Archive a completed change |
| `/opsx:sync` | Sync delta specs to main specs |
| `/opsx:bulk-archive` | Archive multiple changes at once |
| `/opsx:explore` | Explore ideas before creating a change |
| `/opsx:onboard` | Guided walkthrough of the full workflow |

## Key Paths

- `openspec/project.md` - Project context and conventions
- `openspec/specs/` - Main capability specs (7 specs)
- `openspec/changes/` - Active changes (proposal, design, tasks, delta specs)
- `openspec/archive/` - Completed changes
- `.claude/commands/opsx/` - Slash command definitions
- `.claude/skills/` - Skill definitions

## When to Use OpenSpec

Use OPSX when the request:
- Adds new capabilities or features
- Makes breaking changes (API, schema, architecture)
- Introduces significant performance or security work
- Sounds ambiguous and needs a spec before coding

Skip OPSX for:
- Bug fixes, typos, formatting, comments
- Dependency updates (non-breaking)
- Configuration changes
- Tests for existing behavior
