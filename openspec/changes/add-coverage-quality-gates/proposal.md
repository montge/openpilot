# Change: Add Coverage Quality Gates and Enforcement

## Why
The project needs enforced code coverage thresholds to maintain high quality standards. Current CI runs coverage but doesn't enforce minimum thresholds. Developers need standardized pre-commit hooks and clear quality gates to ensure consistent code quality across the team.

## What Changes
- Configure Codecov with enforced thresholds: 90%+ overall, 80% minimum per component
- Verify pre-commit hooks are installed and working for all developers
- Add coverage enforcement that fails PRs below thresholds
- Update project.md with coverage requirements and conventions
- Clean up MISRA baseline violations in generated code
- Merge pending quality infrastructure PRs (#1 SonarCloud, #2 OpenSpec)

## Impact
- Affected specs: `code-quality`
- Affected code:
  - `.github/workflows/codecov.yml` - threshold enforcement
  - `codecov.yml` - Codecov configuration
  - `.pre-commit-config.yaml` - already configured, verify installation
  - `openspec/project.md` - document conventions
