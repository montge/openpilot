# Proposal: Enhance C++ Coverage Workflow

## Problem Statement

The current C++ coverage workflow (`cpp-coverage.yml`) only covers 2 of 6 C++ test directories:
- **Covered**: `common/tests/`, `system/loggerd/tests/`
- **Missing**: `selfdrive/pandad/tests/`, `tools/replay/tests/`, `tools/cabana/tests/`, `system/camerad/test/`

Additionally:
- The threshold (65%) is below the project standard (80% minimum)
- No component-level coverage reporting for C++ (unlike Python)
- Coverage data not integrated with per-component Codecov targets

## Proposed Solution

Enhance the existing C++ coverage workflow to:
1. **Expand test coverage** to all C++ test directories
2. **Raise threshold** to 80% (matching project standard)
3. **Add component-level reporting** with separate flags for core vs tools
4. **Align with codecov.yml** component targets for unified reporting

## Success Criteria

- [ ] All 6 C++ test directories included in coverage
- [ ] Coverage threshold raised to 80%
- [ ] Component-level coverage visible in Codecov dashboard
- [ ] CI fails if C++ coverage drops below threshold

## Affected Specs

- `code-quality`: Modifies C++ Coverage Reporting requirement

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Build time increase | Run tests in parallel, cache dependencies |
| Missing dependencies for some tests | Add required system packages incrementally |
| Flaky tests causing false failures | Use `|| true` for test runs, fail only on coverage threshold |

## Out of Scope

- Adding new C++ tests (separate effort)
- C++ coverage for header-only code
- Coverage for panda firmware (separate toolchain)
