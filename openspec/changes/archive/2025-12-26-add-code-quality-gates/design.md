# Design: Code Quality Gates

## Context

openpilot has ~404 Python files and ~86 C/C++ files in core directories (selfdrive/, system/, common/). Current state:
- pytest-cov is configured but no coverage thresholds enforced
- MISRA analysis scripts exist but aren't in CI
- SonarCloud and Codecov workflows exist but untested

## Goals / Non-Goals

**Goals:**
- Enforce 90% line coverage, 80% branch coverage
- Integrate MISRA analysis into CI
- Track quality trends over time
- Fail PRs that regress coverage

**Non-Goals:**
- 100% coverage (diminishing returns)
- Fixing all existing MISRA violations (separate effort)
- Modifying upstream openpilot's CI

## Decisions

### Decision 1: Coverage Tools
- **Python**: pytest-cov with coverage.py (already in pyproject.toml)
- **C++**: llvm-cov for clang-compiled code (consistent with existing toolchain)
- **Unified Reporting**: Both feed into Codecov/SonarCloud

**Alternatives considered:**
- gcov: Works but llvm-cov integrates better with clang build
- Coveralls: Less feature-rich than Codecov

### Decision 2: Threshold Strategy
- **Start with baseline**: Measure current coverage first
- **Ratchet approach**: Coverage can only go up, never down
- **Per-module thresholds**: Safety-critical paths (controls/, pandad/) get stricter thresholds

**Rationale:** Avoids blocking all PRs while coverage is below target.

### Decision 3: MISRA Integration
- **cppcheck + MISRA addon**: Baseline MISRA C:2012 (stable, widely used)
- **clang-tidy-automotive**: MISRA C:2025 / C++:2023 (cutting edge)
- **Differential mode**: Only report new violations on PRs

**Rationale:** Two tools catch different issues; differential mode prevents noise.

### Decision 4: CI Workflow Structure
```
PR opened
  ├─ pytest --cov (Python coverage)
  ├─ llvm-cov (C++ coverage)
  ├─ cppcheck-misra (MISRA C:2012)
  └─ clang-tidy-automotive (MISRA 2023/2025)
      ↓
  Coverage check (fail if below threshold or regression)
      ↓
  Upload to Codecov + SonarCloud
```

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| High initial violation count blocks PRs | Use differential/baseline mode initially |
| C++ coverage harder to measure | Start with Python, add C++ incrementally |
| clang-tidy-automotive instability | Can be optional check (not blocking) |
| CI time increases | Parallelize analysis jobs |

## Migration Plan

1. **Phase 1: Baseline** - Measure current coverage (no gates)
2. **Phase 2: Python Gates** - Enable pytest-cov fail_under
3. **Phase 3: C++ Coverage** - Add llvm-cov reporting
4. **Phase 4: MISRA CI** - Add cppcheck-misra to CI (warnings only)
5. **Phase 5: Full Gates** - Enable all quality gates

## Open Questions

1. What is the current Python coverage percentage? (Need to measure)
2. Should clang-tidy-automotive be blocking or advisory?
3. Per-module thresholds: which modules are safety-critical?
