# Design: MISRA Static Analysis Integration

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    openpilot codebase                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ selfdrive/  │  │  system/    │  │  common/    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌──────────────────┐           ┌──────────────────────┐
    │     cppcheck      │           │ clang-tidy-automotive │
    │  + MISRA addon    │           │   (LLVM 20.1.8)       │
    │  (MISRA C:2012)   │           │   (MISRA C:2025)      │
    │                   │           │   (MISRA C++:2023)    │
    └──────────────────┘           └──────────────────────┘
              │                               │
              ▼                               ▼
    ┌──────────────────┐           ┌──────────────────────┐
    │cppcheck-misra.txt │           │ automotive-misra.txt │
    └──────────────────┘           └──────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
                   ┌─────────────────────┐
                   │  Comparison Report   │
                   │  (C:2012 vs C:2025)  │
                   └─────────────────────┘
```

## Tool Configuration

### cppcheck + MISRA Addon (Baseline)
System cppcheck 2.13.0 with MISRA C:2012 addon:

| Feature | Details |
|---------|---------|
| Standard | MISRA C:2012 (with amendments 1 & 2) |
| Coverage | Partial (open source), ~143 rules |
| Location | `/usr/bin/cppcheck` |
| Addon | `/usr/lib/x86_64-linux-gnu/cppcheck/addons/misra.py` |

### clang-tidy-automotive (Target)
Custom LLVM fork with MISRA 2023/2025 checks:

| Check Pattern | Coverage |
|--------------|----------|
| automotive-* | MISRA C:2025 (93/176 rules) |
| automotive-cpp23-* | MISRA C++:2023 (5 dedicated + 75 aliases) |

## Directory Exclusions

Both analyses will exclude:
- `third_party/` - Vendored dependencies
- `*_repo/` - Submodule repositories (msgq_repo, opendbc_repo, etc.)
- `tinygrad_repo/` - ML framework
- `cereal/gen/` - Generated Cap'n Proto code
- `.venv/` - Python virtual environment

## Output Format

Reports will be generated in clang-tidy's standard format:
```
file.cpp:line:col: warning: message [check-name]
```

This enables:
- Integration with IDE/editors
- CI/CD parsing
- SonarQube/SonarCloud import (if desired)

## Trade-offs

### Using Two Tools vs One
**Pros:**
- Validates automotive fork against established checks
- Provides broader coverage
- Enables comparison of detection capabilities

**Cons:**
- More complex setup
- Potential for duplicate findings
- Different LLVM versions may cause inconsistencies

### Full vs Incremental Analysis
**Decision:** Start with full analysis on core directories
**Rationale:** Need baseline before incremental CI integration

## Future Considerations

1. **CI Integration:** After establishing baseline, add PR-level checks
2. **Suppression Management:** Need policy for intentional violations
3. **Version Tracking:** clang-tidy-automotive is evolving rapidly
4. **SonarCloud Integration:** Could feed findings into existing SonarCloud setup
