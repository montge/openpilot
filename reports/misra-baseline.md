# MISRA C:2012 Baseline Report

**Date**: 2025-12-25
**Tool**: cppcheck 2.13.0 with MISRA addon
**Branch**: feature/test-ci-pipelines (develop)

## Executive Summary

| Metric | Count |
|--------|-------|
| Total MISRA violations | 3,255 |
| In generated code | 2,012 (62%) |
| In non-generated code | **1,003** (38%) |
| Unique rules violated | 23 |

**Key Finding**: The majority of violations (62%) are in auto-generated code (acados MPC solver). Focus remediation efforts on the 1,003 violations in manually-written code.

## Top MISRA C:2012 Violations

| Rule | Count | Description | Severity |
|------|-------|-------------|----------|
| 15.5 | 533 | A function should have a single point of exit | Advisory |
| 8.4 | 422 | A compatible declaration shall be visible | Required |
| 12.1 | 175 | Precedence of operators within expressions | Advisory |
| 10.4 | 173 | Both operands shall have the same essential type | Required |
| 11.9 | 170 | The macro NULL shall be the only permitted null pointer constant | Required |
| 2.7 | 128 | There should be no unused parameters | Advisory |
| 14.4 | 127 | The controlling expression of an if/while shall have boolean type | Required |
| 17.7 | 123 | The value returned by non-void functions shall be used | Required |
| 8.9 | 120 | An object should be defined at block scope if accessed only in a single function | Advisory |
| 8.2 | 104 | Function types shall be in prototype form | Required |

## Other cppcheck Findings

| Type | Count | Description |
|------|-------|-------------|
| cstyleCast | 79 | C-style cast used (prefer C++ casts) |
| constParameterPointer | 43 | Parameter could be pointer to const |
| constParameterCallback | 13 | Callback parameter could be const |
| variableScope | 10 | Variable scope could be reduced |
| preprocessorErrorDirective | 6 | #error directive encountered |

## Files Analyzed

| Directory | Files | Purpose |
|-----------|-------|---------|
| selfdrive/ | 52 | Driving functionality |
| system/ | 24 | System services |
| common/ | 12 | Common utilities |
| **Total** | **88** | |

## Generated Code Exclusions

The following paths contain auto-generated code and should be excluded from MISRA compliance requirements:

- `selfdrive/controls/lib/lateral_mpc_lib/c_generated_code/` (acados lateral MPC)
- `selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code/` (acados longitudinal MPC)
- `cereal/gen/` (Cap'n Proto generated code)

## Priority Remediation

### High Priority (Required Rules)

1. **Rule 8.4** (422 violations): Add forward declarations to headers
2. **Rule 10.4** (173 violations): Fix mixed type arithmetic
3. **Rule 14.4** (127 violations): Use explicit boolean comparisons
4. **Rule 17.7** (123 violations): Check return values

### Medium Priority (Advisory Rules)

1. **Rule 15.5** (533 violations): Refactor functions for single exit (low priority)
2. **Rule 12.1** (175 violations): Add explicit parentheses
3. **Rule 2.7** (128 violations): Remove or mark unused parameters

## Baseline Threshold

For CI quality gates, start with:
- **Non-blocking**: Report findings, don't fail the build
- **Track progress**: Count should decrease over time
- **Exclude generated**: Focus on manually-written code

## Next Steps

1. [ ] Exclude generated code from MISRA analysis
2. [ ] Fix high-priority Required rule violations
3. [ ] Set up differential analysis (new violations only)
4. [ ] Ratchet down violation count over time
