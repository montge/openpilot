# MISRA C:2012 Baseline Report

**Date**: 2026-07-03
**Tool**: cppcheck 2.21.0 with MISRA addon
**Branch**: fork/post-sync-cleanup (post July-2026 upstream sync, `openpilot/` package layout)

> Regenerated after the July 2026 upstream restructure (commaai `07ec389f4`): all code
> moved under `openpilot/`, upstream deleted a large portion of the C++ tree, and
> generated code is now excluded at analysis time (`-i` in `scripts/lint/cppcheck-misra.sh`)
> rather than counted and subtracted. Numbers are therefore not directly comparable to
> the 2025-12-25 baseline (1,003 actionable of 3,255 total).

## Executive Summary

| Metric | Count |
|--------|-------|
| Total MISRA violations | 235 |
| In generated code | 0 (excluded from analysis) |
| Actionable | **235** |
| Unique rules violated | 6 |
| C/C++ files analyzed | 37 |

## MISRA C:2012 Violations by Rule

| Rule | Count | Description | Severity |
|------|-------|-------------|----------|
| 12.3 | 132 | The comma operator should not be used | Advisory |
| 2.2 | 78 | There shall be no dead code | Required |
| 7.1 | 16 | Octal constants shall not be used | Required |
| 17.2 | 4 | Functions shall not call themselves, either directly or indirectly | Required |
| 13.5 | 4 | The right hand operand of && or \|\| shall not contain persistent side effects | Required |
| 2.3 | 1 | A project should not contain unused type declarations | Advisory |

## Other cppcheck Findings (non-MISRA)

| Type | Count | Description |
|------|-------|-------------|
| dangerousTypeCast | 375 | C-style cast between incompatible types |
| shadowVariable | 171 | Local variable shadows outer variable |
| cstyleCast | 136 | C-style cast used (prefer C++ casts) |
| knownConditionTrueFalse | 39 | Condition is always true/false |
| uninitMemberVarNoCtor | 31 | Member variable not initialized (no constructor) |
| variableScope | 13 | Variable scope could be reduced |
| constParameterPointer | 13 | Parameter could be pointer to const |

## Files Analyzed

`openpilot/selfdrive`, `openpilot/system`, `openpilot/common` — 37 C/C++ files total
(submodules, `.venv`, acados `c_generated_code`, and `locationd/models/generated` excluded).
Files with at least one finding: system 17, selfdrive 7, common 6.

## Generated Code Exclusions

Excluded from analysis entirely (`-i` / `--suppress` in `scripts/lint/cppcheck-misra.sh`):

- `openpilot/selfdrive/controls/lib/lateral_mpc_lib/c_generated_code/` (acados lateral MPC)
- `openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code/` (acados longitudinal MPC)
- `openpilot/selfdrive/locationd/models/generated/` (Kalman filter codegen)
- `openpilot/cereal/gen/` (Cap'n Proto generated code)

## Priority Remediation

### High Priority (Required rules)

1. **Rule 2.2** (78 violations): remove dead code
2. **Rule 7.1** (16 violations): replace octal constants
3. **Rule 17.2** (4 violations): eliminate recursion or document deviation
4. **Rule 13.5** (4 violations): hoist side effects out of `&&`/`||` operands

### Medium Priority (Advisory rules)

1. **Rule 12.3** (132 violations): avoid the comma operator (mostly macro/loop idioms)
2. **Rule 2.3** (1 violation): drop unused type declaration

## CI Threshold

`.github/workflows/misra.yml` gates (non-blocking) on actionable findings from
`reports/cppcheck-misra-report.txt` with `BASELINE=250` (235 actual + small buffer).
Ratchet the threshold down as violations are fixed.
