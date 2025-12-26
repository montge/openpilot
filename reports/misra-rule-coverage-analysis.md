# MISRA C:2012 → C:2025 Rule Coverage Analysis

**Goal**: Ensure clang-tidy-automotive (MISRA C:2025) covers all rules detected by cppcheck (MISRA C:2012)

## Summary

| Status | Count | Description |
|--------|-------|-------------|
| ✅ Covered | TBD | C:2012 rule has C:2025 equivalent in clang-tidy-automotive |
| ⚠️ Partial | TBD | C:2012 rule partially covered or different scope |
| ❌ Gap | TBD | C:2012 rule not covered in clang-tidy-automotive |

## Detailed Rule Mapping

### Directive 4 - Code Design

| C:2012 Rule | Count | Description | C:2025 Equivalent | Status |
|-------------|-------|-------------|-------------------|--------|
| - | - | - | automotive-c23-adv-dir-4.8 | ✅ |
| - | - | - | automotive-c23-req-dir-4.10 | ✅ |

### Rule 2 - Unused Code

| C:2012 Rule | Count | Description | C:2025 Equivalent | Status |
|-------------|-------|-------------|-------------------|--------|
| 2.2 | 92 | No dead code | - | ❌ Gap |
| 2.4 | 1 | Unused tag declarations | - | ❌ Gap |
| 2.5 | 4 | Unused macro declarations | - | ❌ Gap |
| 2.7 | 128 | Unused function parameters | automotive-c23-adv-2.7 | ✅ |

### Rule 5 - Identifiers

| C:2012 Rule | Count | Description | C:2025 Equivalent | Status |
|-------------|-------|-------------|-------------------|--------|
| 5.6 | 24 | Unique typedef names | automotive-c23-req-5.6 | ✅ |
| 5.7 | 4 | Unique tag names | automotive-c23-req-5.7 | ✅ |
| 5.8 | 7 | Unique external identifiers | - | ❌ Gap |

### Rule 7 - Literals

| C:2012 Rule | Count | Description | C:2025 Equivalent | Status |
|-------------|-------|-------------|-------------------|--------|
| 7.1 | 16 | Octal constants | automotive-avoid-octal-number | ✅ |
| 7.4 | 24 | String literal assignment | automotive-c23-req-7.4 | ✅ |

### Rule 8 - Declarations

| C:2012 Rule | Count | Description | C:2025 Equivalent | Status |
|-------------|-------|-------------|-------------------|--------|
| 8.2 | 104 | Function types in prototype | automotive-cpp23-req-8.4.1 | ⚠️ C++ only |
| 8.4 | 422 | Compatible declaration visible | automotive-c23-adv-8.7 | ⚠️ Partial |
| 8.6 | 7 | Identifier with external linkage | - | ❌ Gap |
| 8.7 | 37 | Object/function at block scope | automotive-c23-adv-8.7 | ✅ |
| 8.9 | 120 | Object at block scope if single use | automotive-c23-adv-8.9 | ✅ |
| 8.10 | 2 | Inline functions declared static | - | ❌ Gap |
| 8.11 | 8 | Array dimension explicit | - | ❌ Gap |

### Rule 9 - Initialization

| C:2012 Rule | Count | Description | C:2025 Equivalent | Status |
|-------------|-------|-------------|-------------------|--------|
| 9.2 | 14 | Initializer shall be enclosed in braces | automotive-c23-req-9.2 | ✅ |
| 9.3 | 1 | Partially initialized arrays | - | ⚠️ |
| 9.4 | 1 | Element shall not be initialized more than once | automotive-c23-req-9.4 | ✅ |
| 9.5 | 3 | Designated initializers | automotive-c23-req-9.5 | ✅ |

### Rule 10 - Essential Types

| C:2012 Rule | Count | Description | C:2025 Equivalent | Status |
|-------------|-------|-------------|-------------------|--------|
| 10.1 | 12 | Operands not inappropriate essential type | automotive-c23-req-10.1 | ✅ |
| 10.3 | 12 | Value assigned to narrower essential type | automotive-c23-req-10.3 | ✅ |
| 10.4 | 173 | Arithmetic operations same essential type | automotive-c23-req-10.4 | ✅ |
| 10.5 | 2 | Value cast to inappropriate type | - | ⚠️ |
| 10.6 | 1 | Value assigned to wider essential type | automotive-c23-req-10.6 | ✅ |
| 10.7 | 1 | Composite expression assigned | - | ⚠️ |
| 10.8 | 4 | Value of composite expression cast | - | ⚠️ |

### Rule 11 - Pointer Conversions

| C:2012 Rule | Count | Description | C:2025 Equivalent | Status |
|-------------|-------|-------------|-------------------|--------|
| 11.3 | 6 | Cast pointer to different object type | automotive-c23-req-11.3 | ✅ |
| 11.4 | 5 | Conversion between pointer and integer | automotive-c23-adv-11.4 | ✅ |
| 11.5 | 30 | Conversion from void pointer | - | ❌ Gap |
| 11.6 | 3 | Cast between pointer and arithmetic type | - | ⚠️ |
| 11.8 | 5 | Cast removes const/volatile | automotive-c23-req-11.8 | ✅ |
| 11.9 | 170 | Macro NULL only null pointer constant | automotive-c23-req-11.9 | ✅ |

### Rule 12 - Expressions

| C:2012 Rule | Count | Description | C:2025 Equivalent | Status |
|-------------|-------|-------------|-------------------|--------|
| 12.1 | 175 | Precedence of operators | automotive-c23-req-12.2 | ⚠️ Different |
| 12.3 | 88 | Comma operator not used | automotive-avoid-comma-operator | ✅ |

### Rule 13 - Side Effects

| C:2012 Rule | Count | Description | C:2025 Equivalent | Status |
|-------------|-------|-------------|-------------------|--------|
| 13.1 | 3 | Initializer lists no side effects | automotive-c23-req-13.1 | ✅ |
| 13.3 | 1 | Full expression no multiple side effects | - | ❌ Gap |
| 13.4 | 23 | Result of assignment not used | automotive-c23-adv-13.4 | ✅ |

### Rule 14 - Control Statements

| C:2012 Rule | Count | Description | C:2025 Equivalent | Status |
|-------------|-------|-------------|-------------------|--------|
| 14.2 | 1 | For loop well-formed | automotive-c23-req-14.1 | ⚠️ Different |
| 14.4 | 127 | Controlling expression boolean | automotive-c23-req-14.4 | ✅ |

### Rule 15 - Control Flow

| C:2012 Rule | Count | Description | C:2025 Equivalent | Status |
|-------------|-------|-------------|-------------------|--------|
| 15.1 | 16 | goto not used | automotive-avoid-goto | ✅ |
| 15.4 | 2 | At most one break/goto per loop | automotive-c23-adv-15.4 | ✅ |
| 15.5 | 533 | Single point of exit | automotive-avoid-multiple-return-stmt | ✅ |
| 15.6 | 56 | Loop/selection body in braces | automotive-c23-req-15.2 | ✅ |
| 15.7 | 13 | All if...else if terminated with else | - | ❌ Gap |

### Rule 16 - Switch Statements

| C:2012 Rule | Count | Description | C:2025 Equivalent | Status |
|-------------|-------|-------------|-------------------|--------|
| 16.4 | 1 | Every switch has default | automotive-c23-req-16.5 | ⚠️ |
| 16.6 | 26 | Every switch at least two clauses | automotive-c23-req-16.6 | ✅ |

### Rule 17 - Functions

| C:2012 Rule | Count | Description | C:2025 Equivalent | Status |
|-------------|-------|-------------|-------------------|--------|
| 17.1 | 10 | stdarg.h not used | automotive-avoid-stdarg-header | ✅ |
| 17.2 | 1 | No recursive functions | automotive-c23-req-17.2 | ✅ |
| 17.3 | 68 | Function declared before use | - | ❌ Gap |
| 17.7 | 123 | Return value used or cast to void | automotive-missing-return-value-handling | ✅ |
| 17.8 | 7 | Parameters not modified | - | ❌ Gap |

### Rule 18 - Pointers and Arrays

| C:2012 Rule | Count | Description | C:2025 Equivalent | Status |
|-------------|-------|-------------|-------------------|--------|
| 18.4 | 92 | +/- operators not applied to pointers | automotive-c23-adv-18.4 | ✅ |

### Rule 20 - Preprocessing

| C:2012 Rule | Count | Description | C:2025 Equivalent | Status |
|-------------|-------|-------------|-------------------|--------|
| 20.5 | 2 | #undef not used | automotive-avoid-undef | ✅ |
| 20.7 | 2 | Expression from macro expansion in parentheses | - | ❌ Gap |
| 20.9 | 3 | All macro identifiers defined before use | - | ⚠️ |
| 20.10 | 52 | # and ## not used | automotive-avoid-hash-operator | ✅ |

### Rule 21 - Standard Libraries

| C:2012 Rule | Count | Description | C:2025 Equivalent | Status |
|-------------|-------|-------------|-------------------|--------|
| 21.1 | 5 | Reserved identifiers not defined | automotive-avoid-reserved-macro-identifier | ✅ |
| 21.2 | 3 | Reserved identifiers not declared | automotive-c23-req-21.2 | ✅ |
| 21.3 | 65 | Memory allocation functions not used | automotive-c23-req-21.3 | ✅ |
| 21.6 | 8 | Standard I/O functions not used | automotive-c23-req-21.6 | ✅ |
| 21.7 | 3 | atof/atoi/atol/atoll not used | - | ❌ Gap |
| 21.8 | 17 | Termination functions not used | automotive-c23-req-21.9 | ⚠️ Different |

### Rule 22 - Resources

| C:2012 Rule | Count | Description | C:2025 Equivalent | Status |
|-------------|-------|-------------|-------------------|--------|
| 22.10 | 4 | Value of errno only tested after call | - | ❌ Gap |

## Coverage Summary

### Covered (✅): ~35 rules
Rules with direct or functional equivalents in clang-tidy-automotive.

### Partial Coverage (⚠️): ~10 rules
Rules with different scope, numbering, or partial implementation.

### Gaps (❌): ~15 rules

**High Priority Gaps** (>10 violations in openpilot):
| C:2012 Rule | Count | Description |
|-------------|-------|-------------|
| 2.2 | 92 | Dead code detection |
| 17.3 | 68 | Function declared before use |
| 11.5 | 30 | Conversion from void pointer |
| 21.7 | 3 | atof/atoi not used |

**Lower Priority Gaps** (<10 violations):
- 2.4, 2.5 - Unused declarations
- 5.8 - Unique external identifiers
- 8.6, 8.10, 8.11 - Declaration rules
- 13.3 - Multiple side effects
- 15.7 - else termination
- 17.8 - Parameter modification
- 20.7 - Macro parentheses
- 22.10 - errno testing

## Recommendations

1. **File issues for high-priority gaps** in clang-tidy-automotive:
   - Rule 2.2 (dead code) - 92 violations
   - Rule 17.3 (function declarations) - 68 violations
   - Rule 11.5 (void pointer conversion) - 30 violations

2. **Keep cppcheck as supplementary tool** for rules not covered

3. **Create wrapper script** that runs both tools and deduplicates findings

4. **Track clang-tidy-automotive roadmap** for upcoming rule implementations
