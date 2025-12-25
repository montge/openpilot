# MISRA Analysis Comparison Report

**Date**: 2025-12-21 (Updated - Full Gap Analysis)
**Target**: openpilot (commaai/openpilot)
**Directories**: selfdrive/, system/, common/

## Executive Summary

| Tool | Standard | Findings | Checks |
|------|----------|----------|--------|
| cppcheck + MISRA addon | MISRA C:2012 | 3,255 | ~50 rules |
| clang-tidy-automotive | MISRA C:2025 / C++:2023 | **81,387** | 176 checks |

The significant difference in finding counts is due to:
1. clang-tidy-automotive covers both C and C++ MISRA standards
2. MISRA C:2025 has stricter/different rules than C:2012
3. clang-tidy-automotive has more comprehensive rule implementations

## Coverage Analysis (Excluding Generated Code)

Comparing tools on non-generated code (excluding c_generated_code/, third_party/, etc.):

### Coverage by Rule (Updated with camerad files)

| Rule | Description | cppcheck | automotive | File Coverage |
|------|-------------|----------|------------|---------------|
| 17.7 | Return value ignored | 18 files | 41+ files | **100%** ✓ |
| 10.4 | Type conversion | 27 files | 38+ files | **96.3%** ✓ |
| 15.5 | Single exit point | 18 files | 21+ files | **88.9%** ⚠ |
| 14.4 | Controlling expression | 5 files | 29+ files | **100%** ✓ |
| 2.7 | Unused parameters | 16 files | 11 files | 18.8% ✗ |
| 22.10 | errno testing | 2 files | 1 file | 0% ✗ (different patterns) |
| 8.4 | Compatible declaration | 12 files | 0 | 0% ✗ (no check) |
| 12.1 | Operator precedence | 31 files | 0 | N/A (.cl files) |

### Critical Gaps in clang-tidy-automotive

**Missing checks (not implemented):**
1. **Rule 8.4**: Compatible declaration visible before definition
   - cppcheck finds 20 violations, automotive has no check
   - GitHub: https://github.com/montge/clang-tidy-automotive/issues/29

2. **Rule 12.1**: Operator precedence explicit
   - cppcheck finds in .cl (OpenCL) files
   - Not applicable to clang-tidy (different file type)

**Checks with incomplete detection:**
3. **Rule 14.4**: Controlling expression must be boolean
   - automotive-c23-req-14.4 exists but misses pointer-in-boolean-context
   - e.g., `if (getenv("VAR"))` not detected
   - GitHub: https://github.com/montge/clang-tidy-automotive/issues/30

4. **Rule 2.7**: Unused function parameters
   - automotive-c23-adv-2.7 exists but only 33.3% file coverage
   - GitHub: https://github.com/montge/clang-tidy-automotive/issues/31

5. **Rule 22.10**: errno testing after errno-setting functions
   - automotive-c23-req-22.10 detects different patterns than cppcheck
   - cppcheck: flags fopen/ioctl/poll errno tests (may be false positives)
   - automotive: flags sendMessage/mkdtemp errno tests (correct?)

### Files Not Analyzed by Automotive

Some files have cppcheck findings but aren't analyzed by automotive due to:
1. **Missing from compile_commands.json**: system/camerad/* (0 files)
2. **Script exclusions**: c_generated_code/, selfdrive/ui/, tools/cabana/
3. **OpenCL files (.cl)**: Not supported by clang-tidy

## Phase 1: cppcheck MISRA C:2012 Results

**Total Findings**: 3,255

### Top 10 Rules by Frequency

| Count | Rule | Description |
|-------|------|-------------|
| 533 | misra-c2012-15.5 | Single point of exit |
| 422 | misra-c2012-8.4 | Compatible declaration visible |
| 175 | misra-c2012-12.1 | Operator precedence |
| 173 | misra-c2012-10.4 | Arithmetic type conversion |
| 170 | misra-c2012-11.9 | NULL macro usage |
| 128 | misra-c2012-2.7 | Unused function parameters |
| 127 | misra-c2012-14.4 | Controlling expression boolean |
| 123 | misra-c2012-17.7 | Return value ignored |
| 120 | misra-c2012-8.9 | Object scope |
| 104 | misra-c2012-8.2 | Function prototype form |

## Phase 2: clang-tidy-automotive MISRA C:2025/C++:2023 Results

**Total Findings**: 59,791 (in openpilot source files)

### Top 20 Rules by Frequency

| Count | Rule | Description |
|-------|------|-------------|
| 7,401 | automotive-c23-req-21.2 | Standard library headers |
| 3,605 | automotive-c23-req-9.1 | Object initialization |
| 3,201 | automotive-cpp23-req-8.4.1 | Function declaration |
| 3,130 | automotive-avoid-goto | Avoid goto statements |
| 2,940 | automotive-unterminated-escape-sequence | String escape sequences |
| 2,875 | automotive-cpp23-adv-10.3 | Unnamed namespace vs static |
| 2,831 | automotive-c23-req-15.2 | Control statement body |
| 2,779 | automotive-cpp23-adv-11.3/req-8.2.1 | Pointer conversions |
| 2,439 | automotive-avoid-multiple-return-stmt | Single exit point |
| 2,327 | automotive-c23-req-11.11 | Pointer to array conversion |
| 2,316 | automotive-c23-req-13.1 | Initializer lists |
| 2,136 | automotive-wrong-null-pointer-value | NULL pointer value |
| 1,668 | automotive-avoid-reserved-macro-identifier | Reserved macro names |
| 1,584 | automotive-c23-req-21.11 | setjmp/longjmp |
| 1,506 | automotive-cpp23-req-9.2 | Initialization |
| 1,426 | automotive-c23-req-10.4 | Type conversions |
| 1,409 | automotive-cpp23-req-9.3 | Default initialization |
| 1,397 | automotive-c23-req-14.3 | Control expressions |
| 1,374 | automotive-cpp23-req-7.11 | Literal conversions |
| 1,273 | automotive-c23-req-11.3 | Pointer conversions |

## Rule Category Mapping

### Common Issues Detected by Both Tools

| Category | cppcheck Rule | clang-tidy-automotive Rule |
|----------|---------------|---------------------------|
| Single exit | misra-c2012-15.5 | automotive-avoid-multiple-return-stmt |
| Null pointer | misra-c2012-11.9 | automotive-wrong-null-pointer-value |
| Type conversion | misra-c2012-10.4 | automotive-c23-req-10.4 |
| Object scope | misra-c2012-8.9 | automotive-c23-adv-8.9 |
| Return value | misra-c2012-17.7 | automotive-missing-return-value-handling |

### Unique to clang-tidy-automotive

- **automotive-cpp23-req-21.6**: Dynamic memory allocation (new/delete)
- **automotive-cpp23-adv-10.3**: Unnamed namespace preference
- **automotive-avoid-goto**: Explicit goto avoidance
- **automotive-c23-req-9.1**: Stricter initialization requirements

## Known Issues

### clang-tidy-automotive Bugs

1. **automotive-cpp23-req-8.3.1 crash** (Issue #5)
   - Segfaults on Qt headers
   - Workaround: Exclude check when analyzing Qt files
   - GitHub: https://github.com/montge/clang-tidy-automotive/issues/5 (Fixed)

### Coverage Gaps (GitHub Issues)

| Issue | Rule | Description | Status |
|-------|------|-------------|--------|
| [#29](https://github.com/montge/clang-tidy-automotive/issues/29) | 8.4 | Compatible declaration - no check exists | **Open** |
| [#31](https://github.com/montge/clang-tidy-automotive/issues/31) | 2.7 | Unused parameters - 18.75% file overlap | **Open** |
| [#32](https://github.com/montge/clang-tidy-automotive/issues/32) | 22.10 | errno testing - 0% overlap, different patterns | **Open** |
| [#28](https://github.com/montge/clang-tidy-automotive/issues/28) | 10.4 | int*float/int*size_t - detection fixed | Closed |
| [#30](https://github.com/montge/clang-tidy-automotive/issues/30) | 14.4 | Pointer-in-boolean - detection fixed | Closed |

## Recommendations

### High Priority (Safety Critical)

1. **Return value handling** (misra-c2012-17.7)
   - 123 instances where function return values are ignored
   - Common in system calls and error handling

2. **Type conversions** (misra-c2012-10.4 / automotive-c23-req-10.4)
   - Implicit conversions may lose precision
   - Review arithmetic operations

3. **Object initialization** (automotive-c23-req-9.1)
   - 3,605 instances of potentially uninitialized objects
   - Critical for safety-critical code

### Medium Priority

4. **Single exit point** (misra-c2012-15.5)
   - 533 functions with multiple return statements
   - Consider refactoring complex functions

5. **Pointer conversions** (automotive-c23-req-11.3)
   - Review all void* casts and pointer arithmetic

### Low Priority (Code Quality)

6. **Unused parameters** (misra-c2012-2.7)
   - 128 unused function parameters
   - Consider removing or marking as intentional

7. **Static vs unnamed namespace** (automotive-cpp23-adv-10.3)
   - C++ style preference
   - Not a safety issue

## Next Steps

1. **Triage high-priority findings** in safety-critical paths:
   - selfdrive/controls/
   - selfdrive/pandad/
   - system/loggerd/

2. **Create suppression baseline** for false positives

3. **Integrate into CI** for incremental checking

4. **Track progress** over time

## Files Generated

- `reports/cppcheck-misra-report.txt` - Raw cppcheck output
- `reports/clang-tidy-misra-report.txt` - Raw clang-tidy-automotive output
- `reports/clang-tidy-automotive-issues.md` - Tool bugs encountered
- `scripts/lint/cppcheck-misra.sh` - cppcheck analysis script
- `scripts/lint/clang-tidy-misra.sh` - clang-tidy-automotive script
