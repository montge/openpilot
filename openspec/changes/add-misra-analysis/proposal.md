# Proposal: Add MISRA Static Analysis for openpilot

## Change ID
`add-misra-analysis`

## Summary
Integrate MISRA C/C++ static analysis into openpilot's development workflow using clang-tidy, establishing a baseline with standard clang-tidy safety checks and comparing against the clang-tidy-automotive fork which implements MISRA C:2025 and MISRA C++:2023 rules.

## Motivation
openpilot is safety-critical automotive software. MISRA (Motor Industry Software Reliability Association) guidelines are the industry standard for safety-critical C/C++ code. Adding MISRA compliance checking will:

1. Identify potential safety issues in the codebase
2. Establish coding standards aligned with automotive industry practices
3. Provide a foundation for future safety certifications
4. Enable comparison between standard clang-tidy and MISRA-specific checks

## Scope

### Phase 1: Baseline MISRA Analysis (cppcheck)
- Run cppcheck with MISRA C:2012 addon on openpilot's C/C++ code
- Document baseline findings with rule references
- Establish violation counts by category

### Phase 2: MISRA 2023/2025 Analysis (clang-tidy-automotive)
- Integrate clang-tidy-automotive fork (LLVM 20.1.8)
- Run MISRA C:2025 and C++:2023 checks
- Compare findings with Phase 1 baseline (MISRA C:2012 vs C:2025)

### Phase 3: Integration
- Create CI workflow for MISRA analysis
- Define acceptable violation thresholds
- Document suppression policy for false positives

## Out of Scope
- Full MISRA compliance certification
- Modification of clang-tidy-automotive fork
- Fixing all identified violations (this proposal focuses on analysis setup)

## Dependencies
- cppcheck 2.13.0 with MISRA addon (system: `/usr/bin/cppcheck`)
- clang-tidy-automotive fork at `/home/e/Development/clang-tidy-automotive`

## Risks
- High volume of initial findings may be overwhelming
- Some MISRA rules may not apply to openpilot's use case
- clang-tidy-automotive is under active development

## Success Criteria
1. Successfully run baseline clang-tidy analysis on openpilot
2. Successfully run clang-tidy-automotive MISRA checks
3. Generate comparison report between both toolsets
4. Document findings and create actionable next steps
