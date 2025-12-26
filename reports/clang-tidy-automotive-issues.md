# clang-tidy-automotive Issues Found During openpilot Analysis

## Issue 1: Crash in automotive-cpp23-req-8.3.1 with Qt headers

**Severity**: Critical (crash/segfault)
**Check**: `automotive-cpp23-req-8.3.1`
**Component**: `AvoidImplicitConversionCheck::isSignificantConversion`

### Description
clang-tidy-automotive crashes with a segmentation fault when analyzing files that include Qt headers, specifically when processing `qobjectdefs_impl.h:185:42`.

### Crash Location
```
ASTMatcher: Processing 'automotive-cpp23-req-8.3.1' against:
  ImplicitCastExpr : </usr/include/x86_64-linux-gnu/qt5/QtCore/qobjectdefs_impl.h:185:42>
```

### Stack Trace (abbreviated)
```
clang::ASTContext::getTypeInfoImpl(clang::Type const*) const
clang::ASTContext::getTypeInfo(clang::Type const*) const
clang::tidy::automotive::AvoidImplicitConversionCheck::isSignificantConversion(...)
clang::tidy::automotive::AvoidImplicitConversionCheck::check(...)
```

### Reproduction Steps
1. Build clang-tidy-automotive (LLVM 20.1.8)
2. Run on any C++ file that includes Qt headers:
   ```bash
   clang-tidy -checks="-*,automotive-cpp23-req-8.3.1" -p . tools/cabana/binaryview.cc
   ```

### Affected Files in openpilot
- tools/cabana/binaryview.cc
- tools/cabana/cabana.cc
- tools/cabana/cameraview.cc
- (any Qt-based UI files)

### Workaround
Exclude the check when analyzing Qt files:
```bash
clang-tidy -checks="-*,automotive-*,-automotive-cpp23-req-8.3.1" ...
```

### Suggested Fix
The `AvoidImplicitConversionCheck::isSignificantConversion` function needs null pointer/type validation before calling `getTypeInfo()` on types from Qt template metaprogramming headers.

---

## Report Generated
- Date: 2024-12-21
- clang-tidy-automotive version: LLVM 20.1.8
- Target: openpilot (commaai/openpilot)
