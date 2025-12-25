# Security Alerts Analysis

**Date**: 2025-12-25
**Source**: GitHub CodeQL Code Scanning
**Total Alerts**: 24

## Critical Severity (2)

### Alert #4, #5: C++ Command Injection
**File**: `common/prefix.h:26-32`
**Severity**: Critical
**CWE**: CWE-78 (OS Command Injection)

**Vulnerable Code**:
```cpp
~OpenpilotPrefix() {
  auto param_path = Params().getParamPath();
  if (util::file_exists(param_path)) {
    std::string real_path = util::readlink(param_path);
    system(util::string_format("rm %s -rf", real_path.c_str()).c_str());  // Line 26
    unlink(param_path.c_str());
  }
  if (getenv("COMMA_CACHE") == nullptr) {
    system(util::string_format("rm %s -rf", Path::download_cache_root().c_str()).c_str());  // Line 30
  }
  system(util::string_format("rm %s -rf", Path::comma_home().c_str()).c_str());  // Line 32
  system(util::string_format("rm %s -rf", msgq_path.c_str()).c_str());  // Line 33
}
```

**Issue**: Paths derived from environment variables (`OPENPILOT_PREFIX`, `COMMA_CACHE`) or filesystem operations are passed directly to `system()` without sanitization. A malicious path like `; rm -rf /` could execute arbitrary commands.

**Risk Assessment**:
- This is test infrastructure code (`OpenpilotPrefix` is used in test fixtures)
- Paths come from internal functions, not direct user input
- However, environment variables can be controlled by attackers in some contexts

**Recommended Fix**:
```cpp
// Option 1: Use std::filesystem instead of system()
#include <filesystem>
~OpenpilotPrefix() {
  auto param_path = Params().getParamPath();
  if (util::file_exists(param_path)) {
    std::string real_path = util::readlink(param_path);
    std::filesystem::remove_all(real_path);
    unlink(param_path.c_str());
  }
  if (getenv("COMMA_CACHE") == nullptr) {
    std::filesystem::remove_all(Path::download_cache_root());
  }
  std::filesystem::remove_all(Path::comma_home());
  std::filesystem::remove_all(msgq_path);
}
```

---

## High Severity (3)

### Alert #3: Python Clear-Text Logging of Sensitive Data
**File**: `tools/lib/auth.py:104`
**Severity**: High
**CWE**: CWE-532 (Information Exposure Through Log Files)

**Flagged Code**:
```python
print(f'To sign in, use your browser and navigate to {oauth_uri}')
```

**Analysis**: CodeQL flagged this because `oauth_uri` contains OAuth state parameters. However, this is a **false positive**:
- The URI is meant to be displayed to the user for browser navigation
- No actual passwords/tokens are logged
- This is interactive CLI output, not a log file

**Recommendation**: Mark as false positive in CodeQL or add a comment explaining the intentional behavior.

### Alert #6, #7: Untrusted Checkout in Privileged Workflow
**File**: `.github/workflows/jenkins-pr-trigger.yaml:36`
**Severity**: High
**CWE**: CWE-829 (Inclusion of Functionality from Untrusted Control Sphere)

**Vulnerable Pattern**:
```yaml
on:
  issue_comment:  # Triggered by any commenter
    types: [created, edited]

jobs:
  scan-comments:
    permissions:
      contents: write  # Has write access!
    steps:
    - uses: actions/checkout@v4
      with:
        ref: refs/pull/${{ github.event.issue.number }}/head  # Checks out untrusted PR code
```

**Issue**: The workflow:
1. Triggers on `issue_comment` (any commenter can trigger)
2. Has `contents: write` permission
3. Checks out untrusted PR code
4. Pushes to a branch

**Mitigations Already Present**:
- Checks for write/admin permission before checkout (line 31)
- Only triggers on specific "trigger-jenkins" phrase

**Remaining Risk**: Time-of-check-time-of-use (TOCTOU) - PR code could be modified between permission check and checkout.

**Recommended Fix**: Use `workflow_dispatch` with explicit PR number input, or use `pull_request_target` with careful isolation.

---

## Medium Severity (19)

### Alert #1, #2: JavaScript CDN Without Integrity Check
**File**: `tools/bodyteleop/static/index.html:7,10`
**Severity**: Medium
**CWE**: CWE-829 (Inclusion of Functionality from Untrusted Control Sphere)

**Vulnerable Code**:
```html
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/moment.js/2.29.1/moment.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@^3"></script>
```

**Issue**: Scripts loaded from CDNs without Subresource Integrity (SRI) hashes. If CDN is compromised, malicious code could be injected.

**Note**: Some scripts already have integrity attributes (bootstrap), others don't.

**Recommended Fix**:
```html
<script src="https://code.jquery.com/jquery-3.6.0.min.js"
        integrity="sha256-/xUj+3OJU5yExlq6GSYGSHk7tPXikynS7ogEvDej/m4="
        crossorigin="anonymous"></script>
```

### Alert #8-24: Missing Workflow Permissions
**Files**: `.github/workflows/tests.yaml`, `.github/workflows/stale.yaml`
**Severity**: Medium

**Issue**: Workflows don't explicitly limit `GITHUB_TOKEN` permissions, defaulting to broader access than needed.

**Recommended Fix**: Add explicit permissions block:
```yaml
permissions:
  contents: read
  # Add other permissions as needed
```

---

## Summary by Category

| Category | Critical | High | Medium | Total |
|----------|----------|------|--------|-------|
| Command Injection | 2 | 0 | 0 | 2 |
| Sensitive Data | 0 | 1* | 0 | 1 |
| Untrusted Code | 0 | 2 | 2 | 4 |
| Workflow Security | 0 | 0 | 17 | 17 |
| **Total** | **2** | **3** | **19** | **24** |

*Likely false positive

## Remediation Priority

1. **Immediate**: Fix C++ command injection in `common/prefix.h` (Critical)
2. **High**: Review Jenkins trigger workflow security model
3. **Medium**: Add SRI hashes to CDN scripts
4. **Low**: Add explicit permissions to workflows (defense in depth)

## Files Requiring Changes

| File | Alerts | Priority |
|------|--------|----------|
| `common/prefix.h` | 2 | Critical |
| `.github/workflows/jenkins-pr-trigger.yaml` | 2 | High |
| `tools/bodyteleop/static/index.html` | 2 | Medium |
| `.github/workflows/tests.yaml` | 9 | Low |
| `.github/workflows/stale.yaml` | 1 | Low |
| `tools/lib/auth.py` | 1 | False Positive |
