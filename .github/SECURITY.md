# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in openpilot, please report it responsibly.

### For Critical Safety Issues

Issues that could affect vehicle safety should be reported immediately to:
- Email: security@comma.ai
- Include "SECURITY" in the subject line

### For Other Security Issues

- Open a GitHub Security Advisory (preferred)
- Or email security@comma.ai

### What to Include

1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

### Response Timeline

- Initial response: within 48 hours
- Status update: within 7 days
- Fix timeline: depends on severity

## Security Measures

This project implements multiple layers of security:

### Formal Verification
- **CBMC**: Bounded model checking for C safety code
- **TLA+**: State machine verification for selfdrived
- **SPIN**: Protocol verification for msgq
- **libFuzzer**: Continuous fuzzing with sanitizers

### Static Analysis
- **MISRA C:2012**: Compliance checking for safety-critical C code
- **SonarCloud**: Code quality and security scanning
- **Ruff**: Python linting with security rules

### Dependency Security
- **Dependabot**: Automated security updates
- **pip-audit**: Vulnerability scanning (manual)

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| master  | :white_check_mark: |
| develop | :white_check_mark: |
| < 0.9   | :x:                |
