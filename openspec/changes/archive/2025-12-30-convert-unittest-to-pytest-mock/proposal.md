# Proposal: Convert unittest to pytest-mock

## Why

The project's ruff configuration bans `unittest` imports (`"unittest".msg = "Use pytest"`), but 52 test files in the main codebase still use `from unittest.mock import ...` or `import unittest`. This causes pre-commit and pre-push hooks to fail, blocking contributions.

The project already uses pytest-mock (listed in pyproject.toml dependencies) and has existing patterns in `tools/lib/tests/test_logreader.py` demonstrating proper usage.

## What Changes

Convert all test files from unittest.mock patterns to pytest-mock's `mocker` fixture:

| Pattern | Before | After |
|---------|--------|-------|
| Import | `from unittest.mock import patch, MagicMock` | (remove - use fixture) |
| Decorator | `@patch('module.thing')` | `mocker.patch('module.thing')` |
| Context manager | `with patch('x') as m:` | `m = mocker.patch('x')` |
| MagicMock | `MagicMock()` | `mocker.MagicMock()` |
| PropertyMock | `PropertyMock()` | `mocker.PropertyMock()` |

## Scope

**In scope (52 files):**
- `common/tests/` (14 files)
- `selfdrive/*/tests/` (23 files)
- `system/*/tests/` (9 files)
- `tools/*/tests/` (6 files)

**Out of scope:**
- Submodules: `opendbc_repo/`, `panda/`, `msgq_repo/`, `rednose_repo/`, `tinygrad_repo/`
- These have their own linting rules and are not subject to openpilot's ruff config

## Acceptance Criteria

1. All 52 files pass `ruff check --select TID251`
2. Pre-commit hooks pass without `--no-verify`
3. All converted tests pass (`pytest <file>`)
4. No functional changes to test behavior
