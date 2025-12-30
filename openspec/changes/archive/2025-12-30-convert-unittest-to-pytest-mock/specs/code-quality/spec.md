## ADDED Requirements

### Requirement: pytest-mock for Test Mocking
All test files SHALL use pytest-mock's `mocker` fixture instead of `unittest.mock`.

#### Scenario: Test file uses pytest-mock patterns
- **GIVEN** a test file that requires mocking
- **WHEN** the file is checked by ruff with TID251 rule
- **THEN** no `unittest` or `unittest.mock` imports are present
- **AND** mocking is done via the `mocker` fixture parameter

#### Scenario: Pre-commit hooks pass for test files
- **GIVEN** a developer modifies a test file
- **WHEN** they run `pre-commit run` or commit
- **THEN** the ruff TID251 check passes
- **AND** no manual `--no-verify` bypass is needed
