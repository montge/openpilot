# Project Context

## Purpose
openpilot is an open-source driver assistance system that runs on supported vehicles. It provides adaptive cruise control and lane-keeping functionality, with a focus on safety-critical software development.

## Tech Stack
- Python 3.11+ (primary application logic)
- C++17 (performance-critical components, real-time systems)
- Cap'n Proto (inter-process messaging)
- SCons (build system)
- pytest (testing framework)
- raylib (UI rendering)

## Project Conventions

### Code Style
- 2-space indentation (configured in ruff)
- Line length: 160 characters
- Python: ruff for linting and formatting
- C++: clang-format, clang-tidy
- Use `time.monotonic()` instead of `time.time()`
- Fully qualified imports: `from openpilot.selfdrive.controls.controlsd import ...`

### Pre-commit Hooks
All developers must use pre-commit for consistent code quality:
```bash
# Install hooks (run once)
pre-commit install

# Run on all files
pre-commit run --all-files

# Skip slow checks (mypy) when needed
SKIP=mypy git commit -m "message"
```

Hooks include: ruff, mypy, codespell, openspec-validate

### Coverage Requirements
- **Overall project target**: 90%+
- **Per-component minimum**: 80%
- **Safety-critical modules** (controls): 95%
- **New code (patch)**: 80% minimum

See `codecov.yml` for detailed component targets.

### Architecture Patterns
- Process-based architecture managed by `system/manager/manager.py`
- Pub/sub messaging via ZeroMQ (defined in `cereal/services.py`)
- Hardware abstraction layer in `system/hardware/`
- Car-specific logic abstracted in `selfdrive/car/`

### Testing Strategy
- Use pytest (not unittest)
- Tests run in isolated `OpenpilotPrefix` environments
- Markers: `@pytest.mark.tici` (device-only), `@pytest.mark.slow`
- Run with: `pytest -m "not slow"` for fast iteration
- Coverage measured via pytest-cov, uploaded to Codecov

### Git Workflow
- Fork: `origin` = montge/openpilot (push here)
- Upstream: `upstream` = commaai/openpilot (never push)
- Create feature branches from `develop`
- PRs require passing CI (tests, lint, coverage)

## Domain Context
- Safety-critical automotive software
- Real-time constraints on control loops
- Driver monitoring is mandatory
- Actuation limits enforced in panda firmware

## Important Constraints
- Never disable driver monitoring (`selfdrive/monitoring/`)
- Never weaken actuation limits (`selfdrive/selfdrived/helpers.py`)
- Safety model enforced in `opendbc/safety/`
- Real-time code must avoid memory allocation in hot paths

## External Dependencies
- Panda: CAN bus interface hardware
- Comma device: Target hardware platform
- External APIs: Comma Connect, map services
- ML models: Driving model, driver monitoring model
