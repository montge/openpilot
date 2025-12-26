# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Setup development environment (first time)
tools/op.sh setup           # Installs dependencies, submodules, git-lfs files
source .venv/bin/activate   # Activate Python virtual environment

# Build
scons -u -j$(nproc)         # Full build with all CPU cores
scons -u -j4                # Build with 4 cores

# Linting
scripts/lint/lint.sh        # Run all linters (ruff, mypy, codespell)
scripts/lint/lint.sh --fast # Skip slow checks (mypy, codespell)
scripts/lint/lint.sh ruff   # Run specific linter

# Testing
pytest                      # Run all tests (uses pytest-xdist for parallelism)
pytest path/to/test.py      # Run single test file
pytest path/to/test.py::TestClass::test_method  # Run specific test
pytest -x                   # Stop on first failure
pytest -m "not slow"        # Skip slow tests
```

## Architecture Overview

openpilot is an operating system for robotics, currently used as a driver assistance system. It runs as a set of processes managed by `system/manager/manager.py`, communicating via pub/sub messaging.

### Core Directories

- **selfdrive/**: Main driving functionality
  - `controls/`: Vehicle control (controlsd.py - steering, acceleration)
  - `car/`: Car interface abstraction (card.py communicates with opendbc)
  - `modeld/`: ML models for driving and driver monitoring
  - `locationd/`: Localization, calibration, parameter estimation
  - `selfdrived/`: High-level driving state machine
  - `monitoring/`: Driver monitoring
  - `ui/`: User interface (raylib-based)
  - `pandad/`: Panda device communication (CAN bus interface)

- **system/**: System-level services
  - `manager/`: Process management and lifecycle
  - `loggerd/`: Logging and video encoding
  - `camerad/`: Camera capture (device-specific)
  - `athena/`: Cloud connectivity
  - `hardware/`: Hardware abstraction layer

- **cereal/**: Message definitions (Cap'n Proto schemas in *.capnp files)
  - `services.py`: Defines all pub/sub services with frequencies
  - `log.capnp`: Main logging schema

- **tools/**: Development utilities
  - `replay/`: Replay recorded drives
  - `cabana/`: CAN message analyzer
  - `sim/`: Simulator integration
  - `plotjuggler/`: Log visualization

### External Submodules (symlinked)
- `opendbc/` → `opendbc_repo/opendbc`: Car-specific DBC files and safety code
- `panda/`: Hardware interface firmware
- `rednose/` → `rednose_repo/rednose`: Kalman filter library
- `tinygrad/` → `tinygrad_repo/tinygrad`: ML framework for models
- `msgq/` → `msgq_repo/msgq`: Messaging queue implementation

### Messaging System

Processes communicate via ZeroMQ-based pub/sub defined in `cereal/`. Each service has a defined frequency in `cereal/services.py`. Use `cereal.messaging` for publishing/subscribing.

### Build System

Uses SCons (`SConstruct` at root). Each component has a `SConscript` file. The build:
1. Compiles C++ code with clang
2. Generates Cap'n Proto code from .capnp files
3. Builds Cython extensions

### Import Convention

Use fully qualified imports starting with `openpilot.`:
```python
from openpilot.selfdrive.controls.controlsd import ...
from openpilot.common.params import Params
from openpilot.system.hardware import HARDWARE
```

Banned imports (enforced by ruff): `selfdrive`, `common`, `system`, `tools`, `third_party` without `openpilot.` prefix.

## Git Workflow

This is a fork of commaai/openpilot. Important rules:
- **NEVER push to upstream** (commaai/openpilot)
- Only push to **origin** (montge/openpilot)
- Wait for explicit user approval before pushing
- PRs to upstream will be created manually by the user

Remotes:
- `origin` = montge/openpilot (fork) - OK to push here
- `upstream` = commaai/openpilot - DO NOT push here

## Code Style

- 2-space indentation (configured in ruff)
- Line length: 160 characters
- Use `time.monotonic()` instead of `time.time()`
- Use pytest (not unittest)
- Python 3.11+ required

## Testing Notes

- Tests run in isolated `OpenpilotPrefix` environments (see `conftest.py`)
- Tests marked with `@pytest.mark.tici` only run on comma device hardware
- Tests marked with `@pytest.mark.slow` can be skipped with `-m "not slow"`

## Safety Critical Code

The safety model is enforced in panda firmware (see `opendbc/safety/`). Never disable or weaken:
- Driver monitoring in `selfdrive/monitoring/`
- Actuation limits in `selfdrive/selfdrived/helpers.py`

<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->
