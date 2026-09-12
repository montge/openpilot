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

- **openpilot/cereal/**: Message definitions (Cap'n Proto schemas in *.capnp files)
  - `services.py`: Defines all pub/sub services with frequencies
  - `log.capnp`: Main logging schema

- **tools/**: Development utilities
  - `replay/`: Replay recorded drives
  - `cabana/`: CAN message analyzer
  - `sim/`: Simulator integration
  - `plotjuggler/`: Log visualization

### External Submodules

Upstream removed the top-level convenience symlinks (`opendbc/`, `msgq/`, `rednose/`,
`tinygrad/`, `teleoprtc/`). These are now installed as editable packages from their
submodule checkouts via the `submodules` extra in `pyproject.toml`, so
`import opendbc` resolves to `opendbc_repo/opendbc/` without a symlink.

- `opendbc_repo/`: Car-specific DBC files, car ports, and safety code (`opendbc/safety/`)
- `panda/`: Hardware interface firmware
- `rednose_repo/`: Kalman filter library
- `tinygrad_repo/`: ML framework for models
- `msgq_repo/`: Messaging queue implementation
- `teleoprtc_repo/`: WebRTC signalling

### Messaging System

Processes communicate via ZeroMQ-based pub/sub defined in `openpilot/cereal/`. Each service has a defined frequency in `openpilot/cereal/services.py`. Use `openpilot.cereal.messaging` for publishing/subscribing.

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
from openpilot.common.hardware import HARDWARE
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
- Use pytest as the test runner (see Testing Notes for how this relates to upstream)
- Python 3.12 required (upstream pins `>= 3.12.3, < 3.13`; exact version in `.python-version`)

## Testing Notes

- Upstream dropped pytest in favor of a unittest-based harness: `OpenpilotTestCase`
  in `openpilot/common/test.py`, run via `tools/op.sh test` (`tools/test_runner.py`).
  **This fork keeps pytest.** pytest executes upstream's `OpenpilotTestCase` classes
  natively, and the fork's coverage/Sonar/codecov gates are all built on pytest-cov.
  So both styles coexist: upstream-owned tests inherit `OpenpilotTestCase`, and the
  fork's own suites stay plain pytest classes.
- Tests run in isolated `OpenpilotPrefix` environments. The root `conftest.py` is
  fork-owned (upstream deleted theirs); its autouse fixture deliberately steps aside
  for `OpenpilotTestCase` subclasses, which set up their own prefix in `run()`.
- Hardware-only tests set `COMMA_HARDWARE_TEST = True` on the test class and
  self-skip off-device. Upstream replaced the old `@pytest.mark.tici` marker with this.
- Tests marked with `@pytest.mark.slow` can be skipped with `-m "not slow"`

## Safety Critical Code

The safety model is enforced in panda firmware (see `opendbc_repo/opendbc/safety/`). Never disable or weaken:
- Driver monitoring in `openpilot/selfdrive/monitoring/`
- Actuation limits in `openpilot/selfdrive/selfdrived/helpers.py`

## NVIDIA / DGX Spark Development

The `openpilot/tools/dgx/` directory contains utilities for developing on NVIDIA hardware:

### Quick Start
```bash
# Run setup and GPU check
python openpilot/tools/dgx/setup.py --check

# Benchmark with TensorRT (recommended - 800+ FPS)
pip install tensorrt
python openpilot/tools/dgx/benchmark_tensorrt.py

# Monitor GPU during development
python openpilot/tools/dgx/gpu_monitor.py --monitor
```

### Key Files
- `openpilot/tools/dgx/setup.py` - Environment setup and GPU detection
- `openpilot/tools/dgx/benchmark_tensorrt.py` - TensorRT performance benchmark
- `openpilot/tools/dgx/gpu_monitor.py` - Memory and utilization monitoring
- `openpilot/tools/dgx/training/` - DoRA fine-tuning pipeline

### DoRA Fine-Tuning
For model fine-tuning on DGX Spark:
```bash
pip install torch onnx2pytorch tensorrt
python openpilot/tools/dgx/training/train.py --data ci --epochs 5
```

See `openpilot/tools/dgx/README.md` for complete documentation.

### Hardware Detection
```python
from openpilot.common.hardware.nvidia.gpu import (
    is_nvidia_available, get_best_gpu, get_recommended_precision
)

if is_nvidia_available():
    gpu = get_best_gpu()
    print(f"GPU: {gpu.name}, Precision: {get_recommended_precision(gpu)}")
```

### Notes
- TensorRT achieves 620x speedup over tinygrad on Blackwell GPUs
- tinygrad CUDA backend is not optimized for Blackwell architecture
- Training code requires PyTorch (not included in base openpilot env)
- Models: `driving_supercombo.onnx` (combined vision+policy), `big_driving_supercombo.onnx`, `dmonitoring_model.onnx`
