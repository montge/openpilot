# Change: Sync with Upstream Master

## Why

Our develop branch is 22 commits behind commaai/openpilot master. Before starting new development, we need to sync with upstream to:

1. Get latest bug fixes and improvements
2. Avoid merge conflicts later
3. Ensure our changes are compatible with current upstream

## What Changes

- **Merge upstream/master into develop**:
  - Fetch latest from commaai/openpilot
  - Merge upstream/master into local develop
  - Resolve any merge conflicts
  - Push updated develop to origin

## Notable Upstream Changes (22 commits)

Recent upstream changes include:
- `enable pyopencl on arm64` - ARM64 OpenCL support
- `Modeld: less lat smoothing` - Model tuning
- `SC driving` - New driving mode
- `MacroStiff Model` - New model architecture
- `VW: Enable torqued` - VW torque control
- `switch from mypy to ty` - Type checker change
- Various dependency updates and tooling improvements

## Impact

- Affected specs: None (maintenance sync)
- Affected code: Entire codebase receives upstream updates
- Risk: Low - standard upstream sync operation
- **IMPORTANT**: Must resolve any conflicts with our local changes
