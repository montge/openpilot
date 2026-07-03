# Change: Sync Fork with Upstream Master (July 2026)

## Why
The fork was 204 commits behind upstream/master (merge-base `d7c562e13`, 2026-05-09). Upstream landed a major repo restructure plus a modeld rewrite, and staying diverged made every future merge harder. This sync absorbs upstream `07ec389f4` (2026-07-02) and re-fits all fork features to the new layout.

## What Changes
- Merge upstream `07ec389f4` (branch `upstream-sync-20260702`, PR montge/openpilot#46):
  - Repo restructure (`5edc0bd89`): all code now lives in a real `openpilot/` package; root `cereal/`, `common/`, `selfdrive/`, `system/`, dev `tools/` removed; `third_party/` deleted (deps are vendored pip packages)
  - Hardware HAL moved to `openpilot/common/hardware/` (import path change from `openpilot.system.hardware`)
  - `car.capnp` moved to opendbc (`from opendbc.car.structs import car`); bare `cereal` imports removed (`openpilot.cereal.*`)
  - modeld rewritten: precompiled tinygrad pickles, build-time device selection (CUDA auto-selected), combined `driving_supercombo.onnx` replaces split vision/policy models
  - `bodyteleop`/webjoystick and `statsd` deleted; Params `*_nonblocking` API removed; mypy replaced by `ty`
  - Submodule bumps: msgq, opendbc, panda, tinygrad, rednose, teleoprtc
- Relocate fork features into the new layout: `openpilot/tools/{shadow,stonesoup,dgx,fair}`, nvidia HAL + `shadow_mode.py` to `openpilot/common/hardware/`
- Sweep ~170 stale imports/paths across 112 fork-owned files; adapt fork tests to upstream API changes
- Repair fork CI (much of it pre-existing debt): LFS checkout strategy, dependency closures, achievable coverage gates, CodeQL fork-side alerts

## Impact
- Affected specs: development-workflow (upstream synchronization)
- Affected code: everything fork-owned (relocations + import sweep); no upstream behavior modified beyond documented conflict resolutions
- Risk: Medium — large mechanical surface, but validated by full local build, 7/7 lint checks, ~2,850 passing tests, and Linux CI on PR #46
- Known obsolescence accepted: fork's runtime `DEV=CUDA` modeld hook dropped (upstream selects CUDA at build time); `bodyteleop` SRI hardening dropped with the tool
