# Change: Sync Fork with Upstream Master (September 2026)

## Why
The fork was 394 commits behind upstream/master (merge-base `d606014c`, 2026-07-03) and 291 ahead. Upstream spent this window tightening its own foundations — it replaced pytest with a unittest harness, renamed every "live" cereal service, and removed the blanket lint suppressions it had been carrying — so the longer the fork stayed diverged the more of that tightening would land at once. This sync absorbs upstream `0cf294d8` (2026-09-11).

## What Changes

- Merge upstream `0cf294d8` (14 content conflicts, 674 files changed upstream):
  - **pytest removed upstream** in favour of `OpenpilotTestCase` (`openpilot/common/test.py`) run via `tools/op.sh test` (`tools/test_runner.py`); the root `conftest.py` and `pytest-cpp` harness were deleted and the `testing` extra trimmed to coverage/ty/ruff/codespell
  - **"live" cereal services renamed** (#38601): `liveCalibration`→`extrinsicsCalibration`, `liveParameters`→`vehicleParameters`, `liveTorqueParameters`→`lateralTorqueParameters`, `liveDelay`→`lateralDelay`, `livePose`→`deviceMotion`, `liveTracks`→`radarTracks`, plus the matching struct names
  - **hardware/tici → hardware/comma** (#38580/#38581), `TICI` → `COMMA_HARDWARE`
  - `car_specific.py` → `car_events.py`, `CarSpecificEvents` → `CarEvents` (#38493)
  - Longitudinal control: `LongCtrlState.starting` removed along with `long_control_state_trans`'s `CP`/`v_ego` args; `get_accel_from_plan` returns `a_target` alone with `should_stop()` split out; `limit_accel_in_turns` folded into a new `get_cruise_accel`
  - `test_models.py` moved to opendbc (#38441); emoji support removed (#38361); `lateral_mpc_lib`, `params_pyx.pyx`, `feedbackd`, `regen`, and the top-level submodule symlinks deleted; LFS hosting moved to Hugging Face (#38841)
  - New upstream lints: `check_indentation.py` and `check_dependencies.py` (a hard dependency budget)
- **Keep pytest as the fork's runner.** pytest executes upstream's `OpenpilotTestCase` classes natively, and the fork's coverage/Sonar/codecov gates are all built on pytest-cov. Restored the pytest deps and `[tool.pytest.ini_options]`, and re-added the root `conftest.py` as fork-owned with its autouse prefix fixture stepping aside for `OpenpilotTestCase` subclasses.
- Adapt fork-owned code to the API changes above (two parallel copies of the longcontrol and drive_helpers test suites, the manager process tests, radard, paramsd, torqued, calibrationd, logreader, power_monitoring, deleter, git).
- Reconcile fork-owned code with upstream's tightened lint: fixed every finding in shared code; scoped the six `ty` rule ignores and `NPY002` to the fork's numpy/torch tooling; reindented six Termux scripts; raised the dependency budget to the fork's measured footprint.
- Fix the SonarCloud workflow's coverage paths, stale since the July restructure (`--cov=selfdrive/system/tools/common` measured nothing under the `openpilot/` package layout).

## Impact
- Affected specs: none — `development-workflow` already covers upstream synchronization and its requirements are unchanged
- Affected code: fork-owned tests and tooling plus lint configuration; no upstream behavior modified beyond the documented conflict resolutions
- Risk: Medium — large mechanical surface, but 9/9 lint checks pass and the test suite was run locally; Linux CI on the sync PR is the authoritative check
- Known limitation: the ONNX driving models are Git LFS objects hosted on Hugging Face, which this environment's network policy blocks, so `scons` cannot compile the modeld tinygrad pickles here and the model-dependent tests could not be exercised locally. CI has LFS access and covers them.
- Deliberate divergence recorded in-tree: the fork keeps pytest (and pays for it in the dependency budget), and relaxes six `ty` rules plus `NPY002` for its own tooling. Both are scoped and commented at the point of divergence.
