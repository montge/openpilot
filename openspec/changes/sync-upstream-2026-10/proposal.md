# Change: Sync Fork with Upstream v0.11.2 (October 2026)

## Why
The fork's `develop` sat at the September sync (merge-base `0cf294d8`, 2026-09-11), 162 commits
behind upstream. Upstream tagged **v0.11.2** at `6939d172` (2026-10-14): the 880M-parameter big
driving model, chestnut (external GPU) support, and the tinygrad compiler-artifact pipeline that
replaced the ONNX-at-runtime model loading. The user runs a chestnut on a VW/Audi MQB car, so the
fork needs to track the release the device runs.

## What Changes

- Merge upstream tag `v0.11.2` (2 content conflicts, 292 files changed on the merge):
  - **modeld now loads tinygrad compiler artifacts** (#38922, #38926, #38933, #38941, #38956):
    `get_model_metadata.py`, `compile_modeld.py`, `compile_dm_warp.py` and `common/file_chunker.py`
    are deleted; the chestnut big model ships only as `big_driving_tinygrad.pkl` (Git LFS) and its
    ONNX is gone from the tree
  - **chestnut**: firmware flashing moved to a startup condition ("flash before going onroad",
    #39078), a new `chestnut/monitoring.py` thread in hardwared polls the USB bridge offroad,
    `chestnutGpuState` split out of `chestnutState`, reworked offroad alerts (#39049-#39056),
    GPU power limit defaulted to 100 W (#38992), big-model load retried once after a crank (#39086)
  - **lint**: `lint.sh` rewritten around bash arrays with a new `check_shell` lint (a minimal
    shellcheck) over every tracked `*.sh`; `check_shebang_format.sh` quoted
  - **packaging**: `msgq-ipc` renamed to `msgq`; `release` detection keys off `.gitmodules`;
    `SConstruct` drops the build-product size check; submodule bumps (msgq, opendbc, panda,
    rednose, tinygrad); AGNOS 20.1; nav alpha; `tools/scripts/profiling` removed
- Conflict resolutions, both in `scripts/lint`: upstream's array-based `lint.sh` with the fork's
  Termux-script exclusion applied to both the `ALL_FILES` and the new `SHELL_FILES` loops;
  upstream's quoted `check_shebang_format.sh` keeping the fork's `grep -a`
- Fork adaptations:
  - The DGX tooling still works from ONNX files and needs the embedded metadata, so the deleted
    parser lives on as fork-owned `openpilot/tools/dgx/model_metadata.py`; five imports and two
    docs repointed
  - `check_shell` findings in four fork scripts (`coverage-check.sh`, `coverage.sh`,
    `cpp-coverage.sh`, `cppcheck-misra.sh`): quoted expansions, word-split strings turned into arrays
  - Newer `ty` (0.0.80) exposed a real bug: `tools/dgx/model_runner.py` tested
    `GPUInfo.supports_nvfp4/fp8/bf16/fp16` as attributes (always truthy, so the precision pick never
    fell past FP4) under a `# type: ignore[truthy-function]`; now called. Same in the DGX test
  - `uv.lock` regenerated: the clean merge kept panda's stale `spidev`/pytest metadata, which no
    longer matches the submodule's `pyproject.toml`, so `uv lock --check` failed
  - `CLAUDE.md` updated for the removed big-model ONNX and the relocated metadata helper

## Impact
- Affected specs: none — `development-workflow` already covers upstream synchronization
- Affected code: fork-owned tooling (`tools/dgx`, `scripts/`) and lint configuration; no upstream
  behavior modified beyond the two documented conflict resolutions
- Risk: Low-Medium — small conflict surface (2 files), lint 10/10 green locally; Linux CI on the
  sync PR is the authoritative check
- Known limitation (unchanged from July/September): the model ONNX files are Git LFS objects on
  Hugging Face, which this environment's network policy blocks, so `scons` cannot build the modeld
  tinygrad pickles here. CI has LFS access.
- Not in this change: nothing in 0.11.2 powers the chestnut's GPU rail down while the car is off.
  The only rail control is `link_up()` (0xF3 value 1, rail on) in `chestnut/flash.py`. With the
  chestnut's 12 V fed from an always-on circuit the idle GPU draw (community-measured ~2 A at
  13.6 V) persists while parked. A fork-side offroad rail-off is a candidate follow-up.
