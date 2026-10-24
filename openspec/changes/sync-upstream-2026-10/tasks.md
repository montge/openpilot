# Tasks: Sync Fork with Upstream v0.11.2 (October 2026)

> **Status:** merge, adaptation and lint are DONE locally; the local build/test run (4.4) is still
> in flight and will be recorded here. Resume at 5.1 (CI on the sync PR) and then 5.2 (merge — the
> user's call).

## 1. Analysis
- [x] 1.1 Fetch upstream tag `v0.11.2` (`6939d172`, 2026-10-14); measure divergence: 162 ahead / 162 behind, merge-base `0cf294d8` (2026-09-11)
- [x] 1.2 Compute the conflict surface: 8 files changed on both sides (`docs.yaml`, `tests.yaml`, `.gitignore`, `SConstruct`, `pyproject.toml`, `check_shebang_format.sh`, `lint.sh`, `uv.lock`) → 2 actual conflicts
- [x] 1.3 Intersect upstream's 22 deletions with fork-touched files: none overlap, but `get_model_metadata.py` is imported by five fork-owned DGX files

## 2. Merge & Conflict Resolution
- [x] 2.1 `git merge --no-ff upstream-v0.11.2` (merge commit `8a5f57e`), 2 content conflicts resolved
- [x] 2.2 `scripts/lint/lint.sh`: upstream's array rewrite + new `check_shell` step; the fork's Termux exclusion (`openpilot/tools/shadow/setup/{camera_stream,setup-ssh,termux-setup}.sh`) applied as a `FORK_EXCLUDE` regex to both the `ALL_FILES` and `SHELL_FILES` loops
- [x] 2.3 `scripts/lint/check_shebang_format.sh`: upstream's `"$@"` quoting, fork's `grep -a` kept
- [x] 2.4 Verified the clean merges did not silently revert fork overrides (the September 5.1 pattern): `tests.yaml` still runs the fork's pytest step, only the `lfs.fetchexclude` target changed (`big_driving_supercombo.onnx` → `big_driving_tinygrad.pkl`, also in `docs.yaml`, `release.yaml`, `repo-maintenance.yaml`); `pyproject.toml` keeps the pytest extras and `[tool.pytest.ini_options]`; root `conftest.py` intact
- [x] 2.5 Submodule pointers taken from upstream (msgq, opendbc, panda, rednose, tinygrad); `uv lock --check` failed on the merged lock (panda's `pyproject.toml` dropped `spidev` and its pytest dev deps for `comma-deps-*`), regenerated with `uv sync`/`uv lock`

## 3. API Adaptation
- [x] 3.1 `openpilot/selfdrive/modeld/get_model_metadata.py` deleted upstream → ported verbatim as fork-owned `openpilot/tools/dgx/model_metadata.py` (tinygrad's `OnnxPBParser` API it builds on is unchanged at the new pointer); imports in `model_runner.py`, `training/teacher.py`, `training/train.py`, `tests/test_model_inference.py` and the two DGX docs repointed
- [x] 3.2 `CLAUDE.md`: big-model ONNX no longer in-tree; metadata helper location

## 4. Lint & Validation
- [x] 4.1 New `check_shell` lint: 17 findings in four fork scripts fixed — quoted `$TIMEOUT`/`$MIN_COVERAGE`/`$RESULT` (`coverage-check.sh`), `PYTEST_ARGS`/`BINARIES`/`TARGET_DIRS` turned into bash arrays (`coverage.sh`, `cpp-coverage.sh`, `cppcheck-misra.sh`), `-j"$(nproc)"`, `./*.profraw`
- [x] 4.2 `ty` 0.0.80: three `GPUInfo.supports_*` methods tested as attributes in `tools/dgx/model_runner.py` (and one in its test) — a real pre-existing bug hidden by `# type: ignore[truthy-function]`; now called
- [x] 4.3 `scripts/lint/lint.sh` green: 10/10 checks (ruff, check_shell, check_dependencies, check_indentation, check_added_large_files, shebang ×2, nomerge, ty, codespell)
- [x] 4.4 Build: `scons -k` succeeds except the modeld tinygrad pickle targets, whose ONNX inputs are Git-LFS objects on huggingface.co that this environment's network policy blocks (as in July/September). CI has LFS access.
- [x] 4.5 Tests: **3578 passed, 178 skipped, 1 xfailed, 58 failed, 12 errors** (`pytest -m "not slow"`, CI's `RAYLIB_BACKEND=headless MAX_EXAMPLES=1 -n logical`). Every failure is accounted for:
      - 4 collection errors + 17 `test_monitoring_policy` + 3 `test_auth` are upstream API drift in fork-owned tests, all fixed on the PR (5.1, 5.1a)
      - 45 need network this environment denies (`api.commadotai.com`, `huggingface.co`, `commadist.azureedge.net`), the same list as September 4.7: `test_paramsd` (27), `test_locationd_scenarios` (8 errors), `test_logreader` (4), `test_lagd` (2), `test_caching`, `tools/dgx test_dataloader`, `test_agnos_updater`
      - 1 `test_raylib_ui`, upstream-owned: starts the real `ui`, which segfaults with no GPU / `DISPLAY`
      - 1 `tools/shadow test_visualize` failure per full parallel run, a different test each time
        (`test_plot_error_histogram`, `test_generate_all_plots`, `test_plot_time_series_steer`). Root-caused
        with `PYTHONTRACEMALLOC`: it is a `ResourceWarning: unclosed file` that pytest's unraisable-exception
        hook turns into a failure of whichever test happens to trigger garbage collection. The file is
        allocated at upstream's `test_logreader.py:141` (`test_direct_parsing`), which opens a
        `NamedTemporaryFile(mode='wb', delete=False)` and only closes it inside a `with` that follows a
        `requests.get(QLOG_FILE)`. With this environment's network policy the fetch raises first and the
        handle leaks. With network (CI) the file is closed normally, so this cannot occur there. Not a fork
        problem and not fixed here: the leaking test is upstream-owned and the fix belongs upstream
        (open the temp file inside the `with`)
      CI covers all of these; the sync PR is the authoritative check.

## 5. Land on develop
- [x] 5.1 Draft PR montge/openpilot#64 opened against develop. First CI round: `unit tests` failed at
      collection on four fork-owned test modules importing symbols upstream removed (`a47d1c5` "remove
      unused code", `42ccc5c` "Remove experimental mode initial cruise speed override"):
      `managed_proc` (common/utils), `V_CRUISE_INITIAL_EXPERIMENTAL_MODE` (selfdrive/car/cruise),
      `clear_token` (tools/lib/auth_config). Dropped the tests of the deleted code and retargeted the two
      experimental-mode cruise tests to assert the new behavior (experimental mode initializes at
      `V_CRUISE_INITIAL` too). 143 tests in those files pass locally. Same pattern as September 2.6:
      upstream deletes code the fork had covered with its own tests.
- [x] 5.1a Two more upstream API drifts surfaced by the local run, both fixed: `DM: Super Leicht Model` (#38942)
      added `sleepProb` to the driver data and a `sleep` distracted type, so the policy test's mocked
      `driver_data` needed the field (17 failures, `MagicMock > float`); `tools: share cancellable browser
      sign-in` (#38893) made `ClientRedirectServer.query_params` per-instance, replaced `webbrowser` with
      an `xdg-open`/`open` subprocess, and made `login()` return a status dict — the three login tests
      rewritten around that (plus one for the declined-callback path)
- [ ] 5.1b CI green on the sync PR (LFS-dependent model tests, process replay, network-dependent suites)
- [ ] 5.2 Merge to develop (the user's call); confirm origin/develop carries the merge

## 6. Post-Sync Follow-ups (separate changes, tracked here for pickup)
- [ ] 6.1 Chestnut offroad power: 0.11.2 never drops the GPU rail (`link_up()` only ever sends 0xF3 value 1). The user's chestnut 12 V is on an always-on circuit, so the idle GPU draw persists while parked. Candidate fork change: send 0xF3 value 0 after modeld stops on the offroad transition and value 1 before onroad processes start (cf. pjtnt11/openpilot#1, closed unmerged)
- [ ] 6.2 September follow-ups still open: 6.1 (np.random.Generator), 6.2 (ty overrides), 6.3 (dependency budget), 6.4 (July carry-overs), 6.5 (SonarCloud token + action migration), 6.6 (duplicate longcontrol/drive_helpers suites)
