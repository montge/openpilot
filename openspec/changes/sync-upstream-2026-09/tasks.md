# Tasks: Sync Fork with Upstream Master (September 2026)

> **Status:** merge, adaptation, lint and local validation are DONE — 9/9 lint checks and
> 3712 tests pass; the 46 that don't are all blocked by this environment's network policy
> or its lack of a GPU, never by the code (see 4.7).
> Resume at section 5 — land on develop — then work the follow-ups (section 6).

## 1. Analysis
- [x] 1.1 Fetch upstream, measure divergence (291 ahead / 394 behind, merge-base `d606014c`, 674 files changed upstream)
- [x] 1.2 Map upstream structural changes: 103 deletions, 19 renames (hardware/tici→comma, car_specific→car_events, driver_camera_dialog→cabin_camera_dialog), top-level submodule symlinks removed, LFS moved to Hugging Face
- [x] 1.3 Compute the conflict surface by intersecting fork-changed and upstream-changed files (22 candidates → 14 actual conflicts)

## 2. Merge & Conflict Resolution
- [x] 2.1 Merge upstream/master (`dd26017e`), 14 content conflicts resolved
- [x] 2.2 Keep pytest as the fork's runner: restore pytest deps in `[testing]`, restore `[tool.pytest.ini_options]` (dropping pytest-cpp, whose harness upstream deleted), re-add the root `conftest.py` as fork-owned
- [x] 2.3 `conftest.py`'s autouse prefix fixture steps aside for `OpenpilotTestCase` subclasses, which enter their own `clean_env` + `OpenpilotPrefix` in `run()`; the `tici` marker is replaced by the `COMMA_HARDWARE_TEST` class attribute
- [x] 2.4 Apply the "live" cereal rename map across 11 fork-owned test files (281 lines), leaving the `LiveParametersV2` / `LiveTorqueParameters` Params *keys* alone — those are unchanged upstream
- [x] 2.5 Hardware: upstream's `COMMA_HARDWARE`/`HardwareComma`/`HardwarePc` selection with the fork's NVIDIA branch and `shadow_mode` re-exports layered on
- [x] 2.6 Accept upstream deletions of fork-tested code: `test_models.py` (moved to opendbc), `emoji.py` (#38361), `migrate_cached_vehicle_params_if_needed`, `get_normalized_origin`
- [x] 2.7 Submodule pointers taken from upstream; `uv lock` regenerated (adds only the fork's pytest stack and its transitive deps)

## 3. API Adaptation
- [x] 3.1 `long_control_state_trans` lost `CP`/`v_ego` and the STARTING state — rewrote both copies of the longcontrol tests (`controls/tests/` and `controls/lib/tests/`) and the hypothesis property tests
- [x] 3.2 `get_accel_from_plan` returns `a_target` alone; `should_stop(v_ego, a_target)` split out — retargeted both copies of the drive_helpers tests, added `TestShouldStop`
- [x] 3.3 `limit_accel_in_turns` folded into `get_cruise_accel` — replaced both `TestLimitAccelInTurns` classes with `TestCruiseAccelInTurns` exercising the same lateral-accel budget through the new entry point
- [x] 3.4 `Track.update` lost the `measured` parameter and attribute (19 call sites)
- [x] 3.5 `ensure_running` requires params/CP and takes a `ValuesView`; `PythonProcess` lost `restart_if_crash` along with the auto-restart branch (added `as_procs()` helpers)
- [x] 3.6 Smaller signature changes: `auth_redirect_link(method, port)`, `PointBuckets` wanting `Sequence[float]` (`.tolist()`, as upstream does), typed `PIDController` gains, `CarSpecificEvents`→`CarEvents`
- [x] 3.7 `test_deleter.py`: restore `time`/`threading` imports dropped by the auto-merge, and use `super().openpilot_setup_method()` — `OpenpilotTestCase` renames `setup_method` on subclasses, so `super().setup_method()` is `None`

## 4. Lint & Validation
- [x] 4.1 `ty`: upstream dropped six repo-wide ignores. Every occurrence in shared code fixed; the six scoped to the fork's numpy/torch tooling via `[[tool.ty.overrides]]`
- [x] 4.2 `ruff`: upstream added rules, dropped ignores, and removed the exclude list that kept `openpilot/cereal` and notebooks unlinted. 64 findings fixed; `NPY002` scoped per-file for the same tooling
- [x] 4.3 New upstream lints: reindented six Termux scripts for `check_indentation` (verified by AST equality); raised `check_dependencies` limits to the fork's measured 47/85/750 with the reasoning recorded in the script
- [x] 4.4 `scripts/lint/lint.sh` green: 9/9 checks
- [x] 4.5 Build: full `scons` succeeds except the three modeld tinygrad pickle targets, which are derived from Git-LFS ONNX files that this environment's network policy cannot fetch (LFS is hosted on huggingface.co as of #38841). CI has LFS access.
- [x] 4.6 Fix the SonarCloud workflow's stale coverage paths (`--cov=selfdrive/system/tools/common` → `--cov=openpilot`), drop the `--ignore` for the deleted `third_party`, narrow `sonar.python.version` to 3.12
- [x] 4.7 Tests: **3712 passed, 178 skipped, 46 failed/errored** (`pytest -m "not slow"`). Started at 199 failures + 14 errors on the raw merge. Every one of the 46 that remain is blocked by this environment, not by the code:
      - 45 need network this session's policy denies — `api.commadotai.com`, `huggingface.co`, `commadist.azureedge.net` — so any test that downloads a route or the AGNOS manifest fails: `test_paramsd` (27), `test_locationd_scenarios` (8), `test_logreader` (4), `test_lagd` (2), `test_caching`, `test_url_file`, `tools/dgx test_dataloader`, `test_agnos_updater`
      - 1 is `test_raylib_ui`, upstream-owned and untouched by the fork: it starts the real `ui` process, which segfaults with no GPU (`/dev/dri` absent) and no `DISPLAY`, even with `RAYLIB_BACKEND=headless`
      CI covers both categories; the sync PR is the authoritative check.

## 5. Land on develop
- [ ] 5.1 Open a draft PR from `claude/upstream-sonarqube-check-mqhc8f` against develop and let Linux CI validate — it is authoritative for the model-dependent tests that cannot run without LFS
- [ ] 5.2 Merge once green; confirm origin/develop carries the merge

## 6. Post-Sync Follow-ups (separate changes, tracked here for pickup)
- [ ] 6.1 Migrate the fork's ~230 legacy `np.random` calls to `np.random.Generator` and drop the `NPY002` per-file ignores. This changes the random streams, so the deterministic harness and stonesoup benchmark expectations need re-baselining — do it deliberately, with before/after results compared.
- [ ] 6.2 Clear the 57 `ty` findings behind `[[tool.ty.overrides]]` (algorithm_harness 22, fair 11, stonesoup 10, shadow 6, dgx 4, algo_bench 3, trackers 1) and remove the override block, matching upstream's strictness everywhere.
- [ ] 6.3 Drive the dependency budget in `scripts/lint/check_dependencies.py` back down toward upstream's 37/65/550. The gap is pytest + xdist/mock/cov/subtests/timeout, hypothesis, opencv-python-headless and matplotlib; consider whether the fair/shadow tooling needs opencv in the default extras.
- [ ] 6.4 Carried over from the July sync, still open: regenerate `reports/misra-baseline.txt` (7.2), shadow device rebuild (7.3), pre-restructure paths in openspec docs and 3 failing spec validations (7.4), CUDA build-time selection on the DGX box (7.5), algorithm-harness coverage back toward 90% (7.6)
- [ ] 6.5 SonarCloud (July 7.7, still open): regenerate the rejected `SONAR_TOKEN`, migrate from the deprecated `sonarcloud-github-action@master` to `sonarqube-scan-action`, then drop `continue-on-error` from the scan step. The coverage paths it feeds are fixed as of this change (4.6), so the scan should report real numbers once the token works.
- [ ] 6.6 The two parallel copies of the longcontrol and drive_helpers test suites (`controls/tests/` and `controls/lib/tests/`) now cover nearly the same ground and both needed the same edits this sync. Consider consolidating them.
