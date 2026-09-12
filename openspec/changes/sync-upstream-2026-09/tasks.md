# Tasks: Sync Fork with Upstream Master (September 2026)

> **Status:** merge, adaptation, lint and validation are DONE, and **CI confirms it** —
> `unit tests` on PR #58 reports 3761 passed / 0 failed, including every test this
> environment could not run. All 26 checks are green. Resume at section 5.2 (merge —
> the user's call) and then the follow-ups (section 6). Note 5.1c: SonarCloud's green
> is not real; its scan never ran.

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
- [x] 5.1 Draft PR montge/openpilot#58 opened against develop. CI needed four fixes, all of them
      fork-owned config the merge had silently reverted or outdated — not problems with the merged code:
      - `algorithm-harness-coverage.yml` (both jobs): `PYTHONPATH=$GITHUB_WORKSPACE` no longer resolves
        `import opendbc` now that upstream deleted the top-level submodule symlinks. Added the submodule
        roots, mirroring SConstruct's `submodule_python_paths`.
      - `cpp-coverage.yml`: upstream's #38408 cut the C++ suite to three Program targets.
        `openpilot/tools/replay/tests` no longer exists (hard scons error) and five of the six binary
        names were stale — silently skipped by `|| true` / `if [[ -f ]]`, so this gate had been measuring
        almost nothing. Now measures 43.54% against its 15% threshold.
      - `tests.yaml`: the merge took upstream's `tools/op.sh test` over the fork's pytest step and dropped
        the `PYTEST` env var. That runner only discovers `unittest.TestCase` subclasses, so the fork's
        pytest classes would have been skipped silently, and module-level `pytest.importorskip` (9 files)
        raises `Skipped` out of the loader — which is what failed. Restored the pytest invocation, keeping
        upstream's `RAYLIB_BACKEND=headless`.
      Pattern worth remembering for the next sync: upstream changes *how* something is invoked, the fork's
      override disappears in a clean merge, and a guard hides the consequence.
- [x] 5.1b CI green on head `5dcc6525`: all 26 checks pass.
      **`unit tests`: 3761 passed, 175 skipped, 1 xfailed, 0 failed** — the
      45 network-blocked tests from 4.7 (test_paramsd, test_locationd_scenarios, test_logreader, test_lagd,
      test_caching, test_url_file, dgx test_dataloader, test_agnos_updater) and `test_raylib_ui` all pass
      with real network and a real runner. `process replay`: 0 changed, 66 passed, 0 errors — the merge
      does not alter driving behavior. The three modeld tinygrad pickles build fine with LFS access.
- [x] 5.1c **SonarCloud's green is an artifact of `continue-on-error: true`, not a passing scan.**
      The "Run tests with coverage" step succeeds (4m48s) and now produces real coverage thanks to the
      `--cov=openpilot` fix in 4.6, but the scan step itself exits 1 after 3 seconds and the coverage is
      never uploaded:
      ```
      ERROR Failed to query JRE metadata: . Please check the property sonar.token
            or the environment variable SONAR_TOKEN.
      INFO  EXECUTION FAILURE
      ```
      This is exactly the rejected token predicted by the July sync's 7.7, so nothing about it is new or
      caused by this merge — but it means the fork has had **no** Sonar analysis since that token expired,
      and the corrected coverage paths will not produce numbers until it is replaced. Two useful details
      for 6.5: the deprecated `sonarcloud-github-action@master` already resolves internally to
      `sonarqube-scan-action`, so that half of the migration is mostly a rename; and the token is the only
      thing standing between the fixed coverage paths and real numbers. Waiting on SonarCloud is therefore
      not a reason to hold 5.2 — it cannot report until 6.5 is done.
- [ ] 5.2 Merge to develop (the user's call); confirm origin/develop carries the merge

## 6. Post-Sync Follow-ups (separate changes, tracked here for pickup)
- [ ] 6.1 Migrate the fork's ~230 legacy `np.random` calls to `np.random.Generator` and drop the `NPY002` per-file ignores. This changes the random streams, so the deterministic harness and stonesoup benchmark expectations need re-baselining — do it deliberately, with before/after results compared.
- [ ] 6.2 Clear the 57 `ty` findings behind `[[tool.ty.overrides]]` (algorithm_harness 22, fair 11, stonesoup 10, shadow 6, dgx 4, algo_bench 3, trackers 1) and remove the override block, matching upstream's strictness everywhere.
- [ ] 6.3 Drive the dependency budget in `scripts/lint/check_dependencies.py` back down toward upstream's 37/65/550. The gap is pytest + xdist/mock/cov/subtests/timeout, hypothesis, opencv-python-headless and matplotlib; consider whether the fair/shadow tooling needs opencv in the default extras.
- [ ] 6.4 Carried over from the July sync, still open: regenerate `reports/misra-baseline.txt` (7.2), shadow device rebuild (7.3), pre-restructure paths in openspec docs and 3 failing spec validations (7.4), CUDA build-time selection on the DGX box (7.5), algorithm-harness coverage back toward 90% (7.6)
- [ ] 6.5 SonarCloud (July 7.7, still open and now confirmed failing — see 5.1c): regenerate the rejected `SONAR_TOKEN`, migrate from the deprecated `sonarcloud-github-action@master` to `sonarqube-scan-action`, then drop `continue-on-error` from the scan step so the gate can never report green without scanning again. The coverage paths it feeds are fixed as of this change (4.6), so the scan should report real numbers once the token works.
- [ ] 6.6 The two parallel copies of the longcontrol and drive_helpers test suites (`controls/tests/` and `controls/lib/tests/`) now cover nearly the same ground and both needed the same edits this sync. Consider consolidating them.
