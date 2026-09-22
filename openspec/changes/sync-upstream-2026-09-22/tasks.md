# Tasks: Sync Fork with Upstream Master (2026-09-22)

## 1. Triage (new tooling)
- [x] 1.1 Add `openpilot/tools/upstream_sync`: git facts, TypeSafe (Jev) judgments, and explicit policy into resolve/adapt/read/routine; tests (13); README with calibration
- [x] 1.2 Calibrate on the September sync (`499644e1c`..`0cf294d85`, 394 commits, 25 known fork-work commits): v1 (signals only) flagged 171 commits for 23/25; v2 (with fork references + `breaks_fork_code`, interface/invocation signals no longer escalate alone) flagged 94 for 22/25; git facts alone flag 71 for 18/25
- [x] 1.3 `.cbmignore`: codebase-memory-mcp skips every `tools`/`scripts` directory, which hid `openpilot/tools` (most fork-owned code); re-index covers them (185,726 nodes)
- [x] 1.4 Run on `develop..upstream/master` (86 commits): 1 resolve, 10 adapt, 6 read, 69 routine, 1 safety note
- [x] 1.5 Post-merge lesson: the DM model commit (#38942, safety p=0.86) *added* `sleepProb`, which broke the fork's monitoring mocks without removing any name, so it sat in routine. Added **fork importers** (fork files importing a module the commit changes). A safety-relevant commit with fork importers now goes to adapt. Re-run (`triage.md`): 1/11/6/68, and the September calibration is unchanged at 22/25 recall, 96 flagged.

## 2. Merge
- [x] 2.1 Merge upstream `521db4c82` into `upstream-sync-20260922` (86 commits); submodules updated (msgq, opendbc, panda, rednose, tinygrad)
- [x] 2.2 `scripts/lint/lint.sh` conflict: took upstream's array-based file collection plus `SHELL_FILES`; fork's Termux exclusion is now `FORK_EXCLUDE`, applied to both loops
- [x] 2.3 `scripts/lint/check_shebang_format.sh` conflict: upstream's `"$@"` quoting plus the fork's `grep -a`
- [x] 2.4 Fork additions to `tests.yaml`, `docs.yaml`, `SConstruct`, `pyproject.toml` all survived the merge (line-level check); upstream's only changes there are the LFS exclude path, the docs runner (commaai-only expression), and the removed build-size check
- [x] 2.5 `uv.lock` taken as upstream's. It still lists `spidev`, which panda dropped (panda #2430), so `uv lock --check` fails upstream too. Harmless under `uv sync --frozen`, and left for upstream's next lock refresh to keep the fork diff at zero.

## 3. Adapt fork code
- [x] 3.1 `tools/dgx` → recurrent driving model (#38916) + deleted `get_model_metadata` (#38922): fork-owned `model_metadata.py` (adds `input_dtypes`, `state_pairs`); `build_model_inputs()`; `generate_labels()` carries recurrent state (`sequential=True` threads it); train/model_runner/benchmark_inference use the new inputs; contract tests rewritten
- [x] 3.2 Latent bug: DGX precision probe tested `GPUInfo.supports_*` without calling them (always FP4; warnings silenced by `type: ignore`)
- [x] 3.3 `test_auth.py` → `auth.login()` rewrite (#38893). The triage put the commit in "read" with `breaks_fork_code` 0.48, and it was a real break. Added declined and provider-mismatch cases.
- [x] 3.4 `check_shell` (#38877): 17 findings in four fork scripts fixed with arrays and quoting; `coverage.sh`'s `-m` filter now actually applies; `cpp-coverage.sh` and `coverage-check.sh` point at paths that exist (the three C++ tests upstream still builds, `openpilot/common`)
- [x] 3.5 Monitoring (#38942 adds a `sleep` distraction type, `sleepProb > 0.75`): taken as upstream wrote it. The fork's `test_monitoring_policy.py` mocks now provide `sleepProb` (17 failures), plus three new tests that the sleep check flags distraction end to end.
- [x] 3.6 Graph check (codebase-memory): of the 17 functions whose signatures changed in this range, the only fork callers are the auth tests from 3.3; no references remain to the five deleted modules (checked with `git grep`, since deleted modules have no graph nodes)

## 4. Validate
- [x] 4.1 `scripts/lint/lint.sh`: 10/10 (after `uv sync --frozen --all-extras`; the stale local `ty` 0.0.56 reported false positives that 0.0.80 does not)
- [x] 4.2 `scons -u -j8`: success (11m51s), including the new tinygrad model and warp artifacts
- [x] 4.3 `pytest -m "not slow" -n 8` (macOS): 3752 passed, 27 failed before the monitoring fix, which cleared 17. The remaining 10:
      - known macOS-only (July notes): `test_messaging::test_recv_one_retry`, `test_uploader` (4), `test_athenad` (2)
      - `test_logmessaged` (2): receives no messages. Upstream-owned; it goes through msgq plus a managed subprocess (the macOS spawn path behind the known failures), and upstream reverted a msgq bump in this range (#38920). Unverified locally; CI decides.
      - `test_plotjuggler::test_demo`: route download plus GUI launch timeout (environment)
- [ ] 4.4 CI on the sync PR (authoritative, including process replay)

## 5. Land
- [ ] 5.1 Push `upstream-sync-20260922` to origin and open the PR against develop (needs the user's go-ahead)
- [ ] 5.2 Merge (the user's call); fast-forward origin `master` mirror to `521db4c82`

## 6. Follow-ups
- [ ] 6.1 Run the ported DGX TensorRT paths on the Spark (engine build with uint8 `state_img_q`, `generate_labels(sequential=True)` round trip)
- [ ] 6.2 Triage tool: fork references include third-party API names that upstream merely stopped calling (`unsqueeze`, `array_equal`). `breaks_fork_code` discounts them, but a definitions-only filter would cut noise at some recall cost. Evaluate against the September labels before changing.
- [ ] 6.3 Carried over from the September sync: 6.1–6.6 there (np.random migration, ty override cleanup, dependency budget, MISRA baseline, SonarCloud token, test-suite consolidation)
