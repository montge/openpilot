# Change: Sync Fork with Upstream Master (late September 2026)

## Why
Ten days after the September sync landed (PR #58, upstream `0cf294d8`), the fork was 86 commits behind upstream/master again. Most of them were cabana and UI polish, but a run of modeld commits reshaped the driving model contract that the fork's DGX tooling is built on. This sync absorbs upstream `521db4c8` (2026-09-20) while the gap is small.

It is also the first sync planned with the new triage tool (`openpilot/tools/upstream_sync`). The tool ranked the 86 commits into 1 resolve, 10 adapt, 6 read, and 69 routine before the merge ran (`triage.md` in this change).

## What Changes
- Add `openpilot/tools/upstream_sync`: git facts (incoming commits, `git merge-tree` conflicts, fork overlay, fork references to removed or re-signatured names) plus TypeSafe (Jev) judgments per commit, combined by explicit weights into a worklist. It was calibrated on the September sync, where 22 of the 25 known fork-work commits landed in resolve or adapt with 24% of commits flagged. Add `.cbmignore` so codebase-memory-mcp indexes `openpilot/tools/` and `scripts/`, which its built-in skip list hid.
- Merge upstream `521db4c8` (2 content conflicts, both in fork-edited lint scripts, both predicted):
  - **Recurrent driving model** (#38916): history queues moved into the ONNX. Inputs are now `new_img` (latest road+wide frame), `desire`, `traffic_convention`, `action_t` and `state_{img,desire,feat}_q`, fed back from `next_state_*` outputs. `features_buffer`, `desire_pulse` and the 12-channel `img`/`big_img` are gone.
  - **modeld compiles with tinygrad's generic ONNX compiler** (#38864, #38922, #38926, #38933): `compile_modeld.py`, `compile_dm_warp.py` and `get_model_metadata.py` were deleted. Model chunking was removed (#38941), and the eGPU model is now precompiled (#38930).
  - **`check_shell`** (#38877): a shellcheck-like lint over every tracked shell script.
  - `auth.login()` rewritten around a cancellable browser sign-in (#38893).
  - **DM model adds a sleep head** (#38942): a new `sleep` distraction type in the monitoring policy. Taken as-is.
  - tinygrad, msgq (reverted), opendbc, panda and rednose bumps.
- Adapt fork-owned code: port `tools/dgx` to the recurrent contract (a fork-owned `model_metadata.py` replaces the deleted helper). Fix a latent always-FP4 precision probe there. Rewrite the fork's auth tests. Update the fork's monitoring tests for `sleepProb` and add sleep-detection coverage. Make four fork scripts pass `check_shell`, fixing their stale pre-restructure paths along the way.

## Impact
- Affected specs: `development-workflow`, which gains an **Upstream Sync Triage** requirement (triage report before merging, facts/judgments/policy split, safety escalation, calibration after each sync).
- Affected code: fork-owned DGX tooling, auth tests, coverage/MISRA scripts, and the new triage tool. No upstream behavior was modified beyond the two lint-script conflict resolutions.
- Risk: Low to medium. The DGX TensorRT paths are verified on CPU only (contract, input building, parsing). A run on the DGX Spark is a follow-up.
