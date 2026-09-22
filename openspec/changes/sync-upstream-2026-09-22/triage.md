# Upstream sync triage

- fork: `develop`  upstream: `upstream/master`  merge-base: `0cf294d85`
- incoming non-merge commits: **86**; files git predicts will conflict: **2**
- fork overlay since merge-base: 24 modified upstream files, 429 fork-only files, 0 deleted
- TypeSafe: 453,358 input tokens (about $0.02). Final tool version; the pre-merge run differed only in the DM commit (routine → adapt)

| bucket | commits |
| --- | --- |
| resolve | 1 |
| adapt | 11 |
| read | 6 |
| routine | 68 |

## Safety review

Upstream changes judged to touch monitoring, actuation limits, or the safety model. Take them as upstream wrote them; never re-apply a fork change that weakens them.

- `5ae0da0e4` DM: Super Leicht Model (#38942) (p=0.86)

## Conflicted files

- `scripts/lint/check_shebang_format.sh` <- `64b4dbf11`
- `scripts/lint/lint.sh` <- `64b4dbf11`

## Conflicts: git predicts a conflict in a file these commits touch (1)

| prio | commit | kind | effort | why |
| --- | --- | --- | --- | --- |
| 10.3 | `64b4dbf11` add shellcheck-like static analysis (#38877) | feature | 1.6 | conflicts: `scripts/lint/check_shebang_format.sh`, `scripts/lint/lint.sh`; fork-edited: `scripts/lint/check_shebang_format.sh`, `scripts/lint/lint.sh`; invocation_change 0.93; uncertain: impact_ci_coverage, impact_packaging_lint |

## Adapt after merge: clean merge, but likely to break or silently drop a fork customization (11)

| prio | commit | kind | effort | why |
| --- | --- | --- | --- | --- |
| 8.9 | `6080cc616` Use a precompiled eGPU driving model (#38930) | ci_or_build | 2.0 | fork-edited: `.github/workflows/docs.yaml`, `.github/workflows/tests.yaml`, `.gitignore`; breaks fork refs 0.50: `big_driving_supercombo`, `chunk_file`, `recurrent`, `spatial`; nvidia_dgx 0.65; invocation_change 0.96; interface_change 0.92; uncertain: impact_ci_coverage, impact_nvidia_dgx, impact_research_tools, breaks_fork_code |
| 8.0 | `1328ace06` Remove model chunking and use LFS for Chestnut releases (#38941) | ci_or_build | 1.8 | fork-edited: `SConstruct`; invocation_change 0.95; interface_change 0.96; uncertain: impact_ci_coverage, breaks_fork_code |
| 7.8 | `b751b04cd` Use tinygrad ONNX and warp compilers for driving and DM (#38922) | ci_or_build | 2.3 | breaks fork refs 0.71: `Compiling`, `DEFAULT`, `OnnxRunner`, `SEED`; nvidia_dgx 0.65; ci_coverage 0.58; research_tools 0.50; packaging_lint 0.50; invocation_change 0.94; interface_change 0.96; uncertain: impact_ci_coverage, impact_nvidia_dgx, impact_shadow_algorithms, impact_research_tools, impact_packaging_lint, safety_relevant |
| 7.3 | `64f9b47b6` modeld: move history queues into ONNX (#38916) | refactor_or_rename | 2.3 | breaks fork refs 0.60: `features_buffer`, `hidden_state`, `spatial`, `unsqueeze`; nvidia_dgx 0.78; research_tools 0.58; invocation_change 0.54; interface_change 0.95; uncertain: impact_research_tools, breaks_fork_code, effort |
| 5.6 | `fa9f56ed7` Compile warp and onnx separately  (#38864) | refactor_or_rename | 1.9 | nvidia_dgx 0.53; invocation_change 0.77; interface_change 0.95; uncertain: impact_nvidia_dgx, impact_research_tools, effort |
| 4.6 | `7b469af43` ci: run docs builds on Namespace (#38921) | ci_or_build | 1.2 | fork-edited: `.github/workflows/docs.yaml`; invocation_change 0.77; uncertain: effort |
| 4.5 | `2c88d1ed6` Revert "bump msgq (#38836)" (#38920) | revert | 1.0 | fork-edited: `pyproject.toml`, `uv.lock`; uncertain: impact_packaging_lint |
| 4.4 | `81ae1a2e2` Use tinygrad generic ONNX compiler artifacts (#38926) | model_update | 1.3 | nvidia_dgx 0.69; invocation_change 0.77; interface_change 0.85; uncertain: impact_research_tools, effort |
| 4.2 | `19f0f69d8` [bot] Update Python packages (#38698) | dependency_bump | 0.7 | fork-edited: `uv.lock` |
| 4.2 | `5ae0da0e4` DM: Super Leicht Model (#38942) | model_update | 0.7 | safety change imported by fork: `openpilot/selfdrive/monitoring/test_monitoring_policy.py` |
| 2.9 | `39146cb24` test_runner: collect parameterized_class tests instead of skipping them with their base (#38820) | test_infra | 0.8 | harness_change 0.96 |

## Read yourself: the judgments were uncertain (6)

| prio | commit | kind | effort | why |
| --- | --- | --- | --- | --- |
| 3.8 | `ebb202f29` tools: share cancellable browser sign-in (#38893) | feature | 1.2 | interface_change 0.62; uncertain: breaks_fork_code, effort |
| 3.1 | `7db773559` rm chestnut power test (#38943) | revert | 0.5 | invocation_change 0.89; interface_change 0.82; uncertain: harness_change |
| 2.9 | `3cd1b6e48` Update tinygrad and use retargetable model artifacts (#38933) | dependency_bump | 1.3 | uncertain: impact_nvidia_dgx, impact_research_tools, impact_packaging_lint, effort |
| 2.6 | `d38264c42` log chestnut telemetry in hardwared (#38866) | feature | 0.4 | interface_change 0.95; uncertain: impact_research_tools |
| 2.3 | `e691acbf3` chestnut stresstest (#38858) | ci_or_build | 0.3 | invocation_change 0.95; uncertain: harness_change |
| 1.6 | `7f6f13997` replay: fix ranges with omitted start (#38963) | bugfix | 0.4 | uncertain: harness_change |

## Routine: no fork touchpoints and no strong signals (68)

| prio | commit | kind | effort | why |
| --- | --- | --- | --- | --- |
| 2.2 | `d06711be1` ui: prime menu (#38860) | ui | 0.2 | interface_change 0.96 |
| 1.8 | `cd1490af4` modeld: 2x faster chestnut build (#38656) | ci_or_build | 0.5 | invocation_change 0.83 |
| 1.7 | `7635dec61` ui: qr widget (#38951) | ui | 0.2 | interface_change 0.95 |
| 1.7 | `df7e0e5e7` Use upstream tinygrad disk tensors for model loading (#38956) | dependency_bump | 0.7 | - |
| 1.4 | `1b1b60ba9` cabana: generate dbc files during builds (#38974) | ci_or_build | 0.3 | invocation_change 0.81 |
| 1.3 | `dd226ac79` update chestnut fw in CI (#38876) | ci_or_build | 0.2 | invocation_change 0.81 |
| 1.2 | `298e51010` print jenkins crash log (#38880) | ci_or_build | 0.1 | invocation_change 0.90 |
| 1.2 | `4d9d1bc34` ui: not paired bookmark alert (#38936) | ui | 0.1 | - |
| 1.2 | `f1ba169fb` remove old profiler setups | docs_or_chore | 0.0 | interface_change 0.66 |
| 1.1 | `f23578404` Revert tinygrad update (#38879) | revert | 0.4 | - |
| 1.1 | `4c0799eb8` cabana: make Charts a native dockable panel (#38883) | ui | 0.1 | interface_change 0.81 |
| 1.0 | `f147094cb` cabana: restore Qt palettes and contrast (#38875) | ui | 0.0 | interface_change 0.84 |
| 1.0 | `5cad99bf7` fix(tools): allow spaces in workspace path for op.sh (#38839) | bugfix | 0.5 | - |
| 1.0 | `8abd13363` offroad alerts: pairing notification (#38948) | feature | 0.1 | interface_change 0.65 |
| 1.0 | `ffef0e6d3` ui: custom alert icons (#38917) | ui | 0.0 | interface_change 0.78 |
| 0.9 | `1aa97338c` Cabana: standardize floating dropdown menus (#38874) | ui | 0.0 | interface_change 0.62 |
| 0.8 | `c6d13eb72` Cinque v3 (#38932) | model_update | 0.2 | - |
| 0.8 | `f992a60bb` op setup: 2x faster on Linux (warm venv) (#38945) | ci_or_build | 0.1 | invocation_change 0.52 |
| 0.8 | `44914a7cf` cabana: remove custom focus-loss handling (#38884) | bugfix | 0.0 | interface_change 0.60 |
| 0.7 | `2eee697f7` cabana: improve theme contrast (#38965) | ui | 0.0 | - |
| 0.7 | `7dcb52405` ui: abstract info stack (#38949) | refactor_or_rename | 0.0 | - |
| 0.7 | `94f71fce0` cabana: add browser sign-in for remote routes (#38894) | feature | 0.0 | - |
| 0.6 | `f15544b48` rm pyserial from updater bundle (#38925) | dependency_bump | 0.1 | - |
| 0.6 | `9c054e285` replay: retry failed segment loads (#38897) | bugfix | 0.1 | - |
| 0.6 | `1090e2824` ui: add pairing provider to prime state (#38950) | feature | 0.1 | - |
| 0.6 | `95ed0213e` cabana: refine signal heatmap grid (#38964) | ui | 0.0 | - |
| 0.6 | `e77f6e56f` cabana: show cumulative byte and bit heatmaps (#38954) | feature | 0.1 | - |
| 0.5 | `6c69ebed7` cabana: improve heatmap readability (#38955) | ui | 0.0 | - |
| 0.5 | `c5cf29cf5` cabana: contain tooltips and capture plot drags (#38896) | ui | 0.0 | - |
| 0.5 | `064a51fe5` model replay: use a mici segment (#38878) | test_infra | 0.1 | - |
| 0.5 | `d3106c2b4` cabana: unify video and live stream titles (#38908) | ui | 0.0 | - |
| 0.4 | `dc338c283` cabana: standardize button spacing (#38873) | ui | 0.0 | - |
| 0.4 | `e5a6a8b73` cabana: make the Signals pane a native dock panel, fix pad (#38889) | ui | 0.0 | - |
| 0.4 | `c8d56a199` cabana: filter multiplexed signals in binary grid (#38960) | bugfix | 0.0 | - |
| 0.4 | `268991775` locationd: fix critical_services spelling (#38912) | bugfix | 0.0 | - |
| 0.3 | `ad5c0b946` tools: optimize route parsing (#38714) | refactor_or_rename | 0.0 | - |
| 0.3 | `a39561058` ui: allow delay of scroller start (#38958) | ui | 0.0 | - |
| 0.3 | `45a7747ca` cabana: fix detached panel menus (#38890) | bugfix | 0.0 | - |
| 0.3 | `cab53438a` AGNOS 19.8 (#38935) | ci_or_build | 0.1 | - |
| 0.3 | `b55e37b9c` cabana: right-align numeric message values (#38959) | ui | 0.0 | - |
| 0.3 | `9a95fdc1a` cabana: fix stale message size warnings (#38961) | bugfix | 0.0 | - |
| 0.3 | `e6aa405e0` ui: fix touch validity on alerts menu (#38915) | bugfix | 0.0 | - |
| 0.3 | `5c365d339` cabana: remove inner video container (#38901) | ui | 0.0 | - |
| 0.3 | `5f50bda7b` cabana: improve signal panel layout (#38910) | ui | 0.0 | - |
| 0.3 | `2ae3598b5` cabana: prevent signal panel collapse (#38903) | ui | 0.0 | - |
| 0.2 | `bca3b0472` cabana: improve dark theme contrast (#38957) | ui | 0.0 | - |
| 0.2 | `3b2a75a4e` cabana: resize heatmap and prevent signal clipping (#38975) | ui | 0.0 | - |
| 0.2 | `4e60afe70` cabana: show persistent fps counter (#38898) | ui | 0.0 | - |
| 0.2 | `164a3a1e3` cabana: stop chart tooltips over menus (#38905) | bugfix | 0.0 | - |
| 0.2 | `680aa6f86` cabana: revert quiet signal brightness clamp (#38970) | revert | 0.0 | - |
| 0.2 | `7e84a6166` cabana: use ImGui decorations for floating panes (#38871) | ui | 0.0 | - |
| 0.2 | `20064d968` cabana: show placeholder for unresolved timestamps (#38870) | ui | 0.0 | - |
| 0.2 | `c715ef476` cabana: flatten signal panel containers (#38904) | ui | 0.0 | - |
| 0.2 | `521db4c82` locationd: reduce initial gyro bias uncertainty (#38981) | bugfix | 0.0 | - |
| 0.2 | `11093e743` cabana: improve dark slider track contrast (#38899) | ui | 0.0 | - |
| 0.2 | `60b3ccf39` cabana: accept prefixed hex values in id filters (#38952) | feature | 0.0 | - |
| 0.2 | `c504b92ab` cabana: clarify signal button states (#38953) | ui | 0.0 | - |
| 0.2 | `b6918cb31` cabana: fix download bar layout (#38900) | ui | 0.0 | - |
| 0.2 | `e6f9b3c6c` cabana: preserve menu bar border on highlight (#38902) | bugfix | 0.0 | - |
| 0.2 | `5b34f151d` cabana: balance signal action spacing (#38911) | ui | 0.0 | - |
| 0.2 | `ea0afb67a` cabana: preserve activity with a brighter baseline (#38971) | ui | 0.0 | - |
| 0.2 | `0d4a4ab91` cabana: disable collapse for dockable panels (#38885) | ui | 0.0 | - |
| 0.2 | `885792de3` cabana: remove message table outer border (#38907) | ui | 0.0 | - |
| 0.2 | `64e5d1e5e` cabana: preserve selected menu color on hover (#38909) | ui | 0.0 | - |
| 0.1 | `5319474ee` cabana: disable tool window collapse (#38906) | ui | 0.0 | - |
| 0.1 | `bd176cb6a` ui: remove question marks (#38938) | ui | 0.0 | - |
| 0.1 | `d375e7ff5` cabana: brighten quiet signals in dark mode (#38969) | ui | 0.0 | - |
| 0.1 | `d643860af` ui: increase info subtext size (#38946) | ui | 0.0 | - |
