# Upstream sync triage

Ranks the commaai/openpilot commits waiting to be merged into this fork by how much
fork-side work each one is likely to need, so a sync starts from a worklist instead of
a 400-commit log.

```bash
export TYPESAFE_API_KEY=...        # https://console.typesafe.ai/
git fetch upstream
uv run --frozen --with typesafe-sdk python -m openpilot.tools.upstream_sync.triage --out upstream_sync_triage
python -m openpilot.tools.upstream_sync.triage --facts-only   # git facts only, no SDK or API calls
```

`uv run --with` adds the SDK in a temporary overlay. Installing it into `.venv`
instead (`requirements.txt`) works, but pushes `scripts/lint/check_dependencies.py`
over its package budget.

Output: `triage.md` (the worklist) and `triage.json` (every signal, for scripting).

## How it decides

**git decides the facts** (`git_facts.py`): the incoming non-merge commits
(`develop..upstream/master`), the files `git merge-tree` predicts will conflict,
which upstream files the fork modifies, adds, or deletes since the merge-base, and
**fork references**. A fork reference is a name that a commit removes, which the
final upstream tree no longer has, or a Python function whose signature it changes,
paired with the fork-owned files that still use it (for example `CarSpecificEvents`
mapped to the fork's tests). git also records **fork importers**: fork files that import
a Python module the commit changes. These catch additions, such as a new message field
the fork's mocks must now provide.

**TypeSafe answers the semantic questions** (`judgments.py`): one request per commit.
The state is the commit message, file list, a diff excerpt, the exact fork touchpoints,
and `FORK_PROFILE`, a description of each fork customization. The questions are:

| id | primitive | asks |
| --- | --- | --- |
| `kind` | Choice | feature, bugfix, car support, model update, refactor, test infra, CI/build, deps, UI, docs, revert |
| `harness_change` | Noul | changes how tests are written or run (the fork keeps pytest) |
| `invocation_change` | Noul | changes how CI or scripts invoke tools, the case where a clean merge silently drops a fork override |
| `interface_change` | Noul | renames, moves or removes something other code references |
| `behavior_change` | Noul | would change a replay of the same drive |
| `safety_relevant` | Noul | touches monitoring, actuation limits, or the safety model |
| `breaks_fork_code` | Noul | whether the fork references git found really break, as opposed to a coincidental name match |
| `impact_<customization>` | Noul, one per `FORK_PROFILE` entry | breaks or requires changing that customization |
| `effort` | Score (0–3) | nothing / trivial / moderate / significant fork follow-up |

**Code decides the policy** (`triage.py`): `WEIGHTS` combines the facts and the
judgments into a priority, and the thresholds place each commit in one bucket:

- **resolve**: git predicts a conflict in a file the commit touches
- **adapt**: merges cleanly, but touches a fork-edited file, breaks fork references,
  is a safety-relevant change to a module that fork code imports (the fork's own checks
  of it must be re-run), or has a strong impact, harness or effort signal. Interface and
  invocation changes alone only raise priority, because most upstream renames never
  reach fork code.
- **read**: every signal is weak, but at least one probability is near 0.5 or the
  effort distribution is spread out, so a person should look
- **routine**: everything else

Commits with `safety_relevant >= 0.5` are also listed in a separate section. They are
always taken as upstream wrote them.

Answers are cached per commit sha under `~/.cache/openpilot_upstream_sync/`, keyed by a
hash of the questions and the model. Re-runs after `git fetch` only pay for new commits.
Changing `WEIGHTS` or thresholds never needs a re-run; editing `FORK_PROFILE` or a
question invalidates the cache.

## Calibration

On the September 2026 sync (394 commits, 25 of them known to have caused fork work),
22 of the 25 landed in resolve or adapt, and 96 commits were flagged in total (24%).
git facts alone caught 18 of the 25 from 71 flagged commits. The misses were subtle API
changes: a changed return value, and a generic method name (`update`) that lost a
parameter. When the report lists a generic name under fork references, confirm its
callers with the codebase-memory graph (`trace_path`) before trusting a low
`breaks_fork_code`.

A full run over about 400 commits takes about 2 minutes and costs about $0.08.

On the 2026-09-22 sync (86 commits), the report predicted both conflicts. Every fork
break was flagged: the DGX tooling and lint scripts in adapt, and the auth tests in
read. The one gap was the DM model's new `sleepProb` field, which broke the fork's
monitoring mocks without removing anything. That gap led to the fork-importers rule.

The judgments only rank what to read. They are not proof that a merge is correct. The
sync is verified the usual way: lint, pytest, process replay, and CI.
