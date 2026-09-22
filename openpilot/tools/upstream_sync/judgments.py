"""Semantic judgments about incoming upstream commits, answered by TypeSafe (Jev).

Each commit gets one System One request: its message, file list, a diff excerpt, the
exact fork touchpoints from git_facts, and a profile of what the fork customizes. The
questions run in parallel over that state. Code in triage.py turns the typed answers
into a priority; nothing here decides what to do with a commit.

Update FORK_PROFILE when the fork gains or drops a customization -- it is the only place
the model learns what "the fork" means.
"""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from openpilot.tools.upstream_sync.git_facts import Commit, SyncFacts, diff_excerpt

MODEL = "jev-latest"

# What this fork carries on top of commaai/openpilot. Keys are stable ids used in code.
FORK_PROFILE: dict[str, str] = {
  "pytest_harness": (
    "The fork runs every test with pytest (fork-owned root conftest.py with an autouse OpenpilotPrefix fixture, " +
    "pytest-xdist, pytest-cov, the COMMA_HARDWARE_TEST class flag, @pytest.mark.slow). Upstream switched to a " +
    "unittest-based OpenpilotTestCase harness run by tools/op.sh test (tools/test_runner.py); the fork must keep " +
    "upstream's test classes runnable under pytest."
  ),
  "ci_coverage": (
    "Fork-owned GitHub Actions configuration: tests.yaml runs pytest instead of upstream's runner, plus coverage " +
    "jobs (pytest-cov with --cov=openpilot, C++ coverage via scripts/cpp-coverage.sh, ratchet scripts), " +
    "SonarCloud and codecov uploads, docs.yaml, and a MISRA baseline. These jobs depend on how upstream invokes " +
    "builds, test targets, PYTHONPATH and environment setup; when upstream changes an invocation, the fork's " +
    "override can be silently dropped by a clean merge."
  ),
  "nvidia_dgx": (
    "NVIDIA GPU support: openpilot/common/hardware/nvidia (GPU detection, precision selection) exported from " +
    "openpilot/common/hardware/__init__.py, and openpilot/tools/dgx (TensorRT benchmarks, DoRA fine-tuning). " +
    "It depends on the modeld model contract: driving_supercombo.onnx inputs, outputs and the model runner."
  ),
  "shadow_algorithms": (
    "Shadow-mode and algorithm evaluation: openpilot/common/hardware/shadow_mode.py, openpilot/tools/shadow, " +
    "the algorithm harness and trackers under openpilot/selfdrive/controls/lib, algo_bench.py, and a small hook " +
    "in openpilot/selfdrive/car/card.py. It consumes controls, planner and car interfaces and cereal log messages."
  ),
  "research_tools": (
    "openpilot/tools/fair (model distillation, quantization, student networks) and openpilot/tools/stonesoup " +
    "(Stone Soup tracking integration). They read modeld outputs, cereal messages and openpilot/tools/lib log " +
    "reader APIs."
  ),
  "verification": (
    "Formal verification and fuzzing under verification/ (CBMC, TLA+, SPIN, fuzz harnesses) that model the panda " +
    "safety code in opendbc and the selfdrived state machine."
  ),
  "packaging_lint": (
    "Packaging and lint configuration: pyproject.toml's 'submodules' extra installs opendbc, msgq, rednose, " +
    "tinygrad and teleoprtc as editable packages from their *_repo checkouts (no top-level symlinks), uv.lock, " +
    "fork-scoped ruff and ty overrides, and scripts/lint (lint.sh, check_dependencies.py)."
  ),
}

KINDS: dict[str, str] = {
  "feature": "Adds a new capability or user-facing behavior",
  "bugfix": "Fixes incorrect behavior without adding a capability",
  "car_support": "Car ports, fingerprints, tuning, or an opendbc/panda bump for vehicles",
  "model_update": "New or retrained driving/monitoring model, or modeld changes that ship one",
  "refactor_or_rename": "Restructures, moves, or renames code without intended behavior change",
  "test_infra": "Changes to tests, test harnesses, or test tooling only",
  "ci_or_build": "CI workflows, build system (SCons), release or setup scripts",
  "dependency_bump": "Updates third-party dependencies, lockfiles, or submodule pins",
  "ui": "User interface visuals or interaction",
  "docs_or_chore": "Documentation, comments, formatting, or trivial housekeeping",
  "revert": "Reverts an earlier commit",
}

EFFORT_LEVELS = [
  "Nothing to do on the fork side: the change merges as-is and no fork customization depends on what it touches",
  "Trivial follow-up: update an import path, a renamed identifier, or a file path in a fork-owned file",
  "Moderate follow-up: adapt fork tests, CI jobs, or configuration to a changed upstream mechanism",
  "Significant rework: a fork customization must be redesigned against new upstream structure or behavior",
]

SAFETY_NOTE = (
  "Driver monitoring (openpilot/selfdrive/monitoring), actuation limits (openpilot/selfdrive/selfdrived/helpers.py) " +
  "and the panda safety model (opendbc safety code) must never be weakened."
)


def build_questions() -> dict[str, dict[str, Any]]:
  q: dict[str, dict[str, Any]] = {
    "kind": {
      "type": "choice",
      "instructions": "Which kind of change is the upstream commit in `commit`, judging from its message, files and `diff`?",
      "criteria": KINDS,
    },
    "harness_change": {
      "type": "noul",
      "instructions": "Does `commit` change how tests are written, discovered, or run (test runner, base test classes, " +
                      "conftest, markers, fixtures, or pytest versus unittest usage), as opposed to only adding or editing " +
                      "test cases?",
    },
    "invocation_change": {
      "type": "noul",
      "instructions": "Does `commit` change how CI workflows or build, lint, or test scripts invoke tools: commands, " +
                      "arguments, targets, job structure, environment variables such as PYTHONPATH, or setup steps?",
    },
    "interface_change": {
      "type": "noul",
      "instructions": "Does `commit` rename, move, or remove a module, function, class, command-line entry point, cereal " +
                      "message or field, service, or Params key that other code or tools could reference?",
    },
    "behavior_change": {
      "type": "noul",
      "instructions": "Does `commit` change on-road driving behavior: control or planner outputs, model outputs, alerts, " +
                      "or engagement logic, such that a replay of the same drive would produce different results?",
    },
    "safety_relevant": {
      "type": "noul",
      "instructions": {
        "policy": SAFETY_NOTE,
        "question": "Does `commit` modify code covered by `policy`, or change limits, checks, or timing it relies on?",
      },
    },
    "breaks_fork_code": {
      "type": "noul",
      "instructions": "`fork_touchpoints.fork_references` maps names that `commit` removes, or whose Python signature it " +
                      "changes, to fork-owned files that still use them. Will those fork files stop working (import errors, " +
                      "missing attributes, wrong arguments, stale message or field names) unless the fork updates them?",
      "criteria": {
        "true": "At least one listed fork file uses a listed name in the way this commit breaks",
        "false": "The list is empty, or the matches are coincidental (same word, different thing) or still compatible",
      },
    },
    "effort": {
      "type": "score",
      "instructions": "Given `fork_profile` and the exact overlap in `fork_touchpoints`, how much fork-side work will merging " +
                      "`commit` require?",
      "criteria": EFFORT_LEVELS,
    },
  }
  # One Noul per fork customization: several can apply to the same commit.
  for key, description in FORK_PROFILE.items():
    q[f"impact_{key}"] = {
      "type": "noul",
      "instructions": {
        "fork_customization": description,
        "question": "Will merging `commit` break `fork_customization` or require the fork to change it, for example by " +
                    "renaming, moving, or removing code, files, messages, commands, or behavior it relies on?",
      },
      "criteria": {
        "true": "The customization depends on something this commit changes",
        "false": "The customization does not depend on anything this commit changes",
      },
    }
  return q


QUESTIONS = build_questions()
# bump when commit_state changes shape, so cached answers to the old state are not reused
STATE_VERSION = 2
MAX_STATE_REFERENCES = 25
QUESTIONS_HASH = hashlib.sha256(json.dumps([MODEL, STATE_VERSION, QUESTIONS], sort_keys=True).encode()).hexdigest()[:12]


def commit_state(repo: Path, facts: SyncFacts, commit: Commit) -> dict[str, Any]:
  paths = set(commit.paths)
  return {
    "commit": {
      "subject": commit.subject,
      "message": commit.body[:4000],
      "files": [f"{f.status} {f.path}" + (" (submodule pin)" if f.submodule else "") for f in commit.files][:300],
    },
    "fork_touchpoints": {
      "files_that_conflict_in_the_full_merge": sorted(paths & set(facts.conflicts)),
      "upstream_files_the_fork_modifies": sorted(paths & facts.overlay.modified),
      "files_the_fork_deleted": sorted(paths & facts.overlay.deleted),
      "fork_references": dict(list(commit.fork_references.items())[:MAX_STATE_REFERENCES]),
    },
    "fork_profile": FORK_PROFILE,
    "diff": diff_excerpt(repo, commit.sha),
  }


class JudgmentCache:
  """Answers per commit sha, invalidated when the questions or model change."""

  def __init__(self, directory: Path):
    self.dir = directory / QUESTIONS_HASH
    self.dir.mkdir(parents=True, exist_ok=True)

  def get(self, sha: str) -> dict[str, Any] | None:
    p = self.dir / f"{sha}.json"
    return json.loads(p.read_text()) if p.exists() else None

  def put(self, sha: str, record: dict[str, Any]) -> None:
    (self.dir / f"{sha}.json").write_text(json.dumps(record, indent=1))


async def judge_commits(repo: Path, facts: SyncFacts, cache: JudgmentCache, concurrency: int = 8,
                        progress=None) -> dict[str, dict[str, Any]]:
  """Return {sha: {"answers": {...}, "usage": {...}}}, reusing cached answers."""
  from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score

  kinds = {"noul": Noul, "choice": Choice, "score": Score}
  questions = {k: kinds[v["type"]](**{f: x for f, x in v.items() if f != "type"}) for k, v in QUESTIONS.items()}

  results: dict[str, dict[str, Any]] = {}
  todo = []
  for c in facts.commits:
    if (hit := cache.get(c.sha)) is not None:
      results[c.sha] = hit
    else:
      todo.append(c)

  sem = asyncio.Semaphore(concurrency)
  async with AsyncTypeSafeClient(model=MODEL) as client:
    async def one(c: Commit) -> None:
      async with sem:
        # building state shells out to git; keep it off the event loop and inside the gate
        state = await asyncio.to_thread(commit_state, repo, facts, c)
        resp = await client.system_one(state, questions)
      dumped = resp.model_dump()
      record = {"model": dumped["model"], "answers": dumped["answers"], "usage": dumped["usage"], "sha": c.sha}
      cache.put(c.sha, record)
      results[c.sha] = record
      if progress:
        progress(c)

    await asyncio.gather(*(one(c) for c in todo))
  return results
