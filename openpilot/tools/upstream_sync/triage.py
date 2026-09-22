#!/usr/bin/env python3
"""
Upstream sync triage: rank incoming commaai/openpilot commits by how much fork work they need.

git supplies the exact facts (incoming commits, predicted merge conflicts, overlap with files
the fork modifies). TypeSafe answers the semantic questions per commit (what kind of change,
does it alter test/CI mechanics or referenced interfaces, which fork customizations it hits,
how much follow-up it needs). This module combines both with explicit weights and writes a
worklist.

Usage:
  uv run --frozen --with typesafe-sdk python -m openpilot.tools.upstream_sync.triage   # develop vs upstream/master
  python -m openpilot.tools.upstream_sync.triage --facts-only        # git facts only, no API calls
  ... triage --upstream <sha> --out /tmp/triage

Requires TYPESAFE_API_KEY; `uv run --with` keeps typesafe-sdk out of .venv (and the dependency budget).
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openpilot.tools.upstream_sync.git_facts import Commit, SyncFacts, collect
from openpilot.tools.upstream_sync.judgments import FORK_PROFILE, JudgmentCache, judge_commits

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = Path.home() / ".cache" / "openpilot_upstream_sync"

# Policy lives here, not in the questions: change weights or thresholds without re-asking.
WEIGHTS = {
  "conflict": 4.0,  # git says the merge conflicts in a file this commit touches
  "fork_modified": 2.0,  # touches an upstream file the fork edits (clean merge can drop the fork's edit)
  "fork_deleted": 1.0,
  "breaks_fork_code": 3.0,  # multiplies P(fork files break) when git found fork references to removed names
  "safety_import": 3.0,  # multiplies P(safety-relevant) when fork code imports a module the commit changes
  "effort": 3.0,  # multiplies the 0-1 normalized effort score
  "fork_impact": 2.0,  # multiplies the strongest per-customization impact probability
  "harness_change": 1.5,  # the fork keeps pytest while upstream owns a unittest harness
  "invocation_change": 1.0,
  "interface_change": 1.0,
}
YES = 0.5
UNCERTAIN = (0.35, 0.65)
LOW_CONFIDENCE = 0.4
SIGNALS = ("harness_change", "invocation_change", "interface_change")
# Signals that move a commit into "adapt" on their own. interface/invocation changes are
# upstream-wide; they only count for the fork through fork_references or a fork impact.
DECISIVE = ("harness_change",)
ADAPT_EFFORT = 2.0


@dataclass
class Triaged:
  commit: Commit
  priority: float
  bucket: str
  kind: str
  effort: float | None
  safety: float | None
  behavior: float | None
  breaks: float | None = None
  references: dict[str, list[str]] = field(default_factory=dict)
  impacts: dict[str, float] = field(default_factory=dict)
  signals: dict[str, float] = field(default_factory=dict)
  conflicts: list[str] = field(default_factory=list)
  fork_modified: list[str] = field(default_factory=list)
  uncertain: list[str] = field(default_factory=list)


def triage_commit(c: Commit, facts: SyncFacts, answers: dict[str, Any] | None) -> Triaged:
  paths = set(c.paths)
  conflicts = sorted(paths & set(facts.conflicts))
  fork_modified = sorted(paths & facts.overlay.modified)
  fork_deleted = paths & facts.overlay.deleted

  priority = WEIGHTS["conflict"] * bool(conflicts) + WEIGHTS["fork_modified"] * bool(fork_modified) \
    + WEIGHTS["fork_deleted"] * bool(fork_deleted)

  t = Triaged(commit=c, priority=0.0, bucket="routine", kind="?", effort=None, safety=None, behavior=None,
              conflicts=conflicts, fork_modified=fork_modified, references=c.fork_references)
  if answers is not None:
    a = answers
    t.kind = a["kind"]["choice"]
    t.effort = a["effort"]["score"]
    t.safety = a["safety_relevant"]["noul"]
    t.behavior = a["behavior_change"]["noul"]
    t.impacts = {k: a[f"impact_{k}"]["noul"] for k in FORK_PROFILE}
    t.signals = {k: a[k]["noul"] for k in SIGNALS}
    if t.references:
      t.breaks = a["breaks_fork_code"]["noul"]

    priority += WEIGHTS["effort"] * t.effort / 3 + WEIGHTS["fork_impact"] * max(t.impacts.values())
    priority += sum(WEIGHTS[k] * p for k, p in t.signals.items()) + WEIGHTS["breaks_fork_code"] * (t.breaks or 0)
    if c.fork_importers:
      priority += WEIGHTS["safety_import"] * t.safety

    # only the answers that can move a commit between buckets are worth a person's second look
    probs = {**{k: t.signals[k] for k in DECISIVE}, **{f"impact_{k}": p for k, p in t.impacts.items()},
             "safety_relevant": t.safety}
    if t.breaks is not None:
      probs["breaks_fork_code"] = t.breaks
    t.uncertain = [k for k, p in probs.items() if UNCERTAIN[0] <= p <= UNCERTAIN[1]]
    if a["effort"]["confidence"] < LOW_CONFIDENCE and t.effort >= 1:
      t.uncertain.append("effort")

  t.priority = round(priority, 2)
  if conflicts:
    t.bucket = "resolve"
  elif fork_modified or fork_deleted or (t.breaks or 0) >= YES or max(t.impacts.values(), default=0) >= YES \
      or any(t.signals.get(k, 0) >= YES for k in DECISIVE) or (t.effort or 0) >= ADAPT_EFFORT \
      or ((t.safety or 0) >= YES and c.fork_importers):
    # a safety change in a module fork code imports: the fork's own checks of it must be re-run
    t.bucket = "adapt"
  elif t.uncertain:
    t.bucket = "read"
  return t


BUCKETS = {
  "resolve": "Conflicts: git predicts a conflict in a file these commits touch",
  "adapt": "Adapt after merge: clean merge, but likely to break or silently drop a fork customization",
  "read": "Read yourself: the judgments were uncertain",
  "routine": "Routine: no fork touchpoints and no strong signals",
}


def render_markdown(facts: SyncFacts, rows: list[Triaged], usage: dict[str, int]) -> str:
  lines = [
    "# Upstream sync triage",
    "",
    f"- fork: `{facts.fork_ref}`  upstream: `{facts.upstream_ref}`  merge-base: `{facts.merge_base[:9]}`",
    f"- incoming non-merge commits: **{len(rows)}**; files git predicts will conflict: **{len(facts.conflicts)}**",
    f"- fork overlay since merge-base: {len(facts.overlay.modified)} modified upstream files, " +
    f"{len(facts.overlay.added)} fork-only files, {len(facts.overlay.deleted)} deleted",
  ]
  if usage:
    lines.append(f"- TypeSafe input tokens this run: {usage.get('input_tokens', 0):,} (cached commits cost nothing)")
  lines += ["", "| bucket | commits |", "| --- | --- |"]
  lines += [f"| {b} | {sum(r.bucket == b for r in rows)} |" for b in BUCKETS]

  safety = sorted((r for r in rows if (r.safety or 0) >= YES), key=lambda r: -(r.safety or 0))
  if safety:
    lines += ["", "## Safety review", "", "Upstream changes judged to touch monitoring, actuation limits, or the safety model. " +
              "Take them as upstream wrote them; never re-apply a fork change that weakens them.", ""]
    lines += [f"- `{r.commit.sha[:9]}` {r.commit.subject} (p={r.safety:.2f})" for r in safety]

  if facts.conflicts:
    lines += ["", "## Conflicted files", ""]
    for path in facts.conflicts:
      touching = [r for r in rows if path in r.commit.paths]
      shas = ", ".join(f"`{r.commit.sha[:9]}`" for r in touching[-5:])
      more = f" (+{len(touching) - 5} earlier)" if len(touching) > 5 else ""
      lines.append(f"- `{path}` <- {shas}{more}" if touching else f"- `{path}` (no commit in this run touches it)")

  for bucket, title in BUCKETS.items():
    group = sorted((r for r in rows if r.bucket == bucket), key=lambda r: -r.priority)
    if not group:
      continue
    lines += ["", f"## {title} ({len(group)})", "", "| prio | commit | kind | effort | why |", "| --- | --- | --- | --- | --- |"]
    for r in group:
      why = []
      if r.conflicts:
        why.append("conflicts: " + ", ".join(f"`{p}`" for p in r.conflicts[:3]))
      if r.fork_modified:
        why.append("fork-edited: " + ", ".join(f"`{p}`" for p in r.fork_modified[:3]))
      if (r.safety or 0) >= YES and r.commit.fork_importers:
        why.append("safety change imported by fork: " + ", ".join(f"`{p}`" for p in r.commit.fork_importers[:3]))
      if r.references and (r.breaks or 0) >= YES:
        names = ", ".join(f"`{n}`" for n in list(r.references)[:4])
        why.append(f"breaks fork refs {r.breaks:.2f}: {names}")
      why += [f"{k} {p:.2f}" for k, p in sorted(r.impacts.items(), key=lambda kv: -kv[1]) if p >= YES]
      why += [f"{k} {p:.2f}" for k, p in r.signals.items() if p >= YES]
      if r.uncertain:
        why.append("uncertain: " + ", ".join(r.uncertain))
      effort = "-" if r.effort is None else f"{r.effort:.1f}"
      subject = r.commit.subject.replace("|", "\\|")
      lines.append(f"| {r.priority:.1f} | `{r.commit.sha[:9]}` {subject} | {r.kind} | {effort} | {'; '.join(why) or '-'} |")

  return "\n".join(lines) + "\n"


def to_json(facts: SyncFacts, rows: list[Triaged]) -> dict[str, Any]:
  return {
    "fork_ref": facts.fork_ref,
    "upstream_ref": facts.upstream_ref,
    "merge_base": facts.merge_base,
    "conflicts": facts.conflicts,
    "commits": [{
      "sha": r.commit.sha, "subject": r.commit.subject, "date": r.commit.date, "bucket": r.bucket,
      "priority": r.priority, "kind": r.kind, "effort": r.effort, "safety": r.safety, "behavior": r.behavior,
      "impacts": r.impacts, "signals": r.signals, "breaks": r.breaks, "references": r.references,
      "fork_importers": r.commit.fork_importers, "conflicts": r.conflicts, "fork_modified": r.fork_modified, "uncertain": r.uncertain,
    } for r in rows],
  }


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--repo", type=Path, default=REPO)
  p.add_argument("--fork", default="develop", help="fork branch that will receive the merge")
  p.add_argument("--upstream", default="upstream/master", help="upstream ref to merge")
  p.add_argument("--out", type=Path, default=Path("upstream_sync_triage"), help="output directory")
  p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
  p.add_argument("--concurrency", type=int, default=8)
  p.add_argument("--limit", type=int, help="only judge the N most recent incoming commits")
  p.add_argument("--facts-only", action="store_true", help="skip TypeSafe; rank on git facts alone")
  args = p.parse_args(argv)

  facts = collect(args.repo, args.fork, args.upstream)
  if args.limit:
    facts.commits = facts.commits[-args.limit:]
  print(f"{len(facts.commits)} incoming commits, {len(facts.conflicts)} predicted conflicted files", file=sys.stderr)

  judged: dict[str, dict[str, Any]] = {}
  usage = {"input_tokens": 0}
  if not args.facts_only:
    if not os.environ.get("TYPESAFE_API_KEY"):
      print("TYPESAFE_API_KEY is not set (or use --facts-only)", file=sys.stderr)
      return 2
    cache = JudgmentCache(args.cache)
    done = 0

    def progress(c: Commit) -> None:
      nonlocal done
      done += 1
      print(f"  judged {done}: {c.sha[:9]} {c.subject[:70]}", file=sys.stderr)

    before = {c.sha for c in facts.commits if cache.get(c.sha) is None}
    judged = asyncio.run(judge_commits(args.repo, facts, cache, args.concurrency, progress))
    usage["input_tokens"] = sum(judged[s]["usage"]["input_tokens"] for s in before)

  rows = [triage_commit(c, facts, judged.get(c.sha, {}).get("answers")) for c in facts.commits]
  args.out.mkdir(parents=True, exist_ok=True)
  (args.out / "triage.md").write_text(render_markdown(facts, rows, usage))
  (args.out / "triage.json").write_text(json.dumps(to_json(facts, rows), indent=1))
  print(f"wrote {args.out / 'triage.md'} and triage.json", file=sys.stderr)
  return 0


if __name__ == "__main__":
  sys.exit(main())
