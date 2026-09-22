import os
import subprocess
from pathlib import Path

import pytest

from openpilot.tools.upstream_sync.git_facts import collect
from openpilot.tools.upstream_sync.judgments import FORK_PROFILE, QUESTIONS, JudgmentCache
from openpilot.tools.upstream_sync.triage import main, render_markdown, triage_commit

GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def git(repo: Path, *args: str) -> str:
  return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                        env={**os.environ, **GIT_ENV}).stdout.strip()


def commit(repo: Path, msg: str, files: dict[str, str]) -> None:
  for path, text in files.items():
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text)
  git(repo, "add", "-A")
  git(repo, "commit", "-qm", msg)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
  """base -> fork edits shared.py and adds fork_tool.py using two upstream helpers;
  upstream edits shared.py (conflict), renames one helper, and changes the other's signature."""
  git(tmp_path, "init", "-q", "-b", "master")
  base_lib = "def helper_fn(a):\n  return a\n\ndef compute_thing(a):\n  return a\n"
  commit(tmp_path, "base", {"shared.py": "x = 1\n", "lib.py": base_lib})
  git(tmp_path, "branch", "develop")
  commit(tmp_path, "upstream: change shared", {"shared.py": "x = 2\n"})
  commit(tmp_path, "upstream: rename helper", {"lib.py": base_lib.replace("helper_fn", "helper_func")})
  commit(tmp_path, "upstream: new arg", {"lib.py": base_lib.replace("helper_fn", "helper_func").replace("thing(a)", "thing(a, b)")})
  git(tmp_path, "checkout", "-q", "develop")
  fork_tool = "from lib import helper_fn, compute_thing\nhelper_fn(compute_thing(1))\n"
  commit(tmp_path, "fork: customize", {"shared.py": "x = 3\n", "fork_tool.py": fork_tool})
  return tmp_path


def answers(**nouls: float) -> dict:
  a = {"kind": {"choice": "bugfix", "confidence": 0.9},
       "effort": {"score": nouls.pop("effort", 0.0), "confidence": nouls.pop("effort_conf", 0.9)}}
  for q in QUESTIONS:
    if QUESTIONS[q]["type"] == "noul":
      a[q] = {"noul": nouls.get(q, 0.02)}
  return a


class TestGitFacts:
  def test_collect(self, repo):
    facts = collect(repo, "develop", "master")
    assert [c.subject for c in facts.commits] == ["upstream: change shared", "upstream: rename helper", "upstream: new arg"]
    assert facts.conflicts == ["shared.py"]
    assert facts.overlay.modified == {"shared.py"}
    assert facts.overlay.added == {"fork_tool.py"}
    assert facts.commits[0].paths == ["shared.py"]

  def test_fork_references(self, repo):
    shared, rename, new_arg = collect(repo, "develop", "master").commits
    assert shared.fork_references == {}
    assert rename.fork_references == {"helper_fn": ["fork_tool.py"]}
    assert new_arg.fork_references == {"compute_thing": ["fork_tool.py"]}

  def test_no_conflicts_when_in_sync(self, repo):
    git(repo, "merge", "-q", "-X", "ours", "master", "-m", "sync")
    facts = collect(repo, "develop", "master")
    assert facts.commits == [] and facts.conflicts == []


class TestTriage:
  def test_buckets_from_git_facts_alone(self, repo):
    facts = collect(repo, "develop", "master")
    rows = [triage_commit(c, facts, None) for c in facts.commits]
    assert [r.bucket for r in rows] == ["resolve", "routine", "routine"]
    assert rows[0].priority > rows[1].priority

  def test_judgments_raise_clean_commits(self, repo):
    facts = collect(repo, "develop", "master")
    clean = facts.commits[1]
    assert triage_commit(clean, facts, answers()).bucket == "routine"
    assert triage_commit(clean, facts, answers(impact_ci_coverage=0.9)).bucket == "adapt"
    assert triage_commit(clean, facts, answers(harness_change=0.8)).bucket == "adapt"
    assert triage_commit(clean, facts, answers(effort=2.0)).bucket == "adapt"
    assert triage_commit(clean, facts, answers(breaks_fork_code=0.9)).bucket == "adapt"
    # upstream-wide signals alone do not make fork work
    assert triage_commit(clean, facts, answers(interface_change=0.99, invocation_change=0.99)).bucket == "routine"
    uncertain = triage_commit(clean, facts, answers(harness_change=0.45, effort=1.2, effort_conf=0.2))
    assert uncertain.bucket == "read" and uncertain.uncertain == ["harness_change", "effort"]

  def test_breaks_needs_references(self, repo):
    facts = collect(repo, "develop", "master")
    facts.commits[1].fork_references = {}
    assert triage_commit(facts.commits[1], facts, answers(breaks_fork_code=0.9)).bucket == "routine"

  def test_report(self, repo):
    facts = collect(repo, "develop", "master")
    rows = [triage_commit(c, facts, answers(safety_relevant=0.9) if i else None) for i, c in enumerate(facts.commits)]
    md = render_markdown(facts, rows, {})
    assert "## Safety review" in md and "`shared.py` <-" in md

  def test_cli_facts_only(self, repo, tmp_path):
    out = tmp_path / "out"
    assert main(["--repo", str(repo), "--fork", "develop", "--upstream", "master", "--out", str(out), "--facts-only"]) == 0
    assert (out / "triage.md").exists() and (out / "triage.json").exists()


class TestJudgments:
  def test_questions_cover_every_fork_customization(self):
    assert {f"impact_{k}" for k in FORK_PROFILE} <= set(QUESTIONS)

  def test_questions_validate_against_sdk(self):
    sdk = pytest.importorskip("typesafe_sdk")
    kinds = {"noul": sdk.Noul, "choice": sdk.Choice, "score": sdk.Score}
    for q in QUESTIONS.values():
      kinds[q["type"]](**{k: v for k, v in q.items() if k != "type"})

  def test_cache_roundtrip(self, tmp_path):
    cache = JudgmentCache(tmp_path)
    assert cache.get("abc") is None
    cache.put("abc", {"answers": {}})
    assert JudgmentCache(tmp_path).get("abc") == {"answers": {}}
