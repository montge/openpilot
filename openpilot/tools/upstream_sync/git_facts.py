"""Deterministic facts about an upstream sync, collected with plain git.

Everything here is exact: which commits are incoming, which files they touch, which of
those files the fork has modified or added, and which files a merge would conflict on.
Semantic judgments live in judgments.py; nothing in this module calls a model.
"""

import builtins
import keyword
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# Files whose diffs are noise for a semantic read (lockfiles, generated data, binaries).
NOISY_SUFFIXES = (".lock", ".svg", ".png", ".jpg", ".onnx", ".pkl", ".bin", ".ts", ".json")
DIFF_EXCERPT_CHARS = 24_000

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
PY_DEF = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\((.*)")
COMMON = set(keyword.kwlist) | set(dir(builtins))
# Fork files whose mentions of an upstream name are prose, not code that can break.
REFERENCE_EXCLUDES = ("openspec/", ".claude/", "docs/")
MAX_REFERENCE_FILES = 8


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
  return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check)


@dataclass
class FileChange:
  status: str  # A/M/D/R/T as reported by git
  path: str
  submodule: bool = False


@dataclass
class Commit:
  sha: str
  subject: str
  body: str
  author: str
  date: str
  files: list[FileChange] = field(default_factory=list)
  # names this commit removes or re-signatures -> fork files that still use them
  fork_references: dict[str, list[str]] = field(default_factory=dict)

  @property
  def paths(self) -> list[str]:
    return [f.path for f in self.files]


@dataclass
class ForkOverlay:
  """What the fork carries on top of the last upstream point it merged."""
  base: str
  modified: set[str]  # upstream files the fork edits
  added: set[str]  # files that exist only in the fork
  deleted: set[str]  # upstream files the fork removed


@dataclass
class SyncFacts:
  fork_ref: str
  upstream_ref: str
  merge_base: str
  overlay: ForkOverlay
  commits: list[Commit]
  conflicts: list[str]  # files a merge of upstream_ref into fork_ref conflicts on


def resolve(repo: Path, ref: str) -> str:
  return git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").stdout.strip()


def fork_overlay(repo: Path, base: str, fork_ref: str) -> ForkOverlay:
  out = git(repo, "diff", "--no-renames", "--name-status", base, fork_ref).stdout
  modified, added, deleted = set(), set(), set()
  for line in out.splitlines():
    status, path = line.split("\t", 1)
    {"M": modified, "A": added, "D": deleted}.get(status[0], modified).add(path)
  return ForkOverlay(base=base, modified=modified, added=added, deleted=deleted)


def incoming_commits(repo: Path, fork_ref: str, upstream_ref: str) -> list[Commit]:
  """Non-merge commits reachable from upstream_ref but not fork_ref, oldest first."""
  sep, end = "\x1f", "\x1e"
  fmt = sep.join(["%H", "%s", "%b", "%an", "%cs"]) + end
  out = git(repo, "log", "--reverse", "--no-merges", f"--format={fmt}", f"{fork_ref}..{upstream_ref}").stdout
  commits = []
  for record in out.split(end):
    record = record.strip("\n")
    if not record:
      continue
    sha, subject, body, author, date = record.split(sep)
    commits.append(Commit(sha=sha, subject=subject, body=body.strip(), author=author, date=date))

  for c in commits:
    c.files = commit_files(repo, c.sha)
  return commits


def commit_files(repo: Path, sha: str) -> list[FileChange]:
  # --raw exposes file modes, which is how a submodule pin bump (mode 160000) shows up
  out = git(repo, "diff-tree", "--no-commit-id", "-r", "--raw", "--no-renames", sha).stdout
  files = []
  for line in out.splitlines():
    meta, path = line.split("\t", 1)
    old_mode, new_mode, _, _, status = meta.lstrip(":").split()
    files.append(FileChange(status=status[0], path=path, submodule="160000" in (old_mode, new_mode)))
  return files


def diff_excerpt(repo: Path, sha: str, limit: int = DIFF_EXCERPT_CHARS) -> str:
  """The commit's patch minus noisy files, truncated to fit a model's state budget."""
  excludes = [f":(exclude,glob)**/*{s}" for s in NOISY_SUFFIXES]
  patch = git(repo, "show", "--format=", "--no-color", "--unified=2", sha, "--", ".", *excludes).stdout
  if len(patch) > limit:
    patch = patch[:limit] + f"\n... [truncated {len(patch) - limit} more characters]"
  return patch


def changed_names(repo: Path, commit: Commit) -> tuple[set[str], set[str]]:
  """(names the commit's diff removes, Python functions whose signature it changes).

  A removed name is an identifier on '-' lines that appears on no '+' line, plus the
  module path of every deleted .py file. Whether it is really gone from upstream is
  checked separately against the tree.
  """
  excludes = [f":(exclude,glob)**/*{s}" for s in NOISY_SUFFIXES]
  patch = git(repo, "show", "--format=", "--no-color", "--unified=0", commit.sha, "--", ".", *excludes).stdout
  minus: set[str] = set()
  plus: set[str] = set()
  defs_minus: dict[str, str] = {}
  defs_plus: dict[str, str] = {}
  for line in patch.splitlines():
    if line.startswith(("---", "+++")) or line[:1] not in "+-":
      continue
    tokens, defs = (minus, defs_minus) if line[0] == "-" else (plus, defs_plus)
    tokens.update(IDENT.findall(line[1:]))
    if m := PY_DEF.match(line[1:]):
      defs[m.group(1)] = m.group(2).strip()

  removed = minus - plus - COMMON
  for f in commit.files:
    if f.status == "D" and f.path.endswith(".py"):
      removed.add(f.path.removesuffix(".py").removesuffix("/__init__").replace("/", "."))
  # dunders (__init__, __call__) are too generic to attribute to one class by name alone
  resigned = {n for n, args in defs_minus.items()
              if n in defs_plus and defs_plus[n] != args and n not in COMMON and not n.startswith("__")}
  return removed, resigned


def _grep(repo: Path, rev: str, names: set[str], paths: list[str] | None = None) -> list[tuple[str, str]]:
  """(path, name) for every whole-word occurrence of names in rev's tree."""
  if not names:
    return []
  with tempfile.NamedTemporaryFile("w", suffix=".pats") as pats:
    pats.write("\n".join(sorted(names)))
    pats.flush()
    args = ["grep", "-I", "-w", "-o", "-F", "-f", pats.name, rev]
    if paths is not None:
      args += ["--", *paths]
    out = git(repo, *args, check=False).stdout
  hits = []
  for line in out.splitlines():
    _, path, name = line.split(":", 2)
    hits.append((path, name))
  return hits


@dataclass
class UpstreamTree:
  """What exists in the upstream ref being merged: the target a fork reference must still resolve in."""
  names: set[str]
  files: set[str]

  @classmethod
  def load(cls, repo: Path, rev: str) -> "UpstreamTree":
    names = git(repo, "grep", "-I", "-h", "-o", "-w", "-E", IDENT.pattern, rev, check=False).stdout.split()
    return cls(names=set(names), files=set(git(repo, "ls-tree", "-r", "--name-only", rev).stdout.splitlines()))

  def has(self, name: str) -> bool:
    if "." in name:  # a module path from a deleted file
      path = name.replace(".", "/")
      return f"{path}.py" in self.files or f"{path}/__init__.py" in self.files
    return name in self.names


def fork_references(repo: Path, commit: Commit, fork_sha: str, fork_files: list[str], upstream: UpstreamTree) -> dict[str, list[str]]:
  """Names this commit takes away (or re-signatures) that fork-owned files still use.

  A removed name only counts if the upstream ref being merged no longer has it, so a
  name dropped in one commit and restored by a later one is not a break.
  """
  removed, resigned = changed_names(repo, commit)
  candidates = {n for n in removed if not upstream.has(n)} | resigned
  refs: dict[str, set[str]] = {}
  for path, name in _grep(repo, fork_sha, candidates, fork_files):
    refs.setdefault(name, set()).add(path)
  return {name: sorted(paths)[:MAX_REFERENCE_FILES] for name, paths in sorted(refs.items())}


def predicted_conflicts(repo: Path, fork_ref: str, upstream_ref: str) -> list[str]:
  """Files that `git merge upstream_ref` into fork_ref would leave conflicted (no worktree changes)."""
  res = git(repo, "merge-tree", "--write-tree", "--name-only", "--no-messages", fork_ref, upstream_ref, check=False)
  if res.returncode == 0:
    return []
  if res.returncode != 1:
    raise RuntimeError(f"git merge-tree failed: {res.stderr.strip()}")
  # first line is the resulting tree id; the conflicted paths follow
  return [p for p in res.stdout.splitlines()[1:] if p]


def collect(repo: Path, fork_ref: str, upstream_ref: str) -> SyncFacts:
  fork_sha, upstream_sha = resolve(repo, fork_ref), resolve(repo, upstream_ref)
  base = git(repo, "merge-base", fork_sha, upstream_sha).stdout.strip()
  overlay = fork_overlay(repo, base, fork_sha)
  commits = incoming_commits(repo, fork_sha, upstream_sha)

  # fork code that can break: everything the fork added or edits, minus prose
  fork_files = sorted(p for p in overlay.added | overlay.modified if not p.startswith(REFERENCE_EXCLUDES) and not p.endswith(".md"))
  upstream = UpstreamTree.load(repo, upstream_sha)
  with ThreadPoolExecutor(max_workers=8) as pool:
    refs = pool.map(lambda c: fork_references(repo, c, fork_sha, fork_files, upstream), commits)
    for c, r in zip(commits, refs, strict=True):
      c.fork_references = r

  return SyncFacts(
    fork_ref=fork_ref,
    upstream_ref=upstream_ref,
    merge_base=base,
    overlay=overlay,
    commits=commits,
    conflicts=predicted_conflicts(repo, fork_sha, upstream_sha),
  )
