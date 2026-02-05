#!/usr/bin/env bash
# op-wt-rm: Remove an openpilot git worktree and optionally its branch.
#
# Usage:
#   op-wt-rm <worktree-dir-name> [--delete-branch]
set -euo pipefail

MAIN_REPO="/home/e/Development/openpilot"
WT_BASE="/home/e/Development/openpilot-worktrees"
DELETE_BRANCH=false
WT_DIR_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --delete-branch) DELETE_BRANCH=true; shift ;;
        *) WT_DIR_NAME="$1"; shift ;;
    esac
done

if [[ -z "${WT_DIR_NAME}" ]]; then
    echo "Usage: op-wt-rm <worktree-dir-name> [--delete-branch]"
    echo ""; echo "Active worktrees:"
    git -C "${MAIN_REPO}" worktree list
    exit 1
fi

WT_PATH="${WT_BASE}/${WT_DIR_NAME}"

if [[ ! -d "${WT_PATH}" ]]; then
    echo "ERROR: Worktree not found: ${WT_PATH}"
    git -C "${MAIN_REPO}" worktree list
    exit 1
fi

BRANCH=$(git -C "${WT_PATH}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
echo "Removing worktree at ${WT_PATH} (branch: ${BRANCH})..."

git -C "${MAIN_REPO}" worktree remove "${WT_PATH}" --force

if [[ "${DELETE_BRANCH}" == true && -n "${BRANCH}" && "${BRANCH}" != "master" && "${BRANCH}" != "develop" ]]; then
    echo "Deleting branch '${BRANCH}'..."
    git -C "${MAIN_REPO}" branch -d "${BRANCH}" 2>/dev/null || \
        echo "WARNING: Branch '${BRANCH}' not fully merged. Use 'git branch -D ${BRANCH}' to force."
fi

git -C "${MAIN_REPO}" worktree prune
echo "Done."
