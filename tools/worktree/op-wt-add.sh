#!/usr/bin/env bash
# op-wt-add: Create an openpilot git worktree with submodule initialization.
#
# Usage:
#   op-wt-add <branch-name> [--base <base-branch>]
#
# Examples:
#   op-wt-add master
#   op-wt-add feature/privacy-zones
#   op-wt-add feature/add-widgets --base develop
set -euo pipefail

MAIN_REPO="/home/e/Development/openpilot"
WT_BASE="/home/e/Development/openpilot-worktrees"
BRANCH=""
BASE_BRANCH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base) BASE_BRANCH="$2"; shift 2 ;;
        *) BRANCH="$1"; shift ;;
    esac
done

if [[ -z "$BRANCH" ]]; then
    echo "Usage: op-wt-add <branch-name> [--base <base-branch>]"
    exit 1
fi

WT_DIR_NAME="${BRANCH//\//__}"
WT_PATH="${WT_BASE}/${WT_DIR_NAME}"
mkdir -p "${WT_BASE}"

if [[ -d "${WT_PATH}" ]]; then
    echo "ERROR: Worktree already exists at ${WT_PATH}"
    exit 1
fi

echo "Creating worktree for branch '${BRANCH}' at ${WT_PATH}..."

BRANCH_EXISTS=$(git -C "${MAIN_REPO}" branch --list "${BRANCH}" | wc -l)

if [[ "${BRANCH_EXISTS}" -gt 0 ]]; then
    git -C "${MAIN_REPO}" worktree add "${WT_PATH}" "${BRANCH}"
elif [[ -n "${BASE_BRANCH}" ]]; then
    git -C "${MAIN_REPO}" worktree add -b "${BRANCH}" "${WT_PATH}" "${BASE_BRANCH}"
else
    REMOTE_EXISTS=$(git -C "${MAIN_REPO}" branch -r --list "origin/${BRANCH}" | wc -l)
    if [[ "${REMOTE_EXISTS}" -gt 0 ]]; then
        git -C "${MAIN_REPO}" worktree add --track -b "${BRANCH}" "${WT_PATH}" "origin/${BRANCH}"
    else
        echo "ERROR: Branch '${BRANCH}' does not exist locally or on origin."
        echo "Use --base <branch> to create it."
        exit 1
    fi
fi

echo "Initializing submodules..."
git -C "${WT_PATH}" submodule update --init --recursive

echo ""
echo "Worktree ready:"
echo "  Path:   ${WT_PATH}"
echo "  Branch: ${BRANCH}"
echo "  cd ${WT_PATH}"
