#!/usr/bin/env bash
# op-wt-ls: List all openpilot worktrees with status.
set -euo pipefail

MAIN_REPO="/home/e/Development/openpilot"

echo "openpilot worktrees:"
echo "===================="
echo ""

git -C "${MAIN_REPO}" worktree list --porcelain | while IFS= read -r line; do
    case "$line" in
        worktree\ *) WT_PATH="${line#worktree }" ;;
        branch\ *)
            BRANCH="${line#branch refs/heads/}"
            DIRTY=$(git -C "${WT_PATH}" status --porcelain 2>/dev/null | wc -l)
            if [[ "${DIRTY}" -gt 0 ]]; then
                STATUS="(${DIRTY} uncommitted changes)"
            else
                STATUS="(clean)"
            fi
            printf "  %-60s  %-30s  %s\n" "${WT_PATH}" "${BRANCH}" "${STATUS}"
            ;;
        "") ;;
    esac
done
echo ""
