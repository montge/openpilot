#!/usr/bin/env bash
# Coverage Ratchet Script
# Updates the fail_under threshold in pyproject.toml to the current coverage
# This ensures coverage can only go up, never down.
#
# Usage: ./scripts/ratchet-coverage.sh [--dry-run]

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$DIR/.."
cd "$ROOT"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
  DRY_RUN=true
fi

# Activate venv if not already
if [[ -z "$VIRTUAL_ENV" ]]; then
  source .venv/bin/activate
fi

# Get current coverage percentage
CURRENT_COVERAGE=$(coverage report 2>/dev/null | grep "^TOTAL" | awk '{print $NF}' | sed 's/%//')

if [[ -z "$CURRENT_COVERAGE" ]]; then
  echo "Error: Could not determine current coverage. Run tests with coverage first:"
  echo "  pytest --cov=selfdrive --cov=system --cov=common -m 'not slow and not tici'"
  exit 1
fi

# Get current threshold from pyproject.toml
CURRENT_THRESHOLD=$(grep "^fail_under" pyproject.toml | awk -F'=' '{print $2}' | tr -d ' ')

echo "Current coverage:  ${CURRENT_COVERAGE}%"
echo "Current threshold: ${CURRENT_THRESHOLD}%"

# Compare as integers
COVERAGE_INT=${CURRENT_COVERAGE%.*}
THRESHOLD_INT=${CURRENT_THRESHOLD%.*}

if [[ $COVERAGE_INT -gt $THRESHOLD_INT ]]; then
  echo ""
  echo "Coverage improved! Updating threshold to ${COVERAGE_INT}%"

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would update fail_under from $CURRENT_THRESHOLD to $COVERAGE_INT"
  else
    # Update pyproject.toml
    sed -i "s/^fail_under = .*/fail_under = $COVERAGE_INT/" pyproject.toml
    echo "Updated pyproject.toml"
    echo ""
    echo "Don't forget to commit this change:"
    echo "  git add pyproject.toml"
    echo "  git commit -m 'chore: ratchet coverage threshold to ${COVERAGE_INT}%'"
  fi
elif [[ $COVERAGE_INT -lt $THRESHOLD_INT ]]; then
  echo ""
  echo "WARNING: Coverage dropped below threshold!"
  echo "Current: ${COVERAGE_INT}% < Threshold: ${THRESHOLD_INT}%"
  echo ""
  echo "Either add more tests or investigate what reduced coverage."
  exit 1
else
  echo ""
  echo "Coverage unchanged. No update needed."
fi
