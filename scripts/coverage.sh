#!/usr/bin/env bash
# Run tests with coverage and generate reports
# Usage: ./scripts/coverage.sh [pytest args...]

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$DIR/.."
cd "$ROOT"

echo "Running tests with coverage..."

# Default args if none provided (an array, so the -m expression stays one argument)
if [[ $# -gt 0 ]]; then
  PYTEST_ARGS=("$@")
else
  PYTEST_ARGS=(--cov --cov-report=term-missing --cov-report=html -m "not slow" -n auto)
fi

# Activate venv if not already
if [[ -z "$VIRTUAL_ENV" ]]; then
  source .venv/bin/activate
fi

# Run pytest with coverage
pytest "${PYTEST_ARGS[@]}"

echo ""
echo "Coverage report generated:"
echo "  - Terminal: shown above"
echo "  - HTML:     htmlcov/index.html"
echo "  - XML:      coverage.xml (for CI)"
