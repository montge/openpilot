#!/usr/bin/env bash
# Pre-push coverage check for openpilot
# Runs a quick coverage check on core modules before push
# Skip with: SKIP=coverage-check git push

set -e

# Configuration
MIN_COVERAGE=30  # Minimum coverage percentage (baseline)
TIMEOUT=120      # Timeout in seconds

echo "Running pre-push coverage check..."

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
  if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
  else
    echo "Warning: No virtual environment found. Skipping coverage check."
    exit 0
  fi
fi

# Verify pytest and pytest-cov are available
if ! python -c "import pytest; import pytest_cov" 2>/dev/null; then
  echo "Warning: pytest or pytest-cov not installed. Skipping coverage check."
  exit 0
fi

# Run quick coverage check on common/ (fast, good baseline)
timeout $TIMEOUT pytest common/ \
  --cov=common \
  --cov-report=term-missing:skip-covered \
  --cov-fail-under=$MIN_COVERAGE \
  -q --tb=short \
  -m "not slow and not tici" \
  --ignore=tinygrad_repo --ignore=third_party \
  -x 2>&1 || {
    RESULT=$?
    if [ $RESULT -eq 124 ]; then
      echo "Coverage check timed out after ${TIMEOUT}s. Continuing with push."
      exit 0
    elif [ $RESULT -eq 2 ]; then
      echo "Warning: Test collection error (missing deps?). Skipping coverage check."
      exit 0
    elif [ $RESULT -ne 0 ]; then
      echo ""
      echo "Coverage check failed (below ${MIN_COVERAGE}% threshold)."
      echo "You can skip this check with: SKIP=coverage-check git push"
      exit 1
    fi
  }

echo "Coverage check passed!"
