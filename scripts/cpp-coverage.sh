#!/usr/bin/env bash
# C++ Coverage Script
# Builds with coverage instrumentation, runs tests, and generates reports
#
# Usage: ./scripts/cpp-coverage.sh [--html]

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$DIR/.."
cd "$ROOT"

GENERATE_HTML=false
if [[ "$1" == "--html" ]]; then
  GENERATE_HTML=true
fi

echo "=== C++ Coverage Build ==="

# Clean previous coverage data
rm -f *.profraw *.profdata
rm -rf cpp-coverage-report/

# Activate virtual environment if it exists
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  source "$ROOT/.venv/bin/activate"
fi

# Build with coverage instrumentation
echo "Building with --coverage..."
# the C++ tests upstream still builds (#38408); keep in sync with .github/workflows/cpp-coverage.yml
TEST_BINARIES=(
  openpilot/common/tests/test_swaglog
  openpilot/selfdrive/pandad/tests/test_pandad_canprotocol
  openpilot/tools/cabana/tests/test_cabana
)
scons -u -j"$(nproc)" --coverage openpilot/common/tests/ openpilot/selfdrive/pandad/tests/ openpilot/tools/cabana/tests/

echo ""
echo "=== Running C++ Tests ==="

# Set up profraw output
export LLVM_PROFILE_FILE="$ROOT/default_%p.profraw"

for bin in "${TEST_BINARIES[@]}"; do
  if [[ -f "$bin" ]]; then
    echo "Running $bin..."
    "./$bin" || true
  fi
done

echo ""
echo "=== Generating Coverage Report ==="

# Merge profraw files
PROFRAW_FILES=$(ls *.profraw 2>/dev/null || true)
if [[ -z "$PROFRAW_FILES" ]]; then
  echo "No .profraw files found. Tests may not have run."
  exit 1
fi

# Find llvm-profdata (prefer versioned)
LLVM_PROFDATA=$(command -v llvm-profdata-18 || command -v llvm-profdata-17 || command -v llvm-profdata)
LLVM_COV=$(command -v llvm-cov-18 || command -v llvm-cov-17 || command -v llvm-cov)

if [[ -z "$LLVM_PROFDATA" ]] || [[ -z "$LLVM_COV" ]]; then
  echo "ERROR: llvm-profdata and llvm-cov are required. Install llvm."
  exit 1
fi

echo "Using: $LLVM_PROFDATA"
echo "Using: $LLVM_COV"

"$LLVM_PROFDATA" merge -sparse ./*.profraw -o coverage.profdata

# Find all instrumented binaries
OBJECTS=()
for bin in "${TEST_BINARIES[@]}"; do
  if [[ -f "$bin" ]]; then
    OBJECTS+=("-object=$bin")
  fi
done

if [[ ${#OBJECTS[@]} -eq 0 ]]; then
  echo "No test binaries found."
  exit 1
fi

# Generate text report
echo ""
echo "Coverage Summary:"
"$LLVM_COV" report "${OBJECTS[@]}" -instr-profile=coverage.profdata \
  -ignore-filename-regex='third_party|msgq_repo|opendbc_repo|tinygrad_repo|rednose_repo|cereal'

# Generate detailed report
"$LLVM_COV" show "${OBJECTS[@]}" -instr-profile=coverage.profdata \
  -ignore-filename-regex='third_party|msgq_repo|opendbc_repo|tinygrad_repo|rednose_repo|cereal' \
  -format=text > cpp-coverage-details.txt

echo ""
echo "Detailed report: cpp-coverage-details.txt"

# Generate HTML report if requested
if [[ "$GENERATE_HTML" == "true" ]]; then
  echo "Generating HTML report..."
  "$LLVM_COV" show "${OBJECTS[@]}" -instr-profile=coverage.profdata \
    -ignore-filename-regex='third_party|msgq_repo|opendbc_repo|tinygrad_repo|rednose_repo|cereal' \
    -format=html -output-dir=cpp-coverage-report
  echo "HTML report: cpp-coverage-report/index.html"
fi

# Export to lcov format for Codecov
"$LLVM_COV" export "${OBJECTS[@]}" -instr-profile=coverage.profdata \
  -ignore-filename-regex='third_party|msgq_repo|opendbc_repo|tinygrad_repo|rednose_repo|cereal' \
  -format=lcov > cpp-coverage.lcov

echo "LCOV report: cpp-coverage.lcov"

# Cleanup
rm -f *.profraw
