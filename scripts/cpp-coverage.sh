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

# Build with coverage instrumentation
echo "Building with --coverage..."
scons -u -j$(nproc) --coverage

echo ""
echo "=== Running C++ Tests ==="

# Set up profraw output
export LLVM_PROFILE_FILE="$ROOT/default_%p.profraw"

# Run C++ tests (catch2-based)
# Common tests
if [[ -f common/tests/test_runner ]]; then
  echo "Running common tests..."
  ./common/tests/test_runner || true
fi

# Loggerd tests
if [[ -f system/loggerd/tests/test_runner ]]; then
  echo "Running loggerd tests..."
  ./system/loggerd/tests/test_runner || true
fi

echo ""
echo "=== Generating Coverage Report ==="

# Merge profraw files
PROFRAW_FILES=$(ls *.profraw 2>/dev/null || true)
if [[ -z "$PROFRAW_FILES" ]]; then
  echo "No .profraw files found. Tests may not have run."
  exit 1
fi

llvm-profdata merge -sparse *.profraw -o coverage.profdata

# Find all instrumented binaries
BINARIES=""
for bin in common/tests/test_runner system/loggerd/tests/test_runner; do
  if [[ -f "$bin" ]]; then
    BINARIES="$BINARIES -object=$bin"
  fi
done

if [[ -z "$BINARIES" ]]; then
  echo "No test binaries found."
  exit 1
fi

# Generate text report
echo ""
echo "Coverage Summary:"
llvm-cov report $BINARIES -instr-profile=coverage.profdata \
  -ignore-filename-regex='third_party|msgq_repo|opendbc_repo|tinygrad_repo|rednose_repo|cereal'

# Generate detailed report
llvm-cov show $BINARIES -instr-profile=coverage.profdata \
  -ignore-filename-regex='third_party|msgq_repo|opendbc_repo|tinygrad_repo|rednose_repo|cereal' \
  -format=text > cpp-coverage-details.txt

echo ""
echo "Detailed report: cpp-coverage-details.txt"

# Generate HTML report if requested
if [[ "$GENERATE_HTML" == "true" ]]; then
  echo "Generating HTML report..."
  llvm-cov show $BINARIES -instr-profile=coverage.profdata \
    -ignore-filename-regex='third_party|msgq_repo|opendbc_repo|tinygrad_repo|rednose_repo|cereal' \
    -format=html -output-dir=cpp-coverage-report
  echo "HTML report: cpp-coverage-report/index.html"
fi

# Export to lcov format for Codecov
llvm-cov export $BINARIES -instr-profile=coverage.profdata \
  -ignore-filename-regex='third_party|msgq_repo|opendbc_repo|tinygrad_repo|rednose_repo|cereal' \
  -format=lcov > cpp-coverage.lcov

echo "LCOV report: cpp-coverage.lcov"

# Cleanup
rm -f *.profraw
