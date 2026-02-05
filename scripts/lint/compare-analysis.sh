#!/usr/bin/env bash
set -e

# Compare MISRA analysis reports to detect regressions
# Usage: compare-analysis.sh [baseline_dir] [current_dir]
# Part of add-misra-analysis OpenSpec change

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$DIR/../../"
cd "$ROOT"

BASELINE_DIR="${1:-reports/baseline}"
CURRENT_DIR="${2:-reports}"

CPPCHECK_REPORT="cppcheck-misra-report.txt"
CLANG_TIDY_REPORT="clang-tidy-misra-report.txt"

echo "MISRA Analysis Comparison"
echo "========================="
echo "Baseline: $BASELINE_DIR"
echo "Current:  $CURRENT_DIR"
echo ""

compare_report() {
  local name="$1"
  local report="$2"
  local pattern="$3"

  local baseline="$BASELINE_DIR/$report"
  local current="$CURRENT_DIR/$report"

  echo "--- $name ---"

  if [[ ! -f "$baseline" ]]; then
    echo "  No baseline found at $baseline"
    echo "  Run the analysis first, then copy reports/ to reports/baseline/"
    echo ""
    return 0
  fi

  if [[ ! -f "$current" ]]; then
    echo "  No current report found at $current"
    echo "  Run the analysis scripts first"
    echo ""
    return 0
  fi

  local baseline_count current_count
  baseline_count=$(grep -cE "$pattern" "$baseline" 2>/dev/null || echo "0")
  current_count=$(grep -cE "$pattern" "$current" 2>/dev/null || echo "0")

  local delta=$((current_count - baseline_count))

  echo "  Baseline: $baseline_count findings"
  echo "  Current:  $current_count findings"

  if [[ "$delta" -gt 0 ]]; then
    echo "  Delta:    +$delta (REGRESSION)"
    echo ""

    # Show new findings not in baseline
    echo "  New findings:"
    diff <(sort "$baseline") <(sort "$current") | grep "^>" | head -20
    echo ""
    return 1
  elif [[ "$delta" -lt 0 ]]; then
    echo "  Delta:    $delta (IMPROVEMENT)"
  else
    echo "  Delta:    0 (no change)"
  fi
  echo ""
  return 0
}

FAILED=0

compare_report "cppcheck MISRA" "$CPPCHECK_REPORT" "misra-" || FAILED=1
compare_report "clang-tidy MISRA" "$CLANG_TIDY_REPORT" "warning:|error:" || FAILED=1

if [[ "$FAILED" -eq 1 ]]; then
  echo "RESULT: Regressions detected. Review new findings above."
  exit 1
else
  echo "RESULT: No regressions detected."
  exit 0
fi
