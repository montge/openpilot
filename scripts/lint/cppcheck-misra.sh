#!/usr/bin/env bash
set -e

# MISRA C:2012 baseline analysis using cppcheck
# Part of add-misra-analysis OpenSpec change

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$DIR/../../"
cd "$ROOT"

# Output file
REPORT_FILE="${1:-reports/cppcheck-misra-report.txt}"
mkdir -p "$(dirname "$REPORT_FILE")"

# Directories to analyze
TARGET_DIRS="selfdrive system common"

# Exclusions (third-party, submodules, generated code)
EXCLUDES=(
  --suppress="*:third_party/*"
  --suppress="*:msgq_repo/*"
  --suppress="*:opendbc_repo/*"
  --suppress="*:rednose_repo/*"
  --suppress="*:tinygrad_repo/*"
  --suppress="*:teleoprtc_repo/*"
  --suppress="*:cereal/gen/*"
  --suppress="*:.venv/*"
  -i third_party
  -i msgq_repo
  -i opendbc_repo
  -i rednose_repo
  -i tinygrad_repo
  -i teleoprtc_repo
  -i .venv
)

echo "Running cppcheck MISRA C:2012 analysis..."
echo "Target directories: $TARGET_DIRS"
echo "Output: $REPORT_FILE"
echo ""

# Run cppcheck with MISRA addon
# --addon=misra enables MISRA C:2012 checks
# --enable=warning,style,performance enables additional checks
# --error-exitcode=0 prevents non-zero exit on findings
cppcheck \
  --addon=misra \
  --enable=warning,style,performance \
  --error-exitcode=0 \
  --inline-suppr \
  "${EXCLUDES[@]}" \
  --output-file="$REPORT_FILE" \
  $TARGET_DIRS 2>&1

# Count findings
if [[ -f "$REPORT_FILE" ]]; then
  TOTAL=$(grep -c "misra-" "$REPORT_FILE" 2>/dev/null || echo "0")
  echo ""
  echo "Analysis complete. Found $TOTAL MISRA findings."
  echo "Report saved to: $REPORT_FILE"

  # Show summary by rule if findings exist
  if [[ "$TOTAL" -gt 0 ]]; then
    echo ""
    echo "Top 10 rules by frequency:"
    grep -oE '\[misra-[a-z]+-[0-9]+\.[0-9]+\]' "$REPORT_FILE" 2>/dev/null | \
      sort | uniq -c | sort -rn | head -10
  fi
else
  echo "Warning: Report file was not created"
fi
