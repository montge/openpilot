#!/usr/bin/env bash
set -e

# MISRA C:2025 and C++:2023 analysis using clang-tidy-automotive
# Part of add-misra-analysis OpenSpec change

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$DIR/../../"
cd "$ROOT"

# Configuration
CLANG_TIDY_AUTOMOTIVE="${CLANG_TIDY_AUTOMOTIVE:-/home/e/Development/clang-tidy-automotive/build/bin/clang-tidy}"
REPORT_FILE="${1:-reports/clang-tidy-misra-report.txt}"
mkdir -p "$(dirname "$REPORT_FILE")"

# Verify clang-tidy-automotive exists
if [[ ! -x "$CLANG_TIDY_AUTOMOTIVE" ]]; then
  echo "Error: clang-tidy-automotive not found at $CLANG_TIDY_AUTOMOTIVE"
  echo "Set CLANG_TIDY_AUTOMOTIVE environment variable to the correct path"
  exit 1
fi

# Check for compile_commands.json
if [[ ! -f "compile_commands.json" ]]; then
  echo "Error: compile_commands.json not found"
  echo "Run 'scons -u' first to generate compilation database"
  exit 1
fi

# Get list of C/C++ files to analyze from compile_commands.json
# Exclude third_party, submodules, generated code
echo "Extracting files from compile_commands.json..."
FILES=$(python3 -c "
import json
with open('compile_commands.json') as f:
    data = json.load(f)

excludes = [
    'third_party/', 'msgq_repo/', 'opendbc_repo/', 'rednose_repo/',
    'tinygrad_repo/', 'teleoprtc_repo/', 'cereal/gen/', '.venv/',
    'panda/', 'c_generated_code/',
    # Qt-based tools excluded (cabana/replay use Qt heavily)
    'tools/cabana/', 'tools/replay/',
]

for entry in data:
    file = entry['file']
    if file.endswith(('.c', '.cc', '.cpp')):
        if not any(ex in file for ex in excludes):
            print(file)
" | sort -u)

FILE_COUNT=$(echo "$FILES" | wc -l)
echo "Found $FILE_COUNT files to analyze"
echo ""

# Checks to enable - all automotive checks for MISRA C:2025 and C++:2023
# Exclude automotive-cpp23-req-8.3.1 due to crash bug (see reports/clang-tidy-automotive-issues.md)
CHECKS="automotive-*,-automotive-cpp23-req-8.3.1"

echo "Running clang-tidy-automotive MISRA analysis..."
echo "Checks: $CHECKS"
echo "Output: $REPORT_FILE"
echo ""

# Clear previous report
> "$REPORT_FILE"

# Run clang-tidy on each file
ANALYZED=0
for file in $FILES; do
  if [[ -f "$file" ]]; then
    ANALYZED=$((ANALYZED + 1))
    echo -ne "\rAnalyzing file $ANALYZED/$FILE_COUNT: $(basename "$file")          "

    "$CLANG_TIDY_AUTOMOTIVE" \
      -checks="-*,$CHECKS" \
      -p . \
      --quiet \
      "$file" 2>&1 >> "$REPORT_FILE" || true
  fi
done

echo -e "\n"

# Count findings
if [[ -f "$REPORT_FILE" ]]; then
  TOTAL=$(grep -cE "warning:|error:" "$REPORT_FILE" 2>/dev/null || echo "0")
  echo "Analysis complete. Found $TOTAL findings."
  echo "Report saved to: $REPORT_FILE"

  # Show summary by check if findings exist
  if [[ "$TOTAL" -gt 0 ]]; then
    echo ""
    echo "Top 15 checks by frequency:"
    grep -oE '\[automotive-[a-zA-Z0-9.-]+\]' "$REPORT_FILE" 2>/dev/null | \
      sort | uniq -c | sort -rn | head -15
  fi
else
  echo "Warning: Report file was not created"
fi
