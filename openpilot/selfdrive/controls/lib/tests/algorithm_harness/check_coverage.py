#!/usr/bin/env python3
"""
Coverage check script for algorithm test harness.

Enforces 90% minimum coverage requirement for the algorithm harness module.
Run this script to verify coverage meets thresholds before PR.

Usage:
  python selfdrive/controls/lib/tests/algorithm_harness/check_coverage.py

Requirements:
  - pytest-cov
  - coverage

Exit codes:
  0: Coverage meets threshold (>=90%)
  1: Coverage below threshold
  2: Test failures or errors
"""

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Coverage thresholds (as percentages)
OVERALL_THRESHOLD = 90.0
MODULE_THRESHOLD = 85.0

# Modules to check coverage for
ALGORITHM_HARNESS_MODULES = [
  "interface",
  "metrics",
  "runner",
  "adapters",
  "scenario_schema",
  "scenarios",
  "scenario_generator",
]

HARNESS_PATH = "selfdrive/controls/lib/tests/algorithm_harness"


def run_tests_with_coverage() -> tuple[bool, str]:
  """Run pytest with coverage for algorithm harness."""
  cmd = [
    sys.executable,
    "-m",
    "pytest",
    HARNESS_PATH,
    f"--cov={HARNESS_PATH.replace('/', '.')}",
    "--cov-report=xml:algorithm_harness_coverage.xml",
    "--cov-report=term",
    "--cov-branch",
    "-v",
  ]

  result = subprocess.run(cmd, capture_output=True, text=True)
  return result.returncode == 0, result.stdout + result.stderr


def parse_coverage_xml(xml_path: str = "algorithm_harness_coverage.xml") -> dict:
  """Parse coverage XML report and extract module coverages."""
  if not Path(xml_path).exists():
    return {}

  tree = ET.parse(xml_path)
  root = tree.getroot()

  coverage_data = {}

  # Get overall coverage
  if root.attrib.get("line-rate"):
    coverage_data["overall"] = float(root.attrib["line-rate"]) * 100

  # Get per-module coverage
  for package in root.findall(".//package"):
    for cls in package.findall(".//class"):
      filename = cls.attrib.get("filename", "")
      if HARNESS_PATH in filename:
        module_name = Path(filename).stem
        if module_name in ALGORITHM_HARNESS_MODULES:
          line_rate = float(cls.attrib.get("line-rate", 0))
          coverage_data[module_name] = line_rate * 100

  return coverage_data


def check_thresholds(coverage_data: dict) -> tuple[bool, list[str]]:
  """Check if coverage meets thresholds."""
  failures = []

  # Check overall threshold
  overall = coverage_data.get("overall", 0)
  if overall < OVERALL_THRESHOLD:
    failures.append(f"Overall coverage {overall:.1f}% is below threshold {OVERALL_THRESHOLD}%")

  # Check per-module thresholds
  for module in ALGORITHM_HARNESS_MODULES:
    module_coverage = coverage_data.get(module, 0)
    if module_coverage < MODULE_THRESHOLD:
      failures.append(f"Module '{module}' coverage {module_coverage:.1f}% is below threshold {MODULE_THRESHOLD}%")

  return len(failures) == 0, failures


def print_coverage_report(coverage_data: dict):
  """Print coverage report in a readable format."""
  print("\n" + "=" * 60)
  print("ALGORITHM HARNESS COVERAGE REPORT")
  print("=" * 60)

  overall = coverage_data.get("overall", 0)
  status = "PASS" if overall >= OVERALL_THRESHOLD else "FAIL"
  print(f"\nOverall Coverage: {overall:.1f}% [{status}] (threshold: {OVERALL_THRESHOLD}%)")

  print(f"\nPer-Module Coverage (threshold: {MODULE_THRESHOLD}%):")
  print("-" * 40)

  for module in ALGORITHM_HARNESS_MODULES:
    module_coverage = coverage_data.get(module, 0)
    status = "PASS" if module_coverage >= MODULE_THRESHOLD else "FAIL"
    print(f"  {module:25s} {module_coverage:5.1f}% [{status}]")

  print("=" * 60)


def main():
  """Main entry point."""
  print("Running algorithm harness tests with coverage...")

  # Run tests
  tests_passed, output = run_tests_with_coverage()
  print(output)

  if not tests_passed:
    print("\nERROR: Some tests failed. Fix tests before checking coverage.")
    return 2

  # Parse coverage
  coverage_data = parse_coverage_xml()
  if not coverage_data:
    print("\nERROR: Could not parse coverage report.")
    return 2

  # Print report
  print_coverage_report(coverage_data)

  # Check thresholds
  passed, failures = check_thresholds(coverage_data)

  if not passed:
    print("\nCOVERAGE CHECK FAILED:")
    for failure in failures:
      print(f"  - {failure}")
    return 1

  print("\nCOVERAGE CHECK PASSED!")
  return 0


if __name__ == "__main__":
  sys.exit(main())
