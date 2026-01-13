#!/usr/bin/env python3
"""CLI tool for analyzing shadow device comparison logs.

Usage:
  python tools/shadow/analyze.py --shadow /path/to/shadow/segment --prod /path/to/prod/segment
  python tools/shadow/analyze.py --shadow /path/to/shadow/segment --prod /path/to/prod/segment --output report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from openpilot.tools.shadow.align import LogAligner, validate_alignment
from openpilot.tools.shadow.comparison_logger import ComparisonLogger
from openpilot.tools.shadow.metrics import (
  compute_all_metrics,
  format_report_markdown,
)


def load_and_align(
  shadow_dir: Path,
  prod_dir: Path,
  method: str = "auto",
) -> dict[str, Any]:
  """Load and align logs from two directories.

  Args:
    shadow_dir: Path to shadow device logs
    prod_dir: Path to production device logs
    method: Alignment method ("auto", "gps", "frame_id", "timestamp")

  Returns:
    Dictionary with alignment result and validation
  """
  print(f"Loading shadow logs from: {shadow_dir}")
  shadow_frames = ComparisonLogger.load_segment(shadow_dir)
  print(f"  Loaded {len(shadow_frames)} frames")

  print(f"Loading production logs from: {prod_dir}")
  prod_frames = ComparisonLogger.load_segment(prod_dir)
  print(f"  Loaded {len(prod_frames)} frames")

  print(f"Aligning logs using method: {method}")
  aligner = LogAligner()

  if method == "gps":
    result = aligner.align_by_gps(shadow_frames, prod_frames)
  elif method == "frame_id":
    result = aligner.align_by_frame_id(shadow_frames, prod_frames)
  elif method == "timestamp":
    result = aligner.align_by_timestamp(shadow_frames, prod_frames)
  else:  # auto
    result = aligner.auto_align(shadow_frames, prod_frames)

  print(f"  Aligned {len(result.pairs)} pairs")
  print(f"  Shadow-only: {len(result.shadow_only)} frames")
  print(f"  Production-only: {len(result.production_only)} frames")

  validation = validate_alignment(result)
  print(f"  Quality: {result.alignment_quality:.2%}")
  print(f"  Valid: {validation['valid']}")

  return {
    "result": result,
    "validation": validation,
    "shadow_count": len(shadow_frames),
    "prod_count": len(prod_frames),
  }


def analyze_and_report(
  shadow_dir: Path,
  prod_dir: Path,
  output_path: Path | None = None,
  method: str = "auto",
  json_output: bool = False,
) -> int:
  """Run full analysis and generate report.

  Args:
    shadow_dir: Path to shadow device logs
    prod_dir: Path to production device logs
    output_path: Optional path to write report
    method: Alignment method
    json_output: Output JSON instead of markdown

  Returns:
    Exit code (0 for success)
  """
  # Load and align
  alignment = load_and_align(shadow_dir, prod_dir, method)
  result = alignment["result"]
  validation = alignment["validation"]

  if not validation["valid"]:
    print(f"\nWarning: Alignment quality is poor ({validation.get('reason', 'low quality')})")

  # Compute metrics
  print("\nComputing metrics...")
  report = compute_all_metrics(result)

  if json_output:
    # Output as JSON
    output_data = {
      "alignment": {
        "method": result.method,
        "quality": result.alignment_quality,
        "pairs": len(result.pairs),
        "shadow_only": len(result.shadow_only),
        "production_only": len(result.production_only),
        "mean_time_offset_ms": result.mean_time_offset_ms,
      },
      "validation": validation,
      "control_metrics": {
        "steer_rmse": report.control_metrics.steer_rmse,
        "steer_mae": report.control_metrics.steer_mae,
        "steer_max_error": report.control_metrics.steer_max_error,
        "accel_rmse": report.control_metrics.accel_rmse,
        "accel_mae": report.control_metrics.accel_mae,
        "accel_max_error": report.control_metrics.accel_max_error,
      },
    }

    if report.trajectory_metrics:
      output_data["trajectory_metrics"] = {
        "path_rmse": report.trajectory_metrics.path_rmse,
        "path_mae": report.trajectory_metrics.path_mae,
        "path_max_error": report.trajectory_metrics.path_max_error,
        "speed_rmse": report.trajectory_metrics.speed_rmse,
        "speed_mae": report.trajectory_metrics.speed_mae,
        "speed_max_error": report.trajectory_metrics.speed_max_error,
      }

    if report.model_metrics:
      output_data["model_metrics"] = {
        "curvature_rmse": report.model_metrics.curvature_rmse,
        "curvature_mae": report.model_metrics.curvature_mae,
        "curvature_correlation": report.model_metrics.curvature_correlation,
      }

    output_str = json.dumps(output_data, indent=2)
  else:
    # Output as markdown
    output_str = format_report_markdown(report)

  # Write or print output
  if output_path:
    output_path.write_text(output_str)
    print(f"\nReport written to: {output_path}")
  else:
    print("\n" + "=" * 60)
    print(output_str)

  return 0


def main():
  parser = argparse.ArgumentParser(description="Analyze shadow device comparison logs")
  parser.add_argument(
    "--shadow",
    type=Path,
    required=True,
    help="Path to shadow device log segment",
  )
  parser.add_argument(
    "--prod",
    type=Path,
    required=True,
    help="Path to production device log segment",
  )
  parser.add_argument(
    "--output",
    "-o",
    type=Path,
    help="Output file path (default: print to stdout)",
  )
  parser.add_argument(
    "--method",
    choices=["auto", "gps", "frame_id", "timestamp"],
    default="auto",
    help="Alignment method (default: auto)",
  )
  parser.add_argument(
    "--json",
    action="store_true",
    help="Output JSON instead of markdown",
  )

  args = parser.parse_args()

  # Validate paths
  if not args.shadow.exists():
    print(f"Error: Shadow path does not exist: {args.shadow}", file=sys.stderr)
    return 1

  if not args.prod.exists():
    print(f"Error: Production path does not exist: {args.prod}", file=sys.stderr)
    return 1

  return analyze_and_report(
    shadow_dir=args.shadow,
    prod_dir=args.prod,
    output_path=args.output,
    method=args.method,
    json_output=args.json,
  )


if __name__ == "__main__":
  sys.exit(main())
