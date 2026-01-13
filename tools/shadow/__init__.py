"""Shadow device comparison testing tools.

This module provides tools for capturing, synchronizing, and analyzing
logs from shadow devices running in parallel with production devices.
"""

from openpilot.tools.shadow.align import (
  AlignedPair,
  AlignmentResult,
  LogAligner,
  merge_aligned_logs,
  validate_alignment,
)
from openpilot.tools.shadow.comparison_logger import ComparisonLogger, FrameData
from openpilot.tools.shadow.metrics import (
  ComparisonReport,
  ControlMetrics,
  ModelMetrics,
  TrajectoryMetrics,
  compute_all_metrics,
  format_report_markdown,
)

__all__ = [
  "AlignedPair",
  "AlignmentResult",
  "ComparisonLogger",
  "ComparisonReport",
  "ControlMetrics",
  "FrameData",
  "LogAligner",
  "ModelMetrics",
  "TrajectoryMetrics",
  "compute_all_metrics",
  "format_report_markdown",
  "merge_aligned_logs",
  "validate_alignment",
]
