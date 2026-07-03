"""Divergence metrics for shadow device comparison.

Computes metrics to quantify differences between shadow and production
device outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openpilot.tools.shadow.align import AlignedPair, AlignmentResult


@dataclass
class ControlMetrics:
  """Metrics for control command divergence."""

  steer_rmse: float  # Root mean squared error
  steer_mae: float  # Mean absolute error
  steer_max_error: float  # Maximum error
  accel_rmse: float
  accel_mae: float
  accel_max_error: float
  n_samples: int


@dataclass
class TrajectoryMetrics:
  """Metrics for trajectory divergence."""

  path_rmse: float  # Lateral path error (meters)
  path_mae: float
  path_max_error: float
  speed_rmse: float  # Speed error (m/s)
  speed_mae: float
  speed_max_error: float
  n_samples: int


@dataclass
class ModelMetrics:
  """Metrics for model output divergence."""

  curvature_rmse: float
  curvature_mae: float
  curvature_correlation: float  # Pearson correlation
  n_samples: int


@dataclass
class ComparisonReport:
  """Complete comparison report."""

  control_metrics: ControlMetrics
  trajectory_metrics: TrajectoryMetrics | None
  model_metrics: ModelMetrics | None
  alignment_quality: float
  n_aligned_pairs: int
  n_shadow_only: int
  n_production_only: int
  alignment_method: str


def compute_control_metrics(pairs: list[AlignedPair]) -> ControlMetrics:
  """Compute control command divergence metrics.

  Args:
    pairs: List of aligned frame pairs

  Returns:
    ControlMetrics with divergence statistics
  """
  steer_errors: list[float] = []
  accel_errors: list[float] = []

  for pair in pairs:
    shadow_controls = pair.shadow_frame.controls
    prod_controls = pair.production_frame.controls

    # Get steer values (try different keys)
    shadow_steer = shadow_controls.get("steer_torque") or shadow_controls.get("steer", 0.0)
    prod_steer = prod_controls.get("steer_torque") or prod_controls.get("steer", 0.0)

    if shadow_steer is not None and prod_steer is not None:
      steer_errors.append(float(shadow_steer) - float(prod_steer))

    # Get accel values
    shadow_accel = shadow_controls.get("accel", 0.0)
    prod_accel = prod_controls.get("accel", 0.0)

    if shadow_accel is not None and prod_accel is not None:
      accel_errors.append(float(shadow_accel) - float(prod_accel))

  steer_arr = np.array(steer_errors) if steer_errors else np.array([0.0])
  accel_arr = np.array(accel_errors) if accel_errors else np.array([0.0])

  return ControlMetrics(
    steer_rmse=float(np.sqrt(np.mean(steer_arr**2))),
    steer_mae=float(np.mean(np.abs(steer_arr))),
    steer_max_error=float(np.max(np.abs(steer_arr))),
    accel_rmse=float(np.sqrt(np.mean(accel_arr**2))),
    accel_mae=float(np.mean(np.abs(accel_arr))),
    accel_max_error=float(np.max(np.abs(accel_arr))),
    n_samples=len(pairs),
  )


def compute_trajectory_metrics(pairs: list[AlignedPair]) -> TrajectoryMetrics | None:
  """Compute trajectory divergence metrics.

  Args:
    pairs: List of aligned frame pairs

  Returns:
    TrajectoryMetrics or None if no trajectory data
  """
  path_errors: list[float] = []
  speed_errors: list[float] = []

  for pair in pairs:
    shadow_traj = pair.shadow_frame.trajectory
    prod_traj = pair.production_frame.trajectory

    if not shadow_traj or not prod_traj:
      continue

    # Compare Y coordinates (lateral deviation)
    shadow_y = shadow_traj.get("y", [])
    prod_y = prod_traj.get("y", [])

    if shadow_y and prod_y:
      # Compare at common indices
      min_len = min(len(shadow_y), len(prod_y))
      for i in range(min_len):
        path_errors.append(float(shadow_y[i]) - float(prod_y[i]))

    # Compare velocities
    shadow_v = shadow_traj.get("v", [])
    prod_v = prod_traj.get("v", [])

    if shadow_v and prod_v:
      min_len = min(len(shadow_v), len(prod_v))
      for i in range(min_len):
        speed_errors.append(float(shadow_v[i]) - float(prod_v[i]))

  if not path_errors and not speed_errors:
    return None

  path_arr = np.array(path_errors) if path_errors else np.array([0.0])
  speed_arr = np.array(speed_errors) if speed_errors else np.array([0.0])

  return TrajectoryMetrics(
    path_rmse=float(np.sqrt(np.mean(path_arr**2))),
    path_mae=float(np.mean(np.abs(path_arr))),
    path_max_error=float(np.max(np.abs(path_arr))),
    speed_rmse=float(np.sqrt(np.mean(speed_arr**2))),
    speed_mae=float(np.mean(np.abs(speed_arr))),
    speed_max_error=float(np.max(np.abs(speed_arr))),
    n_samples=len(path_errors),
  )


def compute_model_metrics(pairs: list[AlignedPair]) -> ModelMetrics | None:
  """Compute model output divergence metrics.

  Args:
    pairs: List of aligned frame pairs

  Returns:
    ModelMetrics or None if no model data
  """
  shadow_curvatures: list[float] = []
  prod_curvatures: list[float] = []

  for pair in pairs:
    shadow_model = pair.shadow_frame.model_outputs
    prod_model = pair.production_frame.model_outputs

    if not shadow_model or not prod_model:
      continue

    shadow_curv = shadow_model.get("desired_curvature")
    prod_curv = prod_model.get("desired_curvature")

    if shadow_curv is not None and prod_curv is not None:
      shadow_curvatures.append(float(shadow_curv))
      prod_curvatures.append(float(prod_curv))

  if not shadow_curvatures:
    return None

  shadow_arr = np.array(shadow_curvatures)
  prod_arr = np.array(prod_curvatures)
  errors = shadow_arr - prod_arr

  # Compute correlation (handle zero variance case)
  if len(shadow_arr) > 1:
    # Check for zero variance (constant arrays)
    shadow_std = np.std(shadow_arr)
    prod_std = np.std(prod_arr)
    if shadow_std < 1e-10 or prod_std < 1e-10:
      # Can't compute correlation with constant data
      correlation = 0.0
    else:
      correlation = float(np.corrcoef(shadow_arr, prod_arr)[0, 1])
      if np.isnan(correlation):
        correlation = 0.0
  else:
    correlation = 0.0

  return ModelMetrics(
    curvature_rmse=float(np.sqrt(np.mean(errors**2))),
    curvature_mae=float(np.mean(np.abs(errors))),
    curvature_correlation=correlation,
    n_samples=len(shadow_curvatures),
  )


def compute_all_metrics(result: AlignmentResult) -> ComparisonReport:
  """Compute all comparison metrics from alignment result.

  Args:
    result: AlignmentResult from log alignment

  Returns:
    ComparisonReport with all metrics
  """
  return ComparisonReport(
    control_metrics=compute_control_metrics(result.pairs),
    trajectory_metrics=compute_trajectory_metrics(result.pairs),
    model_metrics=compute_model_metrics(result.pairs),
    alignment_quality=result.alignment_quality,
    n_aligned_pairs=len(result.pairs),
    n_shadow_only=len(result.shadow_only),
    n_production_only=len(result.production_only),
    alignment_method=result.method,
  )


def format_report_markdown(report: ComparisonReport) -> str:
  """Format comparison report as markdown.

  Args:
    report: ComparisonReport to format

  Returns:
    Markdown formatted string
  """
  lines = [
    "# Shadow Device Comparison Report",
    "",
    "## Summary",
    "",
    f"- **Aligned Pairs:** {report.n_aligned_pairs}",
    f"- **Shadow Only:** {report.n_shadow_only}",
    f"- **Production Only:** {report.n_production_only}",
    f"- **Alignment Quality:** {report.alignment_quality:.2%}",
    f"- **Alignment Method:** {report.alignment_method}",
    "",
    "## Control Metrics",
    "",
    "| Metric | Steer | Accel |",
    "|--------|-------|-------|",
    f"| RMSE | {report.control_metrics.steer_rmse:.4f} | {report.control_metrics.accel_rmse:.4f} |",
    f"| MAE | {report.control_metrics.steer_mae:.4f} | {report.control_metrics.accel_mae:.4f} |",
    f"| Max Error | {report.control_metrics.steer_max_error:.4f} | {report.control_metrics.accel_max_error:.4f} |",
    "",
  ]

  if report.trajectory_metrics:
    tm = report.trajectory_metrics
    lines.extend(
      [
        "## Trajectory Metrics",
        "",
        "| Metric | Path (m) | Speed (m/s) |",
        "|--------|----------|-------------|",
        f"| RMSE | {tm.path_rmse:.4f} | {tm.speed_rmse:.4f} |",
        f"| MAE | {tm.path_mae:.4f} | {tm.speed_mae:.4f} |",
        f"| Max Error | {tm.path_max_error:.4f} | {tm.speed_max_error:.4f} |",
        "",
      ]
    )

  if report.model_metrics:
    mm = report.model_metrics
    lines.extend(
      [
        "## Model Output Metrics",
        "",
        f"- **Curvature RMSE:** {mm.curvature_rmse:.6f}",
        f"- **Curvature MAE:** {mm.curvature_mae:.6f}",
        f"- **Curvature Correlation:** {mm.curvature_correlation:.4f}",
        "",
      ]
    )

  return "\n".join(lines)


def compute_time_series(
  pairs: list[AlignedPair],
  field: str,
) -> dict[str, list[float]]:
  """Extract time series data for a specific field.

  Args:
    pairs: List of aligned frame pairs
    field: Field to extract ("steer", "accel", "curvature")

  Returns:
    Dictionary with "time", "shadow", "production", "error" arrays
  """
  times: list[float] = []
  shadow_vals: list[float] = []
  prod_vals: list[float] = []
  errors: list[float] = []

  for pair in pairs:
    times.append(pair.shadow_frame.timestamp_mono)

    # Get values based on field
    if field == "steer":
      shadow_val = pair.shadow_frame.controls.get("steer_torque") or pair.shadow_frame.controls.get("steer", 0.0)
      prod_val = pair.production_frame.controls.get("steer_torque") or pair.production_frame.controls.get("steer", 0.0)
    elif field == "accel":
      shadow_val = pair.shadow_frame.controls.get("accel", 0.0)
      prod_val = pair.production_frame.controls.get("accel", 0.0)
    elif field == "curvature":
      shadow_val = pair.shadow_frame.model_outputs.get("desired_curvature", 0.0)
      prod_val = pair.production_frame.model_outputs.get("desired_curvature", 0.0)
    else:
      shadow_val = 0.0
      prod_val = 0.0

    shadow_vals.append(float(shadow_val or 0.0))
    prod_vals.append(float(prod_val or 0.0))
    errors.append(shadow_vals[-1] - prod_vals[-1])

  return {
    "time": times,
    "shadow": shadow_vals,
    "production": prod_vals,
    "error": errors,
  }
