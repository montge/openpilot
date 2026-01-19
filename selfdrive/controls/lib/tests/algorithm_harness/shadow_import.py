"""
Shadow device log import for the algorithm test harness.

Converts shadow device logs (FrameData) to algorithm harness Scenarios,
enabling replay of real-world captured data through the test harness.

Usage:
  from openpilot.selfdrive.controls.lib.tests.algorithm_harness.shadow_import import (
    import_shadow_log,
    import_shadow_segment,
  )

  # Load a segment directory
  scenarios = import_shadow_segment("/data/shadow_logs/route_001")

  # Run through harness
  runner = ScenarioRunner()
  for scenario in scenarios:
    result = runner.run(my_algorithm, scenario)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from openpilot.selfdrive.controls.lib.tests.algorithm_harness.interface import (
  LateralAlgorithmState,
  LongitudinalAlgorithmState,
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.runner import Scenario
from openpilot.tools.shadow.comparison_logger import ComparisonLogger, FrameData

if TYPE_CHECKING:
  pass


def frame_to_lateral_state(frame: FrameData) -> LateralAlgorithmState:
  """Convert a shadow FrameData to LateralAlgorithmState.

  Args:
    frame: Shadow device frame data

  Returns:
    LateralAlgorithmState for algorithm harness
  """
  state = frame.state
  controls = frame.controls
  model_outputs = frame.model_outputs

  return LateralAlgorithmState(
    timestamp_ns=int(frame.timestamp_mono * 1e9),
    v_ego=state.get("v_ego", 0.0),
    a_ego=state.get("a_ego", 0.0),
    active=state.get("lat_active", True),
    steering_angle_deg=controls.get("steering_angle_deg", 0.0) or 0.0,
    steering_rate_deg=state.get("steering_rate_deg", 0.0),
    yaw_rate=state.get("yaw_rate", 0.0),
    desired_curvature=model_outputs.get("desired_curvature", 0.0) or 0.0,
    roll=state.get("roll", 0.0),
    steering_pressed=state.get("steering_pressed", False),
    steer_limited_by_safety=state.get("steer_limited_by_safety", False),
    curvature_limited=state.get("curvature_limited", False),
  )


def frame_to_longitudinal_state(frame: FrameData) -> LongitudinalAlgorithmState:
  """Convert a shadow FrameData to LongitudinalAlgorithmState.

  Args:
    frame: Shadow device frame data

  Returns:
    LongitudinalAlgorithmState for algorithm harness
  """
  state = frame.state

  return LongitudinalAlgorithmState(
    timestamp_ns=int(frame.timestamp_mono * 1e9),
    v_ego=state.get("v_ego", 0.0),
    a_ego=state.get("a_ego", 0.0),
    active=state.get("long_active", True),
    a_target=state.get("a_target", 0.0),
    should_stop=state.get("should_stop", False),
    brake_pressed=state.get("brake_pressed", False),
    cruise_standstill=state.get("cruise_standstill", False),
    accel_limits=tuple(state.get("accel_limits", (-3.5, 2.0))),
  )


def import_shadow_log(
  frames: list[FrameData],
  name: str = "shadow_log",
  description: str | None = None,
  mode: str = "lateral",
) -> Scenario:
  """Convert shadow device frames to a harness Scenario.

  Args:
    frames: List of shadow device FrameData objects
    name: Scenario name
    description: Optional scenario description
    mode: "lateral" or "longitudinal" - determines state type

  Returns:
    Scenario object for algorithm harness
  """
  if mode == "lateral":
    states = [frame_to_lateral_state(f) for f in frames]
    # Ground truth is the actual steering command from shadow device
    ground_truth = [f.controls.get("steer_torque", 0.0) or 0.0 for f in frames]
  elif mode == "longitudinal":
    states = [frame_to_longitudinal_state(f) for f in frames]
    # Ground truth is the actual acceleration command
    ground_truth = [f.controls.get("accel", 0.0) or 0.0 for f in frames]
  else:
    raise ValueError(f"Unknown mode: {mode}. Use 'lateral' or 'longitudinal'")

  return Scenario(
    name=name,
    description=description or f"Shadow device log ({len(frames)} frames)",
    states=states,
    ground_truth=ground_truth,
    metadata={
      "source": "shadow_device",
      "mode": mode,
      "frame_count": len(frames),
      "first_frame_id": frames[0].frame_id if frames else None,
      "last_frame_id": frames[-1].frame_id if frames else None,
      "duration_s": (frames[-1].timestamp_mono - frames[0].timestamp_mono) if len(frames) > 1 else 0,
    },
  )


def import_shadow_segment(
  segment_dir: str | Path,
  name: str | None = None,
  mode: str = "lateral",
  subsample: int = 1,
  max_frames: int | None = None,
) -> Scenario:
  """Import a shadow device segment directory as a Scenario.

  Args:
    segment_dir: Path to segment directory containing log files
    name: Optional scenario name (defaults to directory name)
    mode: "lateral" or "longitudinal"
    subsample: Take every Nth frame (1 = all frames)
    max_frames: Maximum number of frames to import

  Returns:
    Scenario object for algorithm harness
  """
  segment_dir = Path(segment_dir)
  frames = ComparisonLogger.load_segment(segment_dir)

  if not frames:
    raise ValueError(f"No frames found in {segment_dir}")

  # Subsample if requested
  if subsample > 1:
    frames = frames[::subsample]

  # Limit frame count if requested
  if max_frames is not None and len(frames) > max_frames:
    frames = frames[:max_frames]

  scenario_name = name or segment_dir.name
  return import_shadow_log(frames, name=scenario_name, mode=mode)


def import_shadow_segments(
  root_dir: str | Path,
  mode: str = "lateral",
  subsample: int = 1,
  max_frames_per_segment: int | None = None,
) -> list[Scenario]:
  """Import all shadow segments from a directory.

  Args:
    root_dir: Root directory containing segment subdirectories
    mode: "lateral" or "longitudinal"
    subsample: Take every Nth frame (1 = all frames)
    max_frames_per_segment: Maximum frames per segment

  Returns:
    List of Scenario objects
  """
  root_dir = Path(root_dir)
  scenarios: list[Scenario] = []

  for segment_dir in sorted(root_dir.iterdir()):
    if segment_dir.is_dir():
      try:
        scenario = import_shadow_segment(
          segment_dir,
          mode=mode,
          subsample=subsample,
          max_frames=max_frames_per_segment,
        )
        scenarios.append(scenario)
      except ValueError:
        # Skip directories without valid log files
        continue

  return scenarios


def compare_shadow_to_harness(
  frames: list[FrameData],
  harness_outputs: list[float],
  mode: str = "lateral",
) -> dict:
  """Compare shadow device outputs to algorithm harness outputs.

  Args:
    frames: Original shadow device frames
    harness_outputs: Outputs from running algorithm through harness
    mode: "lateral" or "longitudinal"

  Returns:
    Dictionary with comparison metrics
  """
  import numpy as np

  if len(frames) != len(harness_outputs):
    raise ValueError(
      f"Frame count ({len(frames)}) doesn't match output count ({len(harness_outputs)})"
    )

  # Extract shadow device outputs
  if mode == "lateral":
    shadow_outputs = [f.controls.get("steer_torque", 0.0) or 0.0 for f in frames]
  else:
    shadow_outputs = [f.controls.get("accel", 0.0) or 0.0 for f in frames]

  shadow_arr = np.array(shadow_outputs)
  harness_arr = np.array(harness_outputs)
  errors = harness_arr - shadow_arr

  # Compute metrics
  metrics = {
    "n_samples": len(frames),
    "shadow_mean": float(np.mean(shadow_arr)),
    "shadow_std": float(np.std(shadow_arr)),
    "harness_mean": float(np.mean(harness_arr)),
    "harness_std": float(np.std(harness_arr)),
    "error_rmse": float(np.sqrt(np.mean(errors**2))),
    "error_mae": float(np.mean(np.abs(errors))),
    "error_max": float(np.max(np.abs(errors))),
    "error_mean": float(np.mean(errors)),
    "error_std": float(np.std(errors)),
  }

  # Correlation if sufficient variance
  if np.std(shadow_arr) > 1e-10 and np.std(harness_arr) > 1e-10:
    corr = np.corrcoef(shadow_arr, harness_arr)[0, 1]
    if not np.isnan(corr):
      metrics["correlation"] = float(corr)

  return metrics


def format_shadow_comparison_report(metrics: dict, algorithm_name: str = "Algorithm") -> str:
  """Format shadow comparison metrics as a report.

  Args:
    metrics: Metrics dictionary from compare_shadow_to_harness
    algorithm_name: Name of the tested algorithm

  Returns:
    Formatted markdown report string
  """
  lines = [
    f"# Shadow Device vs {algorithm_name} Comparison",
    "",
    "## Summary",
    "",
    f"- **Samples:** {metrics['n_samples']}",
    "",
    "## Output Statistics",
    "",
    "| Source | Mean | Std |",
    "|--------|------|-----|",
    f"| Shadow Device | {metrics['shadow_mean']:.4f} | {metrics['shadow_std']:.4f} |",
    f"| {algorithm_name} | {metrics['harness_mean']:.4f} | {metrics['harness_std']:.4f} |",
    "",
    "## Error Metrics",
    "",
    f"- **RMSE:** {metrics['error_rmse']:.4f}",
    f"- **MAE:** {metrics['error_mae']:.4f}",
    f"- **Max Error:** {metrics['error_max']:.4f}",
    f"- **Mean Error (Bias):** {metrics['error_mean']:.4f}",
    f"- **Error Std:** {metrics['error_std']:.4f}",
  ]

  if "correlation" in metrics:
    lines.extend([
      "",
      "## Correlation",
      "",
      f"- **Pearson Correlation:** {metrics['correlation']:.4f}",
    ])

  return "\n".join(lines)
