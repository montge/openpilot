"""Shadow log adapter for algorithm harness.

Converts shadow device comparison logs into algorithm harness scenarios
for replay and analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
  import pandas as pd

  PANDAS_AVAILABLE = True
except ImportError:
  PANDAS_AVAILABLE = False

from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenario_schema import (
  DifficultyLevel,
  ScenarioMetadata,
  ScenarioType,
)
from openpilot.tools.shadow.align import LogAligner
from openpilot.tools.shadow.comparison_logger import ComparisonLogger, FrameData


def shadow_frames_to_dataframe(
  frames: list[FrameData],
  device_id: str = "shadow",
) -> pd.DataFrame:
  """Convert shadow frames to a Pandas DataFrame for harness.

  Args:
    frames: List of FrameData from shadow logger
    device_id: Identifier prefix for columns

  Returns:
    DataFrame compatible with algorithm harness
  """
  if not PANDAS_AVAILABLE:
    raise ImportError("pandas is required for shadow log conversion")

  rows = []
  for frame in frames:
    row: dict[str, Any] = {
      "timestamp_ns": int(frame.timestamp_mono * 1e9),
      "frame_id": frame.frame_id,
      "active": frame.state.get("lat_active", False) or frame.state.get("long_active", False),
    }

    # Vehicle state from controls
    controls = frame.controls
    row["v_ego"] = controls.get("v_ego", 0.0)
    row["a_ego"] = controls.get("a_ego", 0.0)
    row["steering_angle_deg"] = controls.get("steering_angle_deg", 0.0)
    row["yaw_rate"] = controls.get("yaw_rate", 0.0)

    # Control commands
    row["gt_steer_cmd"] = controls.get("steer_torque") or controls.get("steer", 0.0)
    row["gt_accel_cmd"] = controls.get("accel", 0.0)

    # Model outputs
    model = frame.model_outputs
    row["desired_curvature"] = model.get("desired_curvature", 0.0)

    # Trajectory (first point if available)
    traj = frame.trajectory
    if traj.get("x"):
      row["gt_trajectory_x"] = traj["x"][0] if len(traj["x"]) > 0 else 0.0
    else:
      row["gt_trajectory_x"] = 0.0
    if traj.get("y"):
      row["gt_trajectory_y"] = traj["y"][0] if len(traj["y"]) > 0 else 0.0
    else:
      row["gt_trajectory_y"] = 0.0

    rows.append(row)

  return pd.DataFrame(rows)


def create_shadow_scenario(
  shadow_frames: list[FrameData],
  name: str,
  description: str = "Scenario from shadow device logs",
  scenario_type: ScenarioType = ScenarioType.CUSTOM,
  difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
) -> tuple[pd.DataFrame, ScenarioMetadata]:
  """Create an algorithm harness scenario from shadow logs.

  Args:
    shadow_frames: Frames from shadow device
    name: Scenario name
    description: Scenario description
    scenario_type: Type of scenario
    difficulty: Difficulty level

  Returns:
    Tuple of (DataFrame, ScenarioMetadata)
  """
  if not shadow_frames:
    raise ValueError("No frames provided")

  df = shadow_frames_to_dataframe(shadow_frames)

  # Calculate duration
  if len(df) > 1:
    duration_s = (df["timestamp_ns"].iloc[-1] - df["timestamp_ns"].iloc[0]) / 1e9
    dt_s = duration_s / (len(df) - 1)
  else:
    duration_s = 0.0
    dt_s = 0.01

  metadata = ScenarioMetadata(
    name=name,
    description=description,
    scenario_type=scenario_type,
    difficulty=difficulty,
    duration_s=duration_s,
    dt_s=dt_s,
    num_steps=len(df),
    source="shadow_device",
  )

  return df, metadata


def create_comparison_scenario(
  shadow_dir: Path,
  prod_dir: Path,
  name: str,
  align_method: str = "auto",
) -> tuple[pd.DataFrame, ScenarioMetadata, dict[str, Any]]:
  """Create a comparison scenario from aligned shadow and production logs.

  Args:
    shadow_dir: Path to shadow device logs
    prod_dir: Path to production device logs
    name: Scenario name
    align_method: Alignment method ("auto", "gps", "frame_id", "timestamp")

  Returns:
    Tuple of (DataFrame, ScenarioMetadata, comparison_metrics)
  """
  # Load logs
  shadow_frames = ComparisonLogger.load_segment(shadow_dir)
  prod_frames = ComparisonLogger.load_segment(prod_dir)

  # Align
  aligner = LogAligner()
  if align_method == "gps":
    result = aligner.align_by_gps(shadow_frames, prod_frames)
  elif align_method == "frame_id":
    result = aligner.align_by_frame_id(shadow_frames, prod_frames)
  elif align_method == "timestamp":
    result = aligner.align_by_timestamp(shadow_frames, prod_frames)
  else:
    result = aligner.auto_align(shadow_frames, prod_frames)

  # Create dataframe with both shadow and production columns
  if not PANDAS_AVAILABLE:
    raise ImportError("pandas is required for shadow log conversion")

  rows = []
  for pair in result.pairs:
    shadow = pair.shadow_frame
    prod = pair.production_frame

    row: dict[str, Any] = {
      "timestamp_ns": int(shadow.timestamp_mono * 1e9),
      "frame_id": shadow.frame_id,
      "active": shadow.state.get("lat_active", False),
      # Vehicle state (use shadow)
      "v_ego": shadow.controls.get("v_ego", 0.0),
      "a_ego": shadow.controls.get("a_ego", 0.0),
      "steering_angle_deg": shadow.controls.get("steering_angle_deg", 0.0),
      # Shadow commands
      "shadow_steer_cmd": shadow.controls.get("steer_torque") or shadow.controls.get("steer", 0.0),
      "shadow_accel_cmd": shadow.controls.get("accel", 0.0),
      "shadow_curvature": shadow.model_outputs.get("desired_curvature", 0.0),
      # Production commands (ground truth)
      "gt_steer_cmd": prod.controls.get("steer_torque") or prod.controls.get("steer", 0.0),
      "gt_accel_cmd": prod.controls.get("accel", 0.0),
      "desired_curvature": prod.model_outputs.get("desired_curvature", 0.0),
      # Alignment info
      "alignment_quality": pair.alignment_quality,
      "time_offset_ms": pair.time_offset_ms,
    }
    rows.append(row)

  df = pd.DataFrame(rows)

  # Calculate duration
  if len(df) > 1:
    duration_s = (df["timestamp_ns"].iloc[-1] - df["timestamp_ns"].iloc[0]) / 1e9
    dt_s = duration_s / (len(df) - 1)
  else:
    duration_s = 0.0
    dt_s = 0.01

  metadata = ScenarioMetadata(
    name=name,
    description=f"Comparison scenario aligned via {result.method}",
    scenario_type=ScenarioType.CUSTOM,
    difficulty=DifficultyLevel.MEDIUM,
    duration_s=duration_s,
    dt_s=dt_s,
    num_steps=len(df),
    source="shadow_comparison",
  )

  # Compute comparison metrics
  if len(df) > 0:
    steer_errors = df["shadow_steer_cmd"] - df["gt_steer_cmd"]
    accel_errors = df["shadow_accel_cmd"] - df["gt_accel_cmd"]

    comparison_metrics = {
      "alignment_method": result.method,
      "alignment_quality": result.alignment_quality,
      "aligned_pairs": len(result.pairs),
      "shadow_only": len(result.shadow_only),
      "production_only": len(result.production_only),
      "steer_rmse": float(np.sqrt(np.mean(steer_errors**2))),
      "steer_mae": float(np.mean(np.abs(steer_errors))),
      "accel_rmse": float(np.sqrt(np.mean(accel_errors**2))),
      "accel_mae": float(np.mean(np.abs(accel_errors))),
    }
  else:
    comparison_metrics = {
      "alignment_method": result.method,
      "alignment_quality": 0.0,
      "aligned_pairs": 0,
    }

  return df, metadata, comparison_metrics


def save_scenario_parquet(
  df: pd.DataFrame,
  metadata: ScenarioMetadata,
  output_path: Path,
) -> None:
  """Save scenario as Parquet file with metadata.

  Args:
    df: Scenario DataFrame
    metadata: Scenario metadata
    output_path: Path to save Parquet file
  """
  try:
    import pyarrow as pa
    import pyarrow.parquet as pq
  except ImportError as err:
    raise ImportError("pyarrow is required for Parquet export") from err

  table = pa.Table.from_pandas(df)

  # Add metadata
  existing_meta = table.schema.metadata or {}
  new_meta = {
    b"scenario_metadata": str(metadata.to_dict()).encode(),
    **existing_meta,
  }
  table = table.replace_schema_metadata(new_meta)

  pq.write_table(table, output_path)


def load_scenario_parquet(
  input_path: Path,
) -> tuple[pd.DataFrame, ScenarioMetadata]:
  """Load scenario from Parquet file.

  Args:
    input_path: Path to Parquet file

  Returns:
    Tuple of (DataFrame, ScenarioMetadata)
  """
  try:
    import pyarrow.parquet as pq
  except ImportError as err:
    raise ImportError("pyarrow is required for Parquet loading") from err

  import ast

  table = pq.read_table(input_path)
  df = table.to_pandas()

  # Extract metadata
  schema_meta = table.schema.metadata or {}
  meta_bytes = schema_meta.get(b"scenario_metadata", b"{}")
  meta_dict = ast.literal_eval(meta_bytes.decode())
  metadata = ScenarioMetadata.from_dict(meta_dict)

  return df, metadata
