"""
Scenario loading, saving, and validation for algorithm test harness.

This module provides utilities for:
- Loading scenarios from Parquet files
- Saving scenarios to Parquet files
- Validating scenario data
- Converting between DataFrame and Scenario objects
"""

import json
from pathlib import Path
from typing import Optional, Union
import numpy as np

from openpilot.selfdrive.controls.lib.tests.algorithm_harness.interface import (
  AlgorithmState,
  LateralAlgorithmState,
  LongitudinalAlgorithmState,
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.runner import Scenario
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenario_schema import (
  ScenarioMetadata,
  ScenarioType,
  DifficultyLevel,
  SCHEMA_VERSION,
  REQUIRED_COLUMNS,
  OPTIONAL_COLUMN_DEFAULTS,
  validate_schema,
  get_pyarrow_schema,
)


class ScenarioValidationError(Exception):
  """Raised when scenario validation fails."""
  pass


def load_scenario(path: Union[str, Path], scenario_class: str = "lateral") -> Scenario:
  """
  Load a scenario from a Parquet file.

  Args:
    path: Path to Parquet file
    scenario_class: Type of scenario ("lateral", "longitudinal", "base")

  Returns:
    Scenario object with states and optional ground truth

  Raises:
    ScenarioValidationError: If scenario is invalid
    FileNotFoundError: If file doesn't exist
  """
  try:
    import pandas as pd
  except ImportError:
    raise ImportError("pandas is required for scenario loading. Install with: pip install pandas")

  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(f"Scenario file not found: {path}")

  # Load Parquet file
  df = pd.read_parquet(path)

  # Validate schema
  is_valid, missing = validate_schema(list(df.columns))
  if not is_valid:
    raise ScenarioValidationError(f"Missing required columns: {missing}")

  # Fill missing optional columns with defaults
  for col, default in OPTIONAL_COLUMN_DEFAULTS.items():
    if col not in df.columns:
      df[col] = default

  # Load metadata from Parquet metadata
  try:
    import pyarrow.parquet as pq
    parquet_file = pq.read_table(path)
    metadata_json = parquet_file.schema.metadata.get(b'scenario_metadata', b'{}').decode()
    metadata_dict = json.loads(metadata_json)
    metadata = ScenarioMetadata.from_dict(metadata_dict) if metadata_dict else None
  except Exception:
    metadata = None

  # Convert DataFrame to states
  states = _dataframe_to_states(df, scenario_class)

  # Extract ground truth if present
  ground_truth = None
  if scenario_class == "lateral" and 'gt_steer_cmd' in df.columns:
    ground_truth = df['gt_steer_cmd'].tolist()
  elif scenario_class == "longitudinal" and 'gt_accel_cmd' in df.columns:
    ground_truth = df['gt_accel_cmd'].tolist()

  name = metadata.name if metadata else path.stem
  description = metadata.description if metadata else ""

  return Scenario(
    name=name,
    description=description,
    states=states,
    ground_truth=ground_truth,
    metadata=metadata.to_dict() if metadata else {'source_file': str(path)},
  )


def save_scenario(
  scenario: Scenario,
  path: Union[str, Path],
  metadata: Optional[ScenarioMetadata] = None,
) -> None:
  """
  Save a scenario to a Parquet file.

  Args:
    scenario: Scenario object to save
    path: Output path for Parquet file
    metadata: Optional metadata for the scenario
  """
  try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
  except ImportError:
    raise ImportError("pandas and pyarrow required. Install with: pip install pandas pyarrow")

  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)

  # Convert states to DataFrame
  df = _states_to_dataframe(scenario.states)

  # Add ground truth if present
  if scenario.ground_truth is not None:
    # Determine which column based on state type
    if len(scenario.states) > 0:
      if isinstance(scenario.states[0], LateralAlgorithmState):
        df['gt_steer_cmd'] = scenario.ground_truth
      elif isinstance(scenario.states[0], LongitudinalAlgorithmState):
        df['gt_accel_cmd'] = scenario.ground_truth

  # Create metadata
  if metadata is None:
    dt_s = 0.01
    if len(scenario.states) > 1:
      dt_ns = scenario.states[1].timestamp_ns - scenario.states[0].timestamp_ns
      dt_s = dt_ns / 1e9

    metadata = ScenarioMetadata(
      name=scenario.name,
      description=scenario.description,
      scenario_type=ScenarioType.CUSTOM,
      difficulty=DifficultyLevel.MEDIUM,
      duration_s=len(scenario.states) * dt_s,
      dt_s=dt_s,
      num_steps=len(scenario.states),
    )

  # Convert to PyArrow table with metadata
  table = pa.Table.from_pandas(df)
  metadata_json = json.dumps(metadata.to_dict())
  existing_metadata = table.schema.metadata or {}
  new_metadata = {**existing_metadata, b'scenario_metadata': metadata_json.encode()}
  table = table.replace_schema_metadata(new_metadata)

  # Write Parquet file
  pq.write_table(table, path)


def _dataframe_to_states(df, scenario_class: str) -> list:
  """Convert DataFrame rows to state objects."""
  states = []

  for _, row in df.iterrows():
    if scenario_class == "lateral":
      state = LateralAlgorithmState(
        timestamp_ns=int(row['timestamp_ns']),
        v_ego=float(row['v_ego']),
        a_ego=float(row['a_ego']),
        active=bool(row['active']),
        steering_angle_deg=float(row.get('steering_angle_deg', 0.0)),
        steering_rate_deg=float(row.get('steering_rate_deg', 0.0)),
        yaw_rate=float(row.get('yaw_rate', 0.0)),
        desired_curvature=float(row.get('desired_curvature', 0.0)),
        roll=float(row.get('roll', 0.0)),
        steering_pressed=bool(row.get('steering_pressed', False)),
        steer_limited_by_safety=bool(row.get('steer_limited_by_safety', False)),
        curvature_limited=bool(row.get('curvature_limited', False)),
      )
    elif scenario_class == "longitudinal":
      state = LongitudinalAlgorithmState(
        timestamp_ns=int(row['timestamp_ns']),
        v_ego=float(row['v_ego']),
        a_ego=float(row['a_ego']),
        active=bool(row['active']),
        a_target=float(row.get('a_target', 0.0)),
        should_stop=bool(row.get('should_stop', False)),
        brake_pressed=bool(row.get('brake_pressed', False)),
        cruise_standstill=bool(row.get('cruise_standstill', False)),
      )
    else:
      state = AlgorithmState(
        timestamp_ns=int(row['timestamp_ns']),
        v_ego=float(row['v_ego']),
        a_ego=float(row['a_ego']),
        active=bool(row['active']),
      )

    states.append(state)

  return states


def _states_to_dataframe(states: list):
  """Convert state objects to DataFrame."""
  try:
    import pandas as pd
  except ImportError:
    raise ImportError("pandas required")

  rows = []
  for state in states:
    row = {
      'timestamp_ns': state.timestamp_ns,
      'v_ego': state.v_ego,
      'a_ego': state.a_ego,
      'active': state.active,
    }

    if isinstance(state, LateralAlgorithmState):
      row.update({
        'steering_angle_deg': state.steering_angle_deg,
        'steering_rate_deg': state.steering_rate_deg,
        'yaw_rate': state.yaw_rate,
        'desired_curvature': state.desired_curvature,
        'roll': state.roll,
        'steering_pressed': state.steering_pressed,
        'steer_limited_by_safety': state.steer_limited_by_safety,
        'curvature_limited': state.curvature_limited,
      })
    elif isinstance(state, LongitudinalAlgorithmState):
      row.update({
        'a_target': state.a_target,
        'should_stop': state.should_stop,
        'brake_pressed': state.brake_pressed,
        'cruise_standstill': state.cruise_standstill,
      })

    rows.append(row)

  return pd.DataFrame(rows)


def validate_scenario(scenario: Scenario) -> tuple[bool, list[str]]:
  """
  Validate a scenario for completeness and consistency.

  Args:
    scenario: Scenario to validate

  Returns:
    Tuple of (is_valid, list of error messages)
  """
  errors = []

  # Check basic requirements
  if len(scenario.states) == 0:
    errors.append("Scenario has no states")
    return False, errors

  # Check timestamps are monotonic
  timestamps = [s.timestamp_ns for s in scenario.states]
  if timestamps != sorted(timestamps):
    errors.append("Timestamps are not monotonically increasing")

  # Check for duplicate timestamps
  if len(timestamps) != len(set(timestamps)):
    errors.append("Duplicate timestamps detected")

  # Check ground truth length matches states
  if scenario.ground_truth is not None:
    if len(scenario.ground_truth) != len(scenario.states):
      errors.append(f"Ground truth length ({len(scenario.ground_truth)}) "
                    f"doesn't match states length ({len(scenario.states)})")

  # Check v_ego is reasonable
  v_egos = [s.v_ego for s in scenario.states]
  if any(v < 0 for v in v_egos):
    errors.append("Negative v_ego values detected")
  if any(v > 100 for v in v_egos):  # > 100 m/s = 360 km/h
    errors.append("Unreasonably high v_ego values detected (> 100 m/s)")

  # Check for NaN values
  for i, state in enumerate(scenario.states):
    if np.isnan(state.v_ego) or np.isnan(state.a_ego):
      errors.append(f"NaN values in state at index {i}")
      break

  return len(errors) == 0, errors


def list_scenarios(directory: Union[str, Path]) -> list[dict]:
  """
  List all scenarios in a directory.

  Args:
    directory: Directory to search for Parquet files

  Returns:
    List of dictionaries with scenario info
  """
  directory = Path(directory)
  if not directory.exists():
    return []

  scenarios = []
  for path in directory.glob("**/*.parquet"):
    try:
      import pyarrow.parquet as pq
      parquet_file = pq.read_table(path)
      metadata_json = parquet_file.schema.metadata.get(b'scenario_metadata', b'{}').decode()
      metadata = json.loads(metadata_json)

      scenarios.append({
        'path': str(path),
        'name': metadata.get('name', path.stem),
        'type': metadata.get('scenario_type', 'unknown'),
        'difficulty': metadata.get('difficulty', 'unknown'),
        'duration_s': metadata.get('duration_s', 0),
        'num_steps': metadata.get('num_steps', 0),
      })
    except Exception as e:
      scenarios.append({
        'path': str(path),
        'name': path.stem,
        'error': str(e),
      })

  return scenarios
