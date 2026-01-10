"""
Parquet schema definitions for algorithm test scenarios.

This module defines the schema for storing and loading test scenarios
in Parquet format, enabling efficient storage and fast loading of
large scenario datasets.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Schema version for forward compatibility
SCHEMA_VERSION = "1.0.0"


class ScenarioType(Enum):
  """Types of driving scenarios."""
  HIGHWAY_STRAIGHT = "highway_straight"
  HIGHWAY_CURVE = "highway_curve"
  HIGHWAY_LANE_CHANGE = "highway_lane_change"
  URBAN_STRAIGHT = "urban_straight"
  URBAN_INTERSECTION = "urban_intersection"
  LOW_SPEED_MANEUVER = "low_speed_maneuver"
  EMERGENCY_STOP = "emergency_stop"
  CUT_IN = "cut_in"
  CUT_OUT = "cut_out"
  CUSTOM = "custom"


class DifficultyLevel(Enum):
  """Difficulty levels for scenarios."""
  EASY = "easy"
  MEDIUM = "medium"
  HARD = "hard"
  EXTREME = "extreme"


@dataclass
class ScenarioMetadata:
  """Metadata for a scenario file."""
  name: str
  description: str
  scenario_type: ScenarioType
  difficulty: DifficultyLevel
  duration_s: float
  dt_s: float
  num_steps: int
  schema_version: str = SCHEMA_VERSION

  # Source information
  source: str = "synthetic"  # "synthetic", "route_log", "simulation"
  route_id: Optional[str] = None
  segment_start_s: Optional[float] = None
  segment_end_s: Optional[float] = None

  # Conditions
  weather: str = "clear"
  time_of_day: str = "day"
  road_type: str = "highway"

  def to_dict(self) -> dict:
    """Convert to dictionary for Parquet metadata."""
    return {
      'name': self.name,
      'description': self.description,
      'scenario_type': self.scenario_type.value,
      'difficulty': self.difficulty.value,
      'duration_s': self.duration_s,
      'dt_s': self.dt_s,
      'num_steps': self.num_steps,
      'schema_version': self.schema_version,
      'source': self.source,
      'route_id': self.route_id or '',
      'segment_start_s': self.segment_start_s or 0.0,
      'segment_end_s': self.segment_end_s or 0.0,
      'weather': self.weather,
      'time_of_day': self.time_of_day,
      'road_type': self.road_type,
    }

  @classmethod
  def from_dict(cls, d: dict) -> 'ScenarioMetadata':
    """Create from dictionary."""
    return cls(
      name=d['name'],
      description=d['description'],
      scenario_type=ScenarioType(d['scenario_type']),
      difficulty=DifficultyLevel(d['difficulty']),
      duration_s=float(d['duration_s']),
      dt_s=float(d['dt_s']),
      num_steps=int(d['num_steps']),
      schema_version=d.get('schema_version', SCHEMA_VERSION),
      source=d.get('source', 'synthetic'),
      route_id=d.get('route_id') or None,
      segment_start_s=float(d['segment_start_s']) if d.get('segment_start_s') else None,
      segment_end_s=float(d['segment_end_s']) if d.get('segment_end_s') else None,
      weather=d.get('weather', 'clear'),
      time_of_day=d.get('time_of_day', 'day'),
      road_type=d.get('road_type', 'highway'),
    )


# Column definitions for Parquet schema
# Each tuple: (column_name, data_type, description)
SCENARIO_COLUMNS = [
  # Timestamps
  ('timestamp_ns', 'int64', 'Monotonic timestamp in nanoseconds'),
  ('frame_id', 'int64', 'Frame sequence number'),

  # Vehicle state
  ('v_ego', 'float64', 'Vehicle speed (m/s)'),
  ('a_ego', 'float64', 'Vehicle acceleration (m/s^2)'),
  ('yaw_rate', 'float64', 'Vehicle yaw rate (rad/s)'),
  ('steering_angle_deg', 'float64', 'Current steering angle (degrees)'),
  ('steering_rate_deg', 'float64', 'Steering rate (degrees/s)'),
  ('roll', 'float64', 'Road roll angle (rad)'),

  # Control state
  ('active', 'bool', 'Whether control is active'),
  ('steering_pressed', 'bool', 'Driver steering override'),
  ('brake_pressed', 'bool', 'Brake pedal pressed'),
  ('gas_pressed', 'bool', 'Gas pedal pressed'),

  # Targets (lateral)
  ('desired_curvature', 'float64', 'Target path curvature (1/m)'),
  ('curvature_rate', 'float64', 'Curvature rate of change (1/m/s)'),
  ('steer_limited_by_safety', 'bool', 'Safety system limiting steer'),
  ('curvature_limited', 'bool', 'Curvature limit active'),

  # Targets (longitudinal)
  ('a_target', 'float64', 'Target acceleration (m/s^2)'),
  ('v_target', 'float64', 'Target velocity (m/s)'),
  ('should_stop', 'bool', 'Vehicle should come to stop'),
  ('cruise_standstill', 'bool', 'Cruise in standstill mode'),

  # Lead vehicle (if present)
  ('lead_present', 'bool', 'Lead vehicle detected'),
  ('lead_d_rel', 'float64', 'Lead relative distance (m)'),
  ('lead_v_rel', 'float64', 'Lead relative velocity (m/s)'),
  ('lead_a_rel', 'float64', 'Lead relative acceleration (m/s^2)'),

  # Ground truth outputs (for validation)
  ('gt_steer_cmd', 'float64', 'Ground truth steering command'),
  ('gt_accel_cmd', 'float64', 'Ground truth acceleration command'),
  ('gt_trajectory_x', 'float64', 'Ground truth trajectory X (m)'),
  ('gt_trajectory_y', 'float64', 'Ground truth trajectory Y (m)'),
]

# Required columns (must be present in all scenarios)
REQUIRED_COLUMNS = [
  'timestamp_ns',
  'v_ego',
  'a_ego',
  'active',
]

# Optional columns (may be missing, filled with defaults)
OPTIONAL_COLUMN_DEFAULTS = {
  'frame_id': 0,
  'yaw_rate': 0.0,
  'steering_angle_deg': 0.0,
  'steering_rate_deg': 0.0,
  'roll': 0.0,
  'steering_pressed': False,
  'brake_pressed': False,
  'gas_pressed': False,
  'desired_curvature': 0.0,
  'curvature_rate': 0.0,
  'steer_limited_by_safety': False,
  'curvature_limited': False,
  'a_target': 0.0,
  'v_target': 0.0,
  'should_stop': False,
  'cruise_standstill': False,
  'lead_present': False,
  'lead_d_rel': 100.0,
  'lead_v_rel': 0.0,
  'lead_a_rel': 0.0,
  'gt_steer_cmd': 0.0,
  'gt_accel_cmd': 0.0,
  'gt_trajectory_x': 0.0,
  'gt_trajectory_y': 0.0,
}


def get_pyarrow_schema():
  """
  Get PyArrow schema for scenario Parquet files.

  Returns:
    pyarrow.Schema object
  """
  try:
    import pyarrow as pa
  except ImportError:
    raise ImportError("pyarrow is required for Parquet support. Install with: pip install pyarrow")

  type_map = {
    'int64': pa.int64(),
    'float64': pa.float64(),
    'bool': pa.bool_(),
    'string': pa.string(),
  }

  fields = []
  for col_name, col_type, col_desc in SCENARIO_COLUMNS:
    pa_type = type_map.get(col_type, pa.string())
    field = pa.field(col_name, pa_type, metadata={'description': col_desc})
    fields.append(field)

  return pa.schema(fields)


def validate_schema(df_columns: list[str]) -> tuple[bool, list[str]]:
  """
  Validate that a DataFrame has required columns.

  Args:
    df_columns: List of column names in the DataFrame

  Returns:
    Tuple of (is_valid, list of missing required columns)
  """
  missing = [col for col in REQUIRED_COLUMNS if col not in df_columns]
  return len(missing) == 0, missing
