"""
Comprehensive tests for scenario infrastructure.

Tests cover:
- Scenario schema validation
- Scenario generation
- Scenario loading/saving (with Parquet)
- Scenario validation
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.tests.algorithm_harness.interface import (
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
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenario_generator import (
  generate_highway_straight,
  generate_tight_s_curve,
  generate_highway_lane_change,
  generate_low_speed_maneuver,
  generate_emergency_stop,
  generate_all_seed_scenarios,
)


# ============================================================================
# Schema Tests
# ============================================================================


class TestScenarioMetadata:
  """Tests for ScenarioMetadata."""

  def test_create_metadata(self):
    """Test creating metadata with required fields."""
    metadata = ScenarioMetadata(
      name="test_scenario",
      description="A test scenario",
      scenario_type=ScenarioType.HIGHWAY_STRAIGHT,
      difficulty=DifficultyLevel.EASY,
      duration_s=10.0,
      dt_s=0.01,
      num_steps=1000,
    )

    assert metadata.name == "test_scenario"
    assert metadata.schema_version == SCHEMA_VERSION

  def test_to_dict(self):
    """Test converting metadata to dictionary."""
    metadata = ScenarioMetadata(
      name="test",
      description="desc",
      scenario_type=ScenarioType.HIGHWAY_CURVE,
      difficulty=DifficultyLevel.MEDIUM,
      duration_s=5.0,
      dt_s=0.01,
      num_steps=500,
    )

    d = metadata.to_dict()
    assert d['name'] == "test"
    assert d['scenario_type'] == "highway_curve"
    assert d['difficulty'] == "medium"

  def test_from_dict(self):
    """Test creating metadata from dictionary."""
    d = {
      'name': 'from_dict',
      'description': 'Created from dict',
      'scenario_type': 'urban_straight',
      'difficulty': 'hard',
      'duration_s': 15.0,
      'dt_s': 0.02,
      'num_steps': 750,
    }

    metadata = ScenarioMetadata.from_dict(d)
    assert metadata.name == 'from_dict'
    assert metadata.scenario_type == ScenarioType.URBAN_STRAIGHT
    assert metadata.difficulty == DifficultyLevel.HARD

  def test_round_trip(self):
    """Test metadata survives to_dict/from_dict round trip."""
    original = ScenarioMetadata(
      name="round_trip",
      description="Testing round trip",
      scenario_type=ScenarioType.EMERGENCY_STOP,
      difficulty=DifficultyLevel.EXTREME,
      duration_s=8.0,
      dt_s=0.005,
      num_steps=1600,
      source="route_log",
      route_id="abc123",
      weather="rain",
      time_of_day="night",
    )

    restored = ScenarioMetadata.from_dict(original.to_dict())

    assert restored.name == original.name
    assert restored.scenario_type == original.scenario_type
    assert restored.weather == original.weather
    assert restored.route_id == original.route_id


class TestSchemaValidation:
  """Tests for schema validation."""

  def test_validate_with_required_columns(self):
    """Test validation passes with required columns."""
    columns = REQUIRED_COLUMNS.copy()
    is_valid, missing = validate_schema(columns)
    assert is_valid
    assert len(missing) == 0

  def test_validate_missing_columns(self):
    """Test validation fails with missing columns."""
    columns = ['timestamp_ns', 'v_ego']  # Missing a_ego, active
    is_valid, missing = validate_schema(columns)
    assert not is_valid
    assert 'a_ego' in missing
    assert 'active' in missing

  def test_validate_extra_columns_ok(self):
    """Test validation passes with extra columns."""
    columns = REQUIRED_COLUMNS + ['custom_column', 'another_custom']
    is_valid, missing = validate_schema(columns)
    assert is_valid

  def test_optional_column_defaults(self):
    """Test optional columns have reasonable defaults."""
    for col, default in OPTIONAL_COLUMN_DEFAULTS.items():
      assert col not in REQUIRED_COLUMNS, f"{col} is both required and optional"
      # Check default type is consistent
      if isinstance(default, bool):
        assert default in [True, False]
      elif isinstance(default, float):
        assert not np.isnan(default)


# ============================================================================
# Scenario Generation Tests
# ============================================================================


class TestHighwayStraight:
  """Tests for highway straight scenario generation."""

  def test_generate_basic(self):
    """Test basic generation."""
    scenario, metadata = generate_highway_straight()
    assert len(scenario) > 0
    assert scenario.name == "highway_straight_baseline"

  def test_generate_custom_duration(self):
    """Test custom duration."""
    scenario, metadata = generate_highway_straight(duration_s=10.0, dt_s=0.02)
    assert len(scenario) == 500  # 10 / 0.02
    assert metadata.duration_s == 10.0

  def test_states_are_lateral(self):
    """Test states are LateralAlgorithmState."""
    scenario, _ = generate_highway_straight(duration_s=1.0)
    assert all(isinstance(s, LateralAlgorithmState) for s in scenario.states)

  def test_constant_speed(self):
    """Test speed is constant."""
    scenario, _ = generate_highway_straight(v_ego=25.0)
    speeds = [s.v_ego for s in scenario.states]
    assert all(v == 25.0 for v in speeds)

  def test_has_ground_truth(self):
    """Test ground truth is present."""
    scenario, _ = generate_highway_straight()
    assert scenario.ground_truth is not None
    assert len(scenario.ground_truth) == len(scenario.states)


class TestTightSCurve:
  """Tests for tight S-curve scenario generation."""

  def test_generate_basic(self):
    """Test basic generation."""
    scenario, metadata = generate_tight_s_curve()
    assert len(scenario) > 0
    assert "curve" in scenario.name.lower()

  def test_curvature_varies(self):
    """Test curvature varies throughout scenario."""
    scenario, _ = generate_tight_s_curve()
    curvatures = [s.desired_curvature for s in scenario.states]

    # Should have both positive and negative curvatures (S-curve)
    assert max(curvatures) > 0
    assert min(curvatures) < 0

  def test_yaw_rate_consistent(self):
    """Test yaw rate is consistent with curvature and speed."""
    scenario, _ = generate_tight_s_curve(v_ego=20.0)

    for state in scenario.states:
      expected_yaw = state.v_ego * state.desired_curvature
      assert abs(state.yaw_rate - expected_yaw) < 0.001


class TestHighwayLaneChange:
  """Tests for highway lane change scenario generation."""

  def test_generate_basic(self):
    """Test basic generation."""
    scenario, metadata = generate_highway_lane_change()
    assert len(scenario) > 0
    assert metadata.scenario_type == ScenarioType.HIGHWAY_LANE_CHANGE

  def test_has_straight_sections(self):
    """Test has straight sections before and after lane change."""
    scenario, _ = generate_highway_lane_change(lane_change_time=5.0)

    # First few states should have near-zero curvature
    early_curvatures = [s.desired_curvature for s in scenario.states[:100]]
    assert all(abs(c) < 0.001 for c in early_curvatures)


class TestLowSpeedManeuver:
  """Tests for low-speed maneuver scenario generation."""

  def test_generate_basic(self):
    """Test basic generation."""
    scenario, metadata = generate_low_speed_maneuver()
    assert len(scenario) > 0
    assert metadata.scenario_type == ScenarioType.LOW_SPEED_MANEUVER

  def test_low_speeds(self):
    """Test speeds are low."""
    scenario, _ = generate_low_speed_maneuver(max_v=5.0)
    speeds = [s.v_ego for s in scenario.states]
    assert max(speeds) <= 5.0 + 0.1  # Small tolerance

  def test_tight_curvatures(self):
    """Test has tight curvatures."""
    scenario, _ = generate_low_speed_maneuver(max_curvature=0.05)
    curvatures = [s.desired_curvature for s in scenario.states]

    # Should have curvatures up to max
    assert max(abs(c) for c in curvatures) > 0.01


class TestEmergencyStop:
  """Tests for emergency stop scenario generation."""

  def test_generate_basic(self):
    """Test basic generation."""
    scenario, metadata = generate_emergency_stop()
    assert len(scenario) > 0
    assert metadata.scenario_type == ScenarioType.EMERGENCY_STOP

  def test_states_are_longitudinal(self):
    """Test states are LongitudinalAlgorithmState."""
    scenario, _ = generate_emergency_stop()
    assert all(isinstance(s, LongitudinalAlgorithmState) for s in scenario.states)

  def test_speed_decreases(self):
    """Test speed decreases to zero."""
    scenario, _ = generate_emergency_stop(initial_v=25.0, stop_time=1.0)
    speeds = [s.v_ego for s in scenario.states]

    # Should start at initial speed
    assert speeds[0] == 25.0

    # Should end at or near zero
    assert speeds[-1] < 1.0

  def test_should_stop_flag(self):
    """Test should_stop flag is set correctly."""
    scenario, _ = generate_emergency_stop(stop_time=3.0, duration_s=6.0)

    # Before stop_time
    early_states = [s for s in scenario.states if s.timestamp_ns < 3e9]
    assert all(not s.should_stop for s in early_states)

    # After stop_time
    late_states = [s for s in scenario.states if s.timestamp_ns >= 3e9]
    assert all(s.should_stop for s in late_states)


class TestGenerateAllSeedScenarios:
  """Tests for generate_all_seed_scenarios."""

  def test_generates_all_five(self):
    """Test generates all 5 seed scenarios."""
    scenarios = generate_all_seed_scenarios()
    assert len(scenarios) == 5

  def test_scenario_names(self):
    """Test expected scenario names are present."""
    scenarios = generate_all_seed_scenarios()
    expected_names = [
      'highway_straight',
      'tight_s_curve',
      'highway_lane_change',
      'low_speed_maneuver',
      'emergency_stop',
    ]
    assert set(scenarios.keys()) == set(expected_names)

  def test_all_have_metadata(self):
    """Test all scenarios have metadata."""
    scenarios = generate_all_seed_scenarios()
    for name, (scenario, metadata) in scenarios.items():
      assert scenario is not None
      assert metadata is not None
      assert metadata.name != ""


# ============================================================================
# Scenario Validation Tests
# ============================================================================


class TestScenarioValidation:
  """Tests for scenario validation."""

  def test_valid_scenario(self):
    """Test valid scenario passes validation."""
    from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenarios import validate_scenario

    scenario, _ = generate_highway_straight(duration_s=1.0)
    is_valid, errors = validate_scenario(scenario)
    assert is_valid
    assert len(errors) == 0

  def test_empty_scenario_invalid(self):
    """Test empty scenario fails validation."""
    from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenarios import validate_scenario

    scenario = Scenario(name="empty", states=[])
    is_valid, errors = validate_scenario(scenario)
    assert not is_valid
    assert "no states" in errors[0].lower()

  def test_non_monotonic_timestamps(self):
    """Test non-monotonic timestamps fail validation."""
    from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenarios import validate_scenario
    from openpilot.selfdrive.controls.lib.tests.algorithm_harness.interface import AlgorithmState

    states = [
      AlgorithmState(timestamp_ns=1000, v_ego=10.0, a_ego=0.0),
      AlgorithmState(timestamp_ns=500, v_ego=10.0, a_ego=0.0),  # Out of order
      AlgorithmState(timestamp_ns=2000, v_ego=10.0, a_ego=0.0),
    ]
    scenario = Scenario(name="bad_timestamps", states=states)

    is_valid, errors = validate_scenario(scenario)
    assert not is_valid
    assert any("monotonic" in e.lower() for e in errors)

  def test_mismatched_ground_truth_length(self):
    """Test mismatched ground truth length fails validation."""
    from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenarios import validate_scenario
    from openpilot.selfdrive.controls.lib.tests.algorithm_harness.interface import AlgorithmState

    states = [AlgorithmState(timestamp_ns=i * 1000, v_ego=10.0, a_ego=0.0) for i in range(10)]
    scenario = Scenario(
      name="bad_gt",
      states=states,
      ground_truth=[1.0, 2.0, 3.0],  # Only 3 values for 10 states
    )

    is_valid, errors = validate_scenario(scenario)
    assert not is_valid
    assert any("length" in e.lower() for e in errors)


# ============================================================================
# Parquet I/O Tests (require pandas/pyarrow)
# ============================================================================


class TestParquetIO:
  """Tests for Parquet save/load functionality."""

  @pytest.fixture
  def temp_dir(self):
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
      yield Path(tmpdir)

  def test_save_and_load_lateral(self, temp_dir):
    """Test saving and loading lateral scenario."""
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenarios import save_scenario, load_scenario

    # Generate and save
    original, metadata = generate_highway_straight(duration_s=1.0)
    file_path = temp_dir / "test_lateral.parquet"
    save_scenario(original, file_path, metadata)

    assert file_path.exists()

    # Load and verify
    loaded = load_scenario(file_path, scenario_class="lateral")
    assert loaded.name == original.name
    assert len(loaded.states) == len(original.states)

  def test_save_and_load_longitudinal(self, temp_dir):
    """Test saving and loading longitudinal scenario."""
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenarios import save_scenario, load_scenario

    original, metadata = generate_emergency_stop(duration_s=2.0)
    file_path = temp_dir / "test_long.parquet"
    save_scenario(original, file_path, metadata)

    loaded = load_scenario(file_path, scenario_class="longitudinal")
    assert len(loaded.states) == len(original.states)
    assert all(isinstance(s, LongitudinalAlgorithmState) for s in loaded.states)

  def test_ground_truth_preserved(self, temp_dir):
    """Test ground truth is preserved through save/load."""
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenarios import save_scenario, load_scenario

    original, metadata = generate_tight_s_curve(duration_s=1.0)
    file_path = temp_dir / "test_gt.parquet"
    save_scenario(original, file_path, metadata)

    loaded = load_scenario(file_path, scenario_class="lateral")
    assert loaded.ground_truth is not None
    assert len(loaded.ground_truth) == len(original.ground_truth)

  def test_list_scenarios(self, temp_dir):
    """Test listing scenarios in directory."""
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenarios import list_scenarios
    from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenario_generator import save_seed_scenarios

    # Save some scenarios
    save_seed_scenarios(str(temp_dir))

    # List them
    scenarios = list_scenarios(temp_dir)
    assert len(scenarios) == 5

    # Verify metadata is accessible
    for s in scenarios:
      assert 'name' in s
      assert 'path' in s

  def test_load_nonexistent_file(self):
    """Test loading nonexistent file raises error."""
    pytest.importorskip("pandas")

    from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenarios import load_scenario

    with pytest.raises(FileNotFoundError):
      load_scenario("/nonexistent/path/scenario.parquet")


# ============================================================================
# Integration Tests
# ============================================================================


class TestScenarioRunnerIntegration:
  """Integration tests for scenarios with runner."""

  def test_run_generated_scenario(self, scenario_runner, lateral_pid_adapter):
    """Test running generated scenario through runner."""
    scenario, _ = generate_highway_straight(duration_s=1.0)
    result = scenario_runner.run(lateral_pid_adapter, scenario)

    assert result.success
    assert result.metrics.total_steps == len(scenario.states)

  def test_run_all_seed_scenarios(self, scenario_runner, lateral_pid_adapter):
    """Test running all seed scenarios."""
    scenarios = generate_all_seed_scenarios()

    for name, (scenario, metadata) in scenarios.items():
      # Skip longitudinal scenarios for lateral adapter
      if metadata.scenario_type == ScenarioType.EMERGENCY_STOP:
        continue

      result = scenario_runner.run(lateral_pid_adapter, scenario, name)
      assert result.success, f"Failed on {name}: {result.error_message}"

  def test_compare_on_generated_scenarios(self, scenario_runner):
    """Test comparing algorithms on generated scenarios."""
    from openpilot.selfdrive.controls.lib.tests.algorithm_harness.adapters import LatControlPIDAdapter, LatControlTorqueAdapter

    scenarios = [
      generate_highway_straight(duration_s=1.0)[0],
      generate_tight_s_curve(duration_s=1.0)[0],
    ]

    comparison = scenario_runner.compare(
      LatControlPIDAdapter(),
      LatControlTorqueAdapter(),
      scenarios,
    )

    assert len(comparison['per_scenario']) == 2
    assert 'aggregate' in comparison
