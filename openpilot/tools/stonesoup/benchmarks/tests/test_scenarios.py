"""Unit tests for benchmark scenarios."""

from __future__ import annotations

from openpilot.tools.stonesoup.benchmarks.scenarios import (
  Detection,
  VehicleState,
  create_cut_in_scenario,
  create_cut_out_scenario,
  create_highway_scenario,
  create_multi_vehicle_scenario,
  create_noisy_scenario,
  create_occlusion_scenario,
)


class TestVehicleState:
  """Tests for VehicleState dataclass."""

  def test_creation(self):
    """Test VehicleState creation."""
    state = VehicleState(x=50.0, y=0.0, vx=30.0)

    assert state.x == 50.0
    assert state.y == 0.0
    assert state.vx == 30.0
    assert state.vy == 0.0  # Default

  def test_with_acceleration(self):
    """Test VehicleState with acceleration."""
    state = VehicleState(x=0.0, y=0.0, vx=20.0, ax=-2.0)

    assert state.ax == -2.0


class TestDetection:
  """Tests for Detection dataclass."""

  def test_creation(self):
    """Test Detection creation."""
    det = Detection(d_rel=50.0, v_rel=-5.0, y_rel=0.0, timestamp=1.0)

    assert det.d_rel == 50.0
    assert det.v_rel == -5.0
    assert det.valid  # Default True

  def test_invalid_detection(self):
    """Test invalid (missed) detection."""
    det = Detection(d_rel=0.0, v_rel=0.0, y_rel=0.0, timestamp=1.0, valid=False)

    assert not det.valid


class TestHighwayScenario:
  """Tests for highway following scenario."""

  def test_creation(self):
    """Test scenario creation."""
    scenario = create_highway_scenario()

    assert scenario.name == "highway_following"
    assert scenario.duration == 30.0
    assert scenario.dt == 0.05

  def test_frame_count(self):
    """Test correct number of frames."""
    scenario = create_highway_scenario(duration=10.0, dt=0.1)

    assert scenario.n_frames == 100

  def test_ego_trajectory_length(self):
    """Test ego trajectory has correct length."""
    scenario = create_highway_scenario(duration=5.0, dt=0.05)

    assert len(scenario.ego_trajectory) == scenario.n_frames

  def test_vehicle_trajectory_length(self):
    """Test vehicle trajectory has correct length."""
    scenario = create_highway_scenario()

    for _vid, traj in scenario.vehicle_trajectories.items():
      assert len(traj) == scenario.n_frames

  def test_detections_length(self):
    """Test detections have correct length."""
    scenario = create_highway_scenario()

    for _vid, dets in scenario.detections.items():
      assert len(dets) == scenario.n_frames

  def test_all_detections_valid(self):
    """Test all detections are valid in highway scenario."""
    scenario = create_highway_scenario()

    for det in scenario.detections[1]:
      assert det.valid

  def test_detection_noise_applied(self):
    """Test noise is applied to detections."""
    scenario = create_highway_scenario(noise_std=1.0)

    # Check that some detections have non-zero noise
    has_noise = any(det.noise.get("d", 0) != 0 for det in scenario.detections[1])
    assert has_noise

  def test_custom_parameters(self):
    """Test custom parameters work."""
    scenario = create_highway_scenario(
      duration=15.0,
      lead_distance=60.0,
      lead_velocity=35.0,
      ego_velocity=30.0,
    )

    assert scenario.duration == 15.0
    assert scenario.metadata["lead_distance"] == 60.0


class TestCutInScenario:
  """Tests for cut-in scenario."""

  def test_creation(self):
    """Test scenario creation."""
    scenario = create_cut_in_scenario()

    assert scenario.name == "cut_in"

  def test_lateral_transition(self):
    """Test vehicle moves from lateral position to center."""
    scenario = create_cut_in_scenario(
      cut_in_time=5.0,
      cut_in_duration=3.0,
      initial_lateral=3.5,
    )

    trajectory = scenario.vehicle_trajectories[1]

    # Before cut-in: should be at initial lateral
    pre_cut_in_idx = int(4.0 / scenario.dt)  # t=4s
    assert abs(trajectory[pre_cut_in_idx].y - 3.5) < 0.5

    # After cut-in: should be near center
    post_cut_in_idx = int(9.0 / scenario.dt)  # t=9s
    assert abs(trajectory[post_cut_in_idx].y) < 0.5


class TestCutOutScenario:
  """Tests for cut-out scenario."""

  def test_creation(self):
    """Test scenario creation."""
    scenario = create_cut_out_scenario()

    assert scenario.name == "cut_out"

  def test_two_vehicles(self):
    """Test scenario has two vehicles."""
    scenario = create_cut_out_scenario()

    assert len(scenario.vehicle_trajectories) == 2
    assert 1 in scenario.vehicle_trajectories
    assert 2 in scenario.vehicle_trajectories

  def test_new_lead_revealed(self):
    """Test new lead becomes visible after cut-out."""
    scenario = create_cut_out_scenario(
      cut_out_time=5.0,
      cut_out_duration=3.0,
    )

    dets = scenario.detections[2]  # New lead

    # Before cut-out: new lead not visible
    pre_idx = int(4.0 / scenario.dt)
    assert not dets[pre_idx].valid

    # After cut-out: new lead visible
    post_idx = int(10.0 / scenario.dt)
    assert dets[post_idx].valid


class TestMultiVehicleScenario:
  """Tests for multi-vehicle scenario."""

  def test_creation(self):
    """Test scenario creation."""
    scenario = create_multi_vehicle_scenario(n_vehicles=3)

    assert scenario.name == "multi_vehicle"
    assert len(scenario.vehicle_trajectories) == 3

  def test_variable_vehicle_count(self):
    """Test different vehicle counts."""
    for n in [1, 2, 3]:
      scenario = create_multi_vehicle_scenario(n_vehicles=n)
      assert len(scenario.vehicle_trajectories) == n

  def test_all_detections_at_frame(self):
    """Test getting all detections at a frame."""
    scenario = create_multi_vehicle_scenario(n_vehicles=3)

    all_dets = scenario.get_all_detections_at(100)
    assert len(all_dets) == 3


class TestOcclusionScenario:
  """Tests for occlusion scenario."""

  def test_creation(self):
    """Test scenario creation."""
    scenario = create_occlusion_scenario()

    assert scenario.name == "occlusion"

  def test_occlusion_period(self):
    """Test detections are invalid during occlusion."""
    scenario = create_occlusion_scenario(
      occlusion_start=10.0,
      occlusion_duration=5.0,
    )

    dets = scenario.detections[1]
    timestamps = scenario.timestamps

    # Count invalid detections in occlusion period
    occluded_count = 0
    for i, det in enumerate(dets):
      t = timestamps[i]
      if 10.0 <= t < 15.0:
        if not det.valid:
          occluded_count += 1

    # All detections should be invalid during occlusion
    expected_occluded = int(5.0 / scenario.dt)
    assert occluded_count == expected_occluded

  def test_valid_before_after_occlusion(self):
    """Test detections valid before and after occlusion."""
    scenario = create_occlusion_scenario(
      occlusion_start=10.0,
      occlusion_duration=5.0,
    )

    dets = scenario.detections[1]

    # Before occlusion
    pre_idx = int(5.0 / scenario.dt)
    assert dets[pre_idx].valid

    # After occlusion
    post_idx = int(20.0 / scenario.dt)
    assert dets[post_idx].valid


class TestNoisyScenario:
  """Tests for noisy scenario."""

  def test_creation(self):
    """Test scenario creation."""
    scenario = create_noisy_scenario()

    assert scenario.name == "noisy"

  def test_higher_noise(self):
    """Test noise is higher than base."""
    scenario = create_noisy_scenario(
      base_noise_std=0.5,
      noise_multiplier=3.0,
    )

    assert scenario.metadata["noise_std"] == 1.5

  def test_has_missed_detections(self):
    """Test some detections are missed."""
    scenario = create_noisy_scenario(miss_rate=0.2)

    missed = sum(1 for det in scenario.detections[1] if not det.valid)
    # Should have some missed (probabilistic, so check > 0)
    assert missed > 0

  def test_has_false_alarms(self):
    """Test scenario has false alarms."""
    scenario = create_noisy_scenario(false_alarm_rate=0.2)

    false_alarms = scenario.metadata.get("false_alarms", [])
    # Should have some false alarms
    assert len(false_alarms) > 0


class TestBenchmarkScenarioProperties:
  """Tests for BenchmarkScenario properties and methods."""

  def test_timestamps(self):
    """Test timestamps property."""
    scenario = create_highway_scenario(duration=10.0, dt=0.1)

    timestamps = scenario.timestamps
    assert len(timestamps) == 100
    assert timestamps[0] == 0.0
    assert abs(timestamps[-1] - 9.9) < 0.01

  def test_get_all_detections_at(self):
    """Test getting all detections at a frame."""
    scenario = create_multi_vehicle_scenario(n_vehicles=2)

    dets = scenario.get_all_detections_at(50)
    assert len(dets) == 2

  def test_get_all_detections_at_with_invalid(self):
    """Test get_all_detections_at filters invalid."""
    scenario = create_occlusion_scenario(
      occlusion_start=5.0,
      occlusion_duration=5.0,
    )

    # During occlusion - should return empty list
    occluded_idx = int(7.0 / scenario.dt)
    dets = scenario.get_all_detections_at(occluded_idx)
    assert len(dets) == 0

    # Before occlusion - should return 1
    pre_idx = int(2.0 / scenario.dt)
    dets = scenario.get_all_detections_at(pre_idx)
    assert len(dets) == 1


class TestScenarioReproducibility:
  """Tests for scenario reproducibility."""

  def test_same_seed_same_output(self):
    """Test same scenario twice gives same detections."""
    s1 = create_highway_scenario(noise_std=1.0)
    s2 = create_highway_scenario(noise_std=1.0)

    # Should be identical (same seed)
    for d1, d2 in zip(s1.detections[1], s2.detections[1], strict=True):
      assert d1.d_rel == d2.d_rel
      assert d1.v_rel == d2.v_rel
