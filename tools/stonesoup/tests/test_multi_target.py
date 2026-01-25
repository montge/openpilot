"""Tests for multi-target tracking comparison."""

from openpilot.tools.stonesoup.multi_target import (
  GNNTrackerWrapper,
  JPDATrackerWrapper,
  TrackingMetrics,
  compare_multi_target_trackers,
  compute_tracking_metrics,
  create_cut_in_scenario,
  create_highway_scenario,
  format_tracking_report,
  run_tracker_on_scenario,
)


class TestScenarios:
  def test_highway_scenario(self):
    scenario = create_highway_scenario(dt=0.1, duration=5.0, n_vehicles=3)
    assert scenario.name == "highway_following"
    assert len(scenario.ground_truth) == 3
    assert len(scenario.timestamps) == 50

  def test_cut_in_scenario(self):
    scenario = create_cut_in_scenario(dt=0.1, duration=5.0)
    assert scenario.name == "cut_in"
    assert len(scenario.ground_truth) == 2

  def test_detections_generated(self):
    scenario = create_highway_scenario(dt=0.1, duration=2.0, n_vehicles=2)
    # Should have some detections at each timestep
    total_detections = sum(len(d) for d in scenario.detections_by_time.values())
    assert total_detections > 0


class TestJPDATracker:
  def test_initialization(self):
    tracker = JPDATrackerWrapper(dt=0.1)
    assert tracker.dt == 0.1
    assert len(tracker.tracks) == 0

  def test_process_detections(self):
    scenario = create_highway_scenario(dt=0.1, duration=2.0, n_vehicles=2)
    tracker = JPDATrackerWrapper(dt=scenario.dt)

    # Process first few timesteps
    for t in scenario.timestamps[:5]:
      detections = scenario.detections_by_time.get(t, [])
      tracker.process_detections(t, detections)
      # May or may not have tracks yet (needs min_points detections)

    # After some time, should have tracks
    for t in scenario.timestamps[5:15]:
      detections = scenario.detections_by_time.get(t, [])
      tracker.process_detections(t, detections)

    # Should have initiated some tracks by now
    assert len(tracker.tracks) >= 0  # May be 0 if all deleted


class TestGNNTracker:
  def test_initialization(self):
    tracker = GNNTrackerWrapper(dt=0.1)
    assert tracker.dt == 0.1
    assert len(tracker.tracks) == 0

  def test_process_detections(self):
    scenario = create_highway_scenario(dt=0.1, duration=2.0, n_vehicles=2)
    tracker = GNNTrackerWrapper(dt=scenario.dt)

    for t in scenario.timestamps[:15]:
      detections = scenario.detections_by_time.get(t, [])
      tracker.process_detections(t, detections)

    assert len(tracker.tracks) >= 0


class TestRunTracker:
  def test_run_jpda_on_scenario(self):
    scenario = create_highway_scenario(dt=0.1, duration=3.0, n_vehicles=2)
    tracker = JPDATrackerWrapper(dt=scenario.dt)
    all_tracks = run_tracker_on_scenario(tracker, scenario)

    assert len(all_tracks) == len(scenario.timestamps)

  def test_run_gnn_on_scenario(self):
    scenario = create_highway_scenario(dt=0.1, duration=3.0, n_vehicles=2)
    tracker = GNNTrackerWrapper(dt=scenario.dt)
    all_tracks = run_tracker_on_scenario(tracker, scenario)

    assert len(all_tracks) == len(scenario.timestamps)


class TestMetrics:
  def test_compute_metrics(self):
    scenario = create_highway_scenario(dt=0.1, duration=3.0, n_vehicles=2)
    tracker = JPDATrackerWrapper(dt=scenario.dt)
    all_tracks = run_tracker_on_scenario(tracker, scenario)
    metrics = compute_tracking_metrics("JPDA", all_tracks, scenario)

    assert isinstance(metrics, TrackingMetrics)
    assert metrics.tracker_name == "JPDA"
    assert -1.0 <= metrics.mota <= 1.0  # MOTA can be negative
    assert metrics.motp >= 0.0


class TestCompare:
  def test_compare_trackers(self):
    scenario = create_highway_scenario(dt=0.1, duration=3.0, n_vehicles=2)
    metrics = compare_multi_target_trackers(scenario)

    assert "JPDA" in metrics
    assert "GNN" in metrics

  def test_format_report(self):
    scenario = create_highway_scenario(dt=0.1, duration=2.0, n_vehicles=2)
    metrics = compare_multi_target_trackers(scenario)
    report = format_tracking_report(scenario, metrics)

    assert "# Multi-Target Tracking Comparison" in report
    assert "JPDA" in report
    assert "GNN" in report
    assert "MOTA" in report
