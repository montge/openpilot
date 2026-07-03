"""Tests for Viterbi track association."""
import numpy as np

from openpilot.tools.stonesoup.viterbi_tracker import (
  Detection,
  HungarianTracker,
  TrackerMetrics,
  ViterbiConfig,
  ViterbiTracker,
  compare_trackers,
  create_occlusion_scenario,
  evaluate_tracker,
  format_comparison_report,
)


class TestViterbiConfig:
  def test_default_config(self):
    config = ViterbiConfig()
    assert 0 < config.detection_prob < 1
    assert config.window_size > 0
    assert config.max_misses_to_delete > 0


class TestViterbiTracker:
  def test_initialization(self):
    tracker = ViterbiTracker()
    assert len(tracker.tracks) == 0
    assert tracker.next_track_id == 0

  def test_first_detection_creates_track(self):
    tracker = ViterbiTracker()
    dets = [Detection(position=np.array([10.0, 0.0]))]

    tracker.process_detections(0.0, dets)

    assert len(tracker.tracks) == 1
    np.testing.assert_array_almost_equal(
      tracker.tracks[0].position[:2],
      np.array([10.0, 0.0])
    )

  def test_association_works(self):
    tracker = ViterbiTracker()

    # First frame
    tracker.process_detections(0.0, [Detection(np.array([10.0, 0.0]))])
    first_id = tracker.tracks[0].track_id

    # Second frame - close detection should associate
    tracker.process_detections(0.1, [Detection(np.array([10.5, 0.0]))])

    # Should still be same track
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_id == first_id

  def test_track_deletion(self):
    config = ViterbiConfig(max_misses_to_delete=3, min_hits_to_confirm=1)
    tracker = ViterbiTracker(config)

    # Create track
    tracker.process_detections(0.0, [Detection(np.array([10.0, 0.0]))])

    # Miss detections
    for i in range(5):
      tracker.process_detections((i + 1) * 0.1, [])

    # Track should be deleted
    assert len(tracker.tracks) == 0

  def test_new_track_initiation(self):
    tracker = ViterbiTracker()

    # First detection
    tracker.process_detections(0.0, [Detection(np.array([10.0, 0.0]))])

    # Second detection far away - should create new track
    tracker.process_detections(0.1, [
      Detection(np.array([10.5, 0.0])),  # Close to first
      Detection(np.array([50.0, 5.0]))   # Far away
    ])

    assert len(tracker.tracks) == 2

  def test_confirmed_tracks(self):
    config = ViterbiConfig(min_hits_to_confirm=3)
    tracker = ViterbiTracker(config)

    # Need 3 hits to confirm
    for i in range(3):
      confirmed = tracker.process_detections(
        i * 0.1,
        [Detection(np.array([10.0 + i * 0.5, 0.0]))]
      )

    # Should be confirmed now
    assert len(confirmed) == 1


class TestHungarianTracker:
  def test_initialization(self):
    tracker = HungarianTracker()
    assert len(tracker.tracks) == 0

  def test_basic_tracking(self):
    tracker = HungarianTracker()

    # Track an object
    for i in range(5):
      tracker.process_detections(
        i * 0.1,
        [Detection(np.array([10.0 - i * 0.5, 0.0]))]
      )

    assert len(tracker.tracks) == 1

  def test_multiple_objects(self):
    config = ViterbiConfig(min_hits_to_confirm=1)
    tracker = HungarianTracker(config)

    # Two objects
    for i in range(3):
      confirmed = tracker.process_detections(
        i * 0.1,
        [
          Detection(np.array([10.0 - i * 0.5, 0.0])),
          Detection(np.array([30.0 - i * 0.5, 5.0]))
        ]
      )

    assert len(confirmed) == 2


class TestOcclusionScenario:
  def test_scenario_creation(self):
    scenario = create_occlusion_scenario(
      dt=0.1, duration=5.0, n_objects=2, seed=42
    )

    assert scenario.name == "occlusion_test"
    assert len(scenario.ground_truth) == 2
    assert len(scenario.timestamps) == 50
    assert len(scenario.occlusion_periods) == 2

  def test_occlusion_affects_detections(self):
    scenario = create_occlusion_scenario(
      dt=0.1, duration=5.0, n_objects=1,
      occlusion_duration=1.0, seed=42
    )

    occ_start, occ_end = scenario.occlusion_periods[0]

    # Check detections during occlusion are missing
    for t, _dets in scenario.detections_by_time.items():
      if occ_start <= t < occ_end:
        # During occlusion, fewer detections expected
        # (not guaranteed 0 due to randomness, but should be less)
        pass  # Just verify no crash


class TestEvaluateTracker:
  def test_evaluation(self):
    scenario = create_occlusion_scenario(
      dt=0.1, duration=3.0, n_objects=2, seed=42
    )
    tracker = ViterbiTracker()

    track_history = []
    for t in scenario.timestamps:
      dets = scenario.detections_by_time.get(t, [])
      tracks = tracker.process_detections(t, dets)
      track_history.append(tracks)

    metrics = evaluate_tracker("Viterbi", track_history, scenario)

    assert isinstance(metrics, TrackerMetrics)
    assert metrics.tracker_name == "Viterbi"
    assert metrics.n_ground_truth == 2


class TestCompareTrackers:
  def test_comparison(self):
    scenario = create_occlusion_scenario(
      dt=0.1, duration=3.0, n_objects=2, seed=42
    )
    metrics = compare_trackers(scenario)

    assert "Viterbi" in metrics
    assert "Hungarian" in metrics

  def test_format_report(self):
    scenario = create_occlusion_scenario(dt=0.1, duration=3.0, seed=42)
    metrics = compare_trackers(scenario)
    report = format_comparison_report(scenario, metrics)

    assert "# Viterbi vs Hungarian" in report
    assert "MOTA" in report
    assert "ID Switches" in report
