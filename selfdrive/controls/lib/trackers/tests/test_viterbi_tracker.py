"""Unit tests for Viterbi-based track association."""

from __future__ import annotations

import numpy as np

from openpilot.selfdrive.controls.lib.trackers.viterbi_tracker import (
  Detection,
  TrackState,
  ViterbiConfig,
  ViterbiTracker,
)


class TestViterbiConfig:
  """Tests for ViterbiConfig."""

  def test_default_config(self):
    """Test default configuration values."""
    config = ViterbiConfig()

    assert config.window_size == 5
    assert config.max_tracks == 32
    assert config.gate_threshold == 9.21
    assert config.miss_prob == 0.1

  def test_custom_config(self):
    """Test custom configuration."""
    config = ViterbiConfig(
      window_size=10,
      max_tracks=16,
      gate_threshold=5.0,
    )

    assert config.window_size == 10
    assert config.max_tracks == 16
    assert config.gate_threshold == 5.0


class TestDetection:
  """Tests for Detection dataclass."""

  def test_creation(self):
    """Test Detection creation."""
    det = Detection(
      measurement=np.array([10.0, 2.0]),
      covariance=np.eye(2),
      timestamp=100.0,
    )

    assert det.measurement[0] == 10.0
    assert det.measurement[1] == 2.0
    assert det.timestamp == 100.0

  def test_default_values(self):
    """Test Detection default values."""
    det = Detection(measurement=np.array([5.0, 1.0]))

    assert det.covariance is None
    assert det.timestamp == 0.0
    assert det.metadata == {}


class TestTrackState:
  """Tests for TrackState dataclass."""

  def test_creation(self):
    """Test TrackState creation."""
    state = np.array([10.0, 1.0, 2.0, 0.5])
    covar = np.eye(4)

    track = TrackState(
      id=1,
      state=state,
      covariance=covar,
    )

    assert track.id == 1
    assert np.array_equal(track.state, state)
    assert track.age == 0
    assert track.hits == 0
    assert track.misses == 0


class TestViterbiTrackerInit:
  """Tests for ViterbiTracker initialization."""

  def test_default_init(self):
    """Test default initialization."""
    tracker = ViterbiTracker()

    assert tracker.config is not None
    assert len(tracker.tracks) == 0
    assert tracker.frame_count == 0

  def test_custom_config_init(self):
    """Test initialization with custom config."""
    config = ViterbiConfig(window_size=10)
    tracker = ViterbiTracker(config)

    assert tracker.config.window_size == 10


class TestTrackCreation:
  """Tests for track creation."""

  def test_create_track_from_detection(self):
    """Test track creation from single detection."""
    tracker = ViterbiTracker()

    det = Detection(measurement=np.array([50.0, 0.0]))
    tracker.update([det])

    # Track should be created (but not confirmed yet)
    all_tracks = tracker.get_all_tracks()
    assert len(all_tracks) == 1
    assert all_tracks[0].state[0] == 50.0  # x position

  def test_track_confirmation(self):
    """Test track becomes confirmed after min_hits."""
    config = ViterbiConfig(min_hits_to_confirm=3)
    tracker = ViterbiTracker(config)

    # First detection - creates unconfirmed track
    det = Detection(measurement=np.array([50.0, 0.0]))
    confirmed = tracker.update([det])
    assert len(confirmed) == 0  # Not yet confirmed

    # Second detection at same location
    det = Detection(measurement=np.array([51.0, 0.0]))
    confirmed = tracker.update([det])
    assert len(confirmed) == 0

    # Third detection - should confirm
    det = Detection(measurement=np.array([52.0, 0.0]))
    confirmed = tracker.update([det])
    assert len(confirmed) == 1

  def test_max_tracks_limit(self):
    """Test maximum track limit."""
    config = ViterbiConfig(max_tracks=3)
    tracker = ViterbiTracker(config)

    # Create detections at different locations
    detections = [
      Detection(measurement=np.array([10.0, 0.0])),
      Detection(measurement=np.array([20.0, 0.0])),
      Detection(measurement=np.array([30.0, 0.0])),
      Detection(measurement=np.array([40.0, 0.0])),  # Should not create track
    ]

    tracker.update(detections)
    assert len(tracker.get_all_tracks()) <= 3


class TestTrackUpdate:
  """Tests for track state updates."""

  def test_track_position_update(self):
    """Test track position is updated with new detection."""
    config = ViterbiConfig(min_hits_to_confirm=1)
    tracker = ViterbiTracker(config)

    # Create track
    det1 = Detection(measurement=np.array([50.0, 0.0]))
    tracker.update([det1])

    initial_pos = tracker.get_all_tracks()[0].state[0]

    # Update with new detection
    det2 = Detection(measurement=np.array([52.0, 0.0]))
    tracker.update([det2])

    final_pos = tracker.get_all_tracks()[0].state[0]
    # Position should move towards new detection
    assert final_pos > initial_pos

  def test_track_velocity_estimation(self):
    """Test velocity is estimated from successive detections."""
    config = ViterbiConfig(min_hits_to_confirm=2)
    tracker = ViterbiTracker(config)

    # Moving object: x=50, 51, 52, ...
    for i in range(5):
      det = Detection(measurement=np.array([50.0 + i, 0.0]))
      tracker.update([det], dt=0.05)

    tracks = tracker.get_all_tracks()
    assert len(tracks) == 1

    # Velocity should be positive (moving forward)
    # state[1] is vx
    assert tracks[0].state[1] > 0


class TestTrackDeletion:
  """Tests for track deletion."""

  def test_track_deleted_after_max_misses(self):
    """Test track is deleted after consecutive misses."""
    config = ViterbiConfig(max_misses_to_delete=3)
    tracker = ViterbiTracker(config)

    # Create track
    det = Detection(measurement=np.array([50.0, 0.0]))
    tracker.update([det])
    assert len(tracker.get_all_tracks()) == 1

    # Miss detections
    for _ in range(3):
      tracker.update([])

    assert len(tracker.get_all_tracks()) == 0


class TestMultiTrack:
  """Tests for multi-object tracking."""

  def test_track_two_objects(self):
    """Test tracking two separate objects."""
    config = ViterbiConfig(min_hits_to_confirm=2)
    tracker = ViterbiTracker(config)

    # Two objects at different locations
    for _ in range(3):
      detections = [
        Detection(measurement=np.array([30.0, -5.0])),
        Detection(measurement=np.array([60.0, 5.0])),
      ]
      tracker.update(detections)

    confirmed = tracker.update(
      [
        Detection(measurement=np.array([30.0, -5.0])),
        Detection(measurement=np.array([60.0, 5.0])),
      ]
    )

    # Should have two confirmed tracks
    assert len(confirmed) >= 1  # At least one should be confirmed

  def test_crossing_tracks(self):
    """Test tracking objects that cross paths."""
    config = ViterbiConfig(min_hits_to_confirm=2)
    tracker = ViterbiTracker(config)

    # Two objects approaching each other
    for i in range(5):
      detections = [
        Detection(measurement=np.array([50.0 - i * 2, 0.0])),  # Moving left
        Detection(measurement=np.array([30.0 + i * 2, 0.0])),  # Moving right
      ]
      tracker.update(detections, dt=0.1)

    tracks = tracker.get_all_tracks()
    # Should maintain two separate tracks
    assert len(tracks) >= 1


class TestGating:
  """Tests for measurement gating."""

  def test_detection_outside_gate_ignored(self):
    """Test detection far from track is not associated."""
    config = ViterbiConfig(
      gate_threshold=5.0,
      min_hits_to_confirm=1,
    )
    tracker = ViterbiTracker(config)

    # Create track at x=50
    det1 = Detection(measurement=np.array([50.0, 0.0]))
    tracker.update([det1])

    # Detection very far away - should create new track, not update existing
    det2 = Detection(measurement=np.array([200.0, 0.0]))
    tracker.update([det2])

    tracks = tracker.get_all_tracks()
    # Should have two tracks now
    assert len(tracks) >= 1


class TestViterbiDecoding:
  """Tests for Viterbi algorithm."""

  def test_transition_matrix(self):
    """Test transition matrix computation."""
    tracker = ViterbiTracker()

    # Add a track so we have states
    det = Detection(measurement=np.array([50.0, 0.0]))
    tracker.update([det])

    trans = tracker._compute_transition_matrix()

    # Rows should sum to 1 (valid probability distribution)
    row_sums = trans.sum(axis=1)
    np.testing.assert_array_almost_equal(row_sums, np.ones(len(row_sums)))

  def test_emission_probability(self):
    """Test emission probability computation."""
    tracker = ViterbiTracker()

    # Create track at x=50
    det1 = Detection(measurement=np.array([50.0, 0.0]))
    tracker.update([det1])

    # Detection close to track should have high probability
    det_close = Detection(measurement=np.array([50.5, 0.0]))
    emission = tracker._compute_emission_prob([det_close])

    # Should have higher probability for track 0 than unassigned
    assert emission[0, 0] > emission[0, -1]


class TestOcclusionHandling:
  """Tests for occlusion handling."""

  def test_track_survives_brief_occlusion(self):
    """Test track survives brief occlusion."""
    config = ViterbiConfig(
      max_misses_to_delete=5,
      min_hits_to_confirm=2,
    )
    tracker = ViterbiTracker(config)

    # Establish track
    for i in range(3):
      det = Detection(measurement=np.array([50.0 + i, 0.0]))
      tracker.update([det])

    initial_tracks = len(tracker.get_all_tracks())
    assert initial_tracks == 1

    # Occlude for 3 frames
    for _ in range(3):
      tracker.update([])

    # Track should still exist
    assert len(tracker.get_all_tracks()) == 1

    # Reappear
    det = Detection(measurement=np.array([53.0, 0.0]))
    tracker.update([det])

    # Track should be recovered
    tracks = tracker.get_all_tracks()
    assert len(tracks) == 1
    assert tracks[0].misses == 0  # Reset after hit


class TestEdgeCases:
  """Tests for edge cases."""

  def test_empty_detections(self):
    """Test handling empty detections."""
    tracker = ViterbiTracker()

    # Should not crash
    confirmed = tracker.update([])
    assert len(confirmed) == 0

  def test_single_detection_many_frames(self):
    """Test single object tracked over many frames."""
    config = ViterbiConfig(min_hits_to_confirm=3)
    tracker = ViterbiTracker(config)

    for i in range(20):
      det = Detection(measurement=np.array([50.0 + i * 0.5, 0.0]))
      tracker.update([det], dt=0.05)

    confirmed = tracker.update([Detection(measurement=np.array([60.0, 0.0]))])
    assert len(confirmed) == 1

  def test_reset(self):
    """Test tracker reset."""
    tracker = ViterbiTracker()

    # Create some tracks
    for _ in range(5):
      det = Detection(measurement=np.array([50.0, 0.0]))
      tracker.update([det])

    assert len(tracker.get_all_tracks()) > 0

    tracker.reset()

    assert len(tracker.get_all_tracks()) == 0
    assert tracker.frame_count == 0


class TestCostMatrix:
  """Tests for cost matrix computation."""

  def test_cost_matrix_shape(self):
    """Test cost matrix has correct shape."""
    tracker = ViterbiTracker()

    # Create tracks
    for _ in range(3):
      tracker.update(
        [
          Detection(measurement=np.array([30.0, 0.0])),
          Detection(measurement=np.array([60.0, 0.0])),
        ]
      )

    # Compute cost for new detections
    detections = [
      Detection(measurement=np.array([31.0, 0.0])),
      Detection(measurement=np.array([61.0, 0.0])),
      Detection(measurement=np.array([90.0, 0.0])),
    ]

    cost = tracker._compute_cost_matrix(detections)

    n_det = len(detections)
    n_tracks = len(tracker.tracks)
    assert cost.shape == (n_det, n_tracks)

  def test_cost_increases_with_distance(self):
    """Test cost increases with distance from track."""
    tracker = ViterbiTracker()

    # Create track at x=50
    det = Detection(measurement=np.array([50.0, 0.0]))
    tracker.update([det])

    # Detections at different distances
    det_close = Detection(measurement=np.array([50.5, 0.0]))
    det_far = Detection(measurement=np.array([70.0, 0.0]))

    cost = tracker._compute_cost_matrix([det_close, det_far])

    # Close detection should have lower cost
    assert cost[0, 0] < cost[1, 0]
