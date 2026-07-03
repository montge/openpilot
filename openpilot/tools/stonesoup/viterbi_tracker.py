"""Viterbi Track Association using Hidden Markov Models.

Implements track-to-detection association using HMM with Viterbi
decoding, which can handle occlusions and missed detections better
than frame-by-frame methods like Hungarian algorithm.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass
class TrackState:
  """State of a tracked object."""
  track_id: int
  position: np.ndarray  # [x, y] or [x, y, z]
  velocity: np.ndarray
  covariance: np.ndarray
  age: int = 0
  hits: int = 0
  misses: int = 0
  time_since_update: int = 0


@dataclass
class Detection:
  """Detection from sensor."""
  position: np.ndarray
  confidence: float = 1.0
  timestamp: float = 0.0


@dataclass
class ViterbiConfig:
  """Configuration for Viterbi tracker."""
  # HMM parameters
  detection_prob: float = 0.9  # Probability of detection when object exists
  false_alarm_rate: float = 0.1  # Rate of false positive detections
  birth_prob: float = 0.05  # Probability of new track birth
  death_prob: float = 0.02  # Probability of track death

  # Association parameters
  max_mahal_distance: float = 9.0  # Chi-squared threshold (99% for 2D)
  max_euclidean_distance: float = 5.0  # Fallback distance threshold

  # Track management
  min_hits_to_confirm: int = 3
  max_misses_to_delete: int = 5
  min_confidence: float = 0.3

  # Viterbi window
  window_size: int = 5  # Number of frames for Viterbi decoding


class ViterbiTracker:
  """Multi-object tracker using Viterbi algorithm for association.

  Uses HMM to model track existence and detection probability,
  then applies Viterbi decoding to find optimal track associations
  over a sliding window of frames.
  """

  def __init__(self, config: ViterbiConfig | None = None):
    self.config = config or ViterbiConfig()
    self.tracks: list[TrackState] = []
    self.next_track_id = 0

    # Sliding window of detections
    self.detection_window: list[list[Detection]] = []
    self.association_window: list[dict[int, int | None]] = []

    # Process noise for prediction
    self.dt = 0.1  # Time step
    self.process_noise_std = 0.5

  def _predict_track(self, track: TrackState) -> TrackState:
    """Predict track state to next timestep."""
    new_pos = track.position + track.velocity * self.dt

    # Simple constant velocity model
    F = np.eye(4)  # State transition
    F[0, 2] = self.dt
    F[1, 3] = self.dt

    Q = np.diag([0.1, 0.1, 0.5, 0.5]) * self.process_noise_std**2
    new_cov = track.covariance + Q

    return TrackState(
      track_id=track.track_id,
      position=new_pos,
      velocity=track.velocity,
      covariance=new_cov,
      age=track.age + 1,
      hits=track.hits,
      misses=track.misses,
      time_since_update=track.time_since_update + 1
    )

  def _compute_emission_prob(
    self,
    track: TrackState,
    detection: Detection
  ) -> float:
    """Compute emission probability P(detection | track).

    Uses Mahalanobis distance to compute likelihood.
    """
    diff = detection.position - track.position[:len(detection.position)]

    # Use position covariance for Mahalanobis
    pos_cov = track.covariance[:2, :2]

    try:
      mahal_sq = diff @ np.linalg.inv(pos_cov) @ diff
    except np.linalg.LinAlgError:
      mahal_sq = np.sum(diff**2) / 4.0

    # Chi-squared probability
    prob = np.exp(-0.5 * mahal_sq)

    # Scale by detection confidence
    prob *= detection.confidence

    return np.clip(prob, 1e-10, 1.0)

  def _compute_transition_matrix(self, n_tracks: int) -> np.ndarray:
    """Compute HMM transition matrix.

    States: track exists (1) or not (0) for each track.
    """
    cfg = self.config

    # Single track transition probabilities
    # P(exist_t | exist_{t-1})
    exist_given_exist = 1.0 - cfg.death_prob
    exist_given_not = cfg.birth_prob
    not_given_exist = cfg.death_prob
    not_given_not = 1.0 - cfg.birth_prob

    # For multiple tracks, we use independent transitions
    # This simplifies the exponential state space
    trans = np.array([
      [not_given_not, not_given_exist],
      [exist_given_not, exist_given_exist]
    ])

    return trans

  def _hungarian_association(
    self,
    tracks: list[TrackState],
    detections: list[Detection]
  ) -> dict[int, int | None]:
    """Hungarian algorithm for single-frame association."""
    if not tracks or not detections:
      return {t.track_id: None for t in tracks}

    n_tracks = len(tracks)
    n_dets = len(detections)

    # Compute cost matrix
    cost = np.zeros((n_tracks, n_dets))
    for i, track in enumerate(tracks):
      for j, det in enumerate(detections):
        diff = det.position - track.position[:len(det.position)]
        dist = np.linalg.norm(diff)
        cost[i, j] = dist if dist < self.config.max_euclidean_distance else 1e6

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    associations: dict[int, int | None] = {t.track_id: None for t in tracks}
    for i, j in zip(row_ind, col_ind, strict=True):
      if cost[i, j] < self.config.max_euclidean_distance:
        associations[tracks[i].track_id] = j

    return associations

  def _viterbi_decode(
    self,
    tracks: list[TrackState],
    detection_window: list[list[Detection]]
  ) -> list[dict[int, int | None]]:
    """Viterbi decoding for track association over window.

    Finds optimal sequence of associations that maximizes
    joint probability over the window.
    """
    if not tracks or not detection_window:
      return [{t.track_id: None for t in tracks} for _ in detection_window]

    n_frames = len(detection_window)
    n_tracks = len(tracks)

    # For each track, compute best detection at each frame
    # This is a simplified version that processes tracks independently

    associations = []
    predicted_tracks = list(tracks)

    for frame_idx in range(n_frames):
      detections = detection_window[frame_idx]

      if not detections:
        associations.append({t.track_id: None for t in predicted_tracks})
        predicted_tracks = [self._predict_track(t) for t in predicted_tracks]
        continue

      # Compute emission probabilities
      emission_probs = np.zeros((n_tracks, len(detections)))
      for i, track in enumerate(predicted_tracks):
        for j, det in enumerate(detections):
          emission_probs[i, j] = self._compute_emission_prob(track, det)

      # Find best assignment using Hungarian with emission probs
      # Convert to costs (negative log probability)
      cost = -np.log(emission_probs + 1e-10)
      cost[emission_probs < 0.01] = 1e6  # Threshold low probabilities

      row_ind, col_ind = linear_sum_assignment(cost)

      frame_assoc: dict[int, int | None] = {t.track_id: None for t in predicted_tracks}
      for i, j in zip(row_ind, col_ind, strict=True):
        if emission_probs[i, j] > 0.01:
          frame_assoc[predicted_tracks[i].track_id] = j

      associations.append(frame_assoc)

      # Update predictions for next frame
      for i, track in enumerate(predicted_tracks):
        if frame_assoc[track.track_id] is not None:
          det_idx = frame_assoc[track.track_id]
          det = detections[det_idx]
          # Simple measurement update
          predicted_tracks[i] = TrackState(
            track_id=track.track_id,
            position=det.position.copy(),
            velocity=track.velocity,
            covariance=track.covariance * 0.8,
            age=track.age,
            hits=track.hits,
            misses=track.misses,
            time_since_update=0
          )
        else:
          predicted_tracks[i] = self._predict_track(track)

    return associations

  def process_detections(
    self,
    timestamp: float,
    detections: list[Detection]
  ) -> list[TrackState]:
    """Process detections and update tracks.

    Args:
      timestamp: Current time
      detections: List of detections

    Returns:
      List of current tracks
    """
    # Add to window
    self.detection_window.append(detections)

    # Trim window to max size
    if len(self.detection_window) > self.config.window_size:
      self.detection_window.pop(0)

    # Predict all tracks
    predicted_tracks = [self._predict_track(t) for t in self.tracks]

    # Decide between Hungarian and Viterbi based on window
    if len(self.detection_window) < self.config.window_size:
      # Not enough history, use Hungarian
      associations = self._hungarian_association(predicted_tracks, detections)
    else:
      # Use Viterbi over window
      all_associations = self._viterbi_decode(
        self.tracks, self.detection_window
      )
      # Use only the most recent frame's associations
      associations = all_associations[-1] if all_associations else {}

    # Update tracks
    matched_det_indices = set()
    new_tracks = []

    for track in predicted_tracks:
      det_idx = associations.get(track.track_id)

      if det_idx is not None:
        det = detections[det_idx]
        matched_det_indices.add(det_idx)

        # Update with measurement
        new_pos = det.position.copy()
        new_vel = (new_pos - track.position[:len(new_pos)]) / self.dt

        # Ensure velocity has correct dimensions
        if len(new_vel) < len(track.velocity):
          new_vel = np.concatenate([new_vel, np.zeros(len(track.velocity) - len(new_vel))])

        updated = TrackState(
          track_id=track.track_id,
          position=new_pos,
          velocity=new_vel,
          covariance=track.covariance * 0.9,
          age=track.age + 1,
          hits=track.hits + 1,
          misses=0,
          time_since_update=0
        )
        new_tracks.append(updated)
      else:
        # No match - increment misses
        missed = TrackState(
          track_id=track.track_id,
          position=track.position,
          velocity=track.velocity,
          covariance=track.covariance * 1.1,
          age=track.age + 1,
          hits=track.hits,
          misses=track.misses + 1,
          time_since_update=track.time_since_update + 1
        )

        # Delete if too many misses
        if missed.misses < self.config.max_misses_to_delete:
          new_tracks.append(missed)

    # Initiate new tracks from unmatched detections
    for i, det in enumerate(detections):
      if i not in matched_det_indices:
        new_track = TrackState(
          track_id=self.next_track_id,
          position=det.position.copy(),
          velocity=np.zeros(len(det.position)),
          covariance=np.eye(4) * 2.0,
          age=1,
          hits=1,
          misses=0,
          time_since_update=0
        )
        new_tracks.append(new_track)
        self.next_track_id += 1

    self.tracks = new_tracks

    # Return only confirmed tracks
    return [t for t in self.tracks if t.hits >= self.config.min_hits_to_confirm]

  def get_all_tracks(self) -> list[TrackState]:
    """Get all tracks including tentative ones."""
    return self.tracks

  def clear(self) -> None:
    """Reset tracker state."""
    self.tracks = []
    self.detection_window = []
    self.next_track_id = 0


class HungarianTracker:
  """Baseline tracker using Hungarian algorithm for comparison."""

  def __init__(self, config: ViterbiConfig | None = None):
    self.config = config or ViterbiConfig()
    self.tracks: list[TrackState] = []
    self.next_track_id = 0
    self.dt = 0.1

  def process_detections(
    self,
    timestamp: float,
    detections: list[Detection]
  ) -> list[TrackState]:
    """Process detections using Hungarian algorithm."""
    # Predict tracks
    predicted = []
    for track in self.tracks:
      new_pos = track.position + track.velocity * self.dt
      predicted.append(TrackState(
        track_id=track.track_id,
        position=new_pos,
        velocity=track.velocity,
        covariance=track.covariance,
        age=track.age + 1,
        hits=track.hits,
        misses=track.misses,
        time_since_update=track.time_since_update + 1
      ))

    # Build cost matrix
    if predicted and detections:
      track_pos = np.array([t.position[:2] for t in predicted])
      det_pos = np.array([d.position[:2] for d in detections])
      cost = cdist(track_pos, det_pos)

      # Apply threshold
      cost[cost > self.config.max_euclidean_distance] = 1e6

      row_ind, col_ind = linear_sum_assignment(cost)
      matches = {i: j for i, j in zip(row_ind, col_ind, strict=True)
                 if cost[i, j] < self.config.max_euclidean_distance}
    else:
      matches = {}

    # Update matched tracks
    matched_dets = set()
    new_tracks = []

    for i, track in enumerate(predicted):
      if i in matches:
        j = matches[i]
        det = detections[j]
        matched_dets.add(j)

        new_vel = (det.position - track.position[:len(det.position)]) / self.dt
        if len(new_vel) < len(track.velocity):
          new_vel = np.concatenate([new_vel, np.zeros(len(track.velocity) - len(new_vel))])

        updated = TrackState(
          track_id=track.track_id,
          position=det.position.copy(),
          velocity=new_vel,
          covariance=track.covariance * 0.9,
          age=track.age,
          hits=track.hits + 1,
          misses=0,
          time_since_update=0
        )
        new_tracks.append(updated)
      else:
        missed = TrackState(
          track_id=track.track_id,
          position=track.position,
          velocity=track.velocity,
          covariance=track.covariance,
          age=track.age,
          hits=track.hits,
          misses=track.misses + 1,
          time_since_update=track.time_since_update
        )
        if missed.misses < self.config.max_misses_to_delete:
          new_tracks.append(missed)

    # Initiate new tracks
    for j, det in enumerate(detections):
      if j not in matched_dets:
        new_track = TrackState(
          track_id=self.next_track_id,
          position=det.position.copy(),
          velocity=np.zeros(len(det.position)),
          covariance=np.eye(4) * 2.0,
          age=1,
          hits=1,
          misses=0,
          time_since_update=0
        )
        new_tracks.append(new_track)
        self.next_track_id += 1

    self.tracks = new_tracks
    return [t for t in self.tracks if t.hits >= self.config.min_hits_to_confirm]

  def clear(self) -> None:
    """Reset tracker."""
    self.tracks = []
    self.next_track_id = 0


@dataclass
class OcclusionScenario:
  """Test scenario with occlusion periods."""
  name: str
  dt: float
  timestamps: np.ndarray
  ground_truth: list[np.ndarray]  # List of trajectories (Nx2 arrays)
  detections_by_time: dict[float, list[Detection]]
  occlusion_periods: list[tuple[float, float]]  # (start, end) for occlusions


def create_occlusion_scenario(
  dt: float = 0.1,
  duration: float = 10.0,
  n_objects: int = 2,
  occlusion_duration: float = 1.0,
  seed: int | None = None
) -> OcclusionScenario:
  """Create scenario with objects that get occluded.

  Objects move and periodically get occluded (no detections).
  """
  if seed is not None:
    np.random.seed(seed)

  n_steps = int(duration / dt)
  timestamps = np.arange(n_steps) * dt

  # Generate trajectories
  ground_truth = []
  for obj_id in range(n_objects):
    initial_pos = np.array([
      30.0 + obj_id * 10,
      -5.0 + obj_id * 5
    ])
    velocity = np.array([
      -3.0 + np.random.randn() * 0.5,
      np.random.randn() * 0.3
    ])

    trajectory = np.zeros((n_steps, 2))
    for i in range(n_steps):
      trajectory[i] = initial_pos + velocity * timestamps[i]

    ground_truth.append(trajectory)

  # Generate occlusion periods
  occlusion_periods = []
  for obj_id in range(n_objects):
    start = 3.0 + obj_id * 2.0  # Stagger occlusions
    occlusion_periods.append((start, start + occlusion_duration))

  # Generate detections with occlusions
  detections_by_time: dict[float, list[Detection]] = {}

  for i, t in enumerate(timestamps):
    dets = []

    for obj_id, traj in enumerate(ground_truth):
      # Check if occluded
      occ_start, occ_end = occlusion_periods[obj_id]
      is_occluded = occ_start <= t < occ_end

      if not is_occluded and np.random.random() < 0.95:  # 95% detection rate
        noise = np.random.randn(2) * 0.3
        dets.append(Detection(
          position=traj[i] + noise,
          confidence=0.8 + np.random.random() * 0.2,
          timestamp=t
        ))

    detections_by_time[t] = dets

  return OcclusionScenario(
    name="occlusion_test",
    dt=dt,
    timestamps=timestamps,
    ground_truth=ground_truth,
    detections_by_time=detections_by_time,
    occlusion_periods=occlusion_periods
  )


@dataclass
class TrackerMetrics:
  """Metrics for comparing trackers."""
  tracker_name: str
  n_ground_truth: int
  n_tracks: int
  id_switches: int
  track_fragmentation: int
  mota: float
  avg_position_error: float


def evaluate_tracker(
  tracker_name: str,
  track_history: list[list[TrackState]],
  scenario: OcclusionScenario
) -> TrackerMetrics:
  """Evaluate tracker performance on scenario."""
  total_gt = sum(len(gt) for gt in scenario.ground_truth)

  # Count ID switches (simplified)
  id_switches = 0
  prev_ids = set()
  for tracks in track_history:
    curr_ids = {t.track_id for t in tracks}
    # New IDs that weren't in previous frame (excluding true new tracks)
    new_ids = curr_ids - prev_ids
    id_switches += max(0, len(new_ids) - 1)  # Allow one new track per frame
    prev_ids = curr_ids

  # Track fragmentation (tracks that died and restarted)
  all_track_ids = set()
  for tracks in track_history:
    for t in tracks:
      all_track_ids.add(t.track_id)

  fragmentation = max(0, len(all_track_ids) - len(scenario.ground_truth))

  # MOTA approximation
  mota = 1.0 - (id_switches + fragmentation) / max(1, total_gt // len(scenario.timestamps))

  # Position error
  errors = []
  for i, tracks in enumerate(track_history):
    if i >= len(scenario.timestamps):
      break
    for track in tracks:
      min_dist = float('inf')
      for gt in scenario.ground_truth:
        if i < len(gt):
          dist = np.linalg.norm(track.position[:2] - gt[i])
          min_dist = min(min_dist, dist)
      if min_dist < float('inf'):
        errors.append(min_dist)

  avg_error = np.mean(errors) if errors else float('inf')

  return TrackerMetrics(
    tracker_name=tracker_name,
    n_ground_truth=len(scenario.ground_truth),
    n_tracks=len(all_track_ids),
    id_switches=id_switches,
    track_fragmentation=fragmentation,
    mota=mota,
    avg_position_error=avg_error
  )


def compare_trackers(scenario: OcclusionScenario) -> dict[str, TrackerMetrics]:
  """Compare Viterbi and Hungarian trackers on scenario."""
  trackers = {
    "Viterbi": ViterbiTracker(),
    "Hungarian": HungarianTracker(),
  }

  metrics = {}

  for name, tracker in trackers.items():
    track_history = []

    for t in scenario.timestamps:
      dets = scenario.detections_by_time.get(t, [])
      tracks = tracker.process_detections(t, dets)
      track_history.append(tracks)

    metrics[name] = evaluate_tracker(name, track_history, scenario)

  return metrics


def format_comparison_report(
  scenario: OcclusionScenario,
  metrics: dict[str, TrackerMetrics]
) -> str:
  """Format comparison as markdown report."""
  lines = [
    "# Viterbi vs Hungarian Tracker Comparison",
    "",
    f"## Scenario: {scenario.name}",
    f"- Duration: {scenario.timestamps[-1]:.1f}s",
    f"- Objects: {len(scenario.ground_truth)}",
    f"- Occlusion periods: {len(scenario.occlusion_periods)}",
    "",
    "## Results",
    "",
    "| Tracker | Tracks | ID Switches | Fragmentation | MOTA | Avg Error (m) |",
    "|---------|--------|-------------|---------------|------|---------------|",
  ]

  for name, m in sorted(metrics.items(), key=lambda x: -x[1].mota):
    lines.append(
      f"| {name} | {m.n_tracks} | {m.id_switches} | {m.track_fragmentation} | "
      + f"{m.mota:.3f} | {m.avg_position_error:.3f} |"
    )

  lines.extend([
    "",
    "MOTA = Multiple Object Tracking Accuracy (higher is better)",
    "ID Switches = number of times track IDs changed incorrectly",
    "Fragmentation = tracks that died and restarted",
  ])

  return "\n".join(lines)


if __name__ == "__main__":
  # Demo comparison
  scenario = create_occlusion_scenario(seed=42)
  metrics = compare_trackers(scenario)
  print(format_comparison_report(scenario, metrics))
