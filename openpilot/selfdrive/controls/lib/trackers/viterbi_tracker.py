"""Viterbi-based track association for multi-object tracking.

Uses Hidden Markov Model with Viterbi decoding to find globally optimal
track-to-detection assignments over a temporal window.

This approach can handle occlusions better than greedy assignment methods
by considering the entire sequence of observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> float:
  """Compute Mahalanobis distance.

  Args:
    x: Point to measure
    mean: Distribution mean
    cov_inv: Inverse covariance matrix

  Returns:
    Mahalanobis distance
  """
  diff = x - mean
  return float(np.sqrt(diff @ cov_inv @ diff))


@dataclass
class TrackState:
  """State of a tracked object.

  Attributes:
    id: Unique track identifier
    state: State vector [x, vx, y, vy, ...]
    covariance: State covariance matrix
    age: Number of frames since track creation
    hits: Number of successful associations
    misses: Number of consecutive missed associations
  """

  id: int
  state: np.ndarray
  covariance: np.ndarray
  age: int = 0
  hits: int = 0
  misses: int = 0


@dataclass
class Detection:
  """A detection from sensors.

  Attributes:
    measurement: Measurement vector [x, y, ...]
    covariance: Measurement noise covariance
    timestamp: Detection timestamp
    metadata: Additional detection metadata
  """

  measurement: np.ndarray
  covariance: np.ndarray | None = None
  timestamp: float = 0.0
  metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ViterbiConfig:
  """Configuration for Viterbi tracker.

  Attributes:
    window_size: Temporal window for Viterbi decoding
    max_tracks: Maximum number of simultaneous tracks
    gate_threshold: Mahalanobis distance gating threshold
    miss_prob: Probability of missed detection
    false_alarm_density: False alarm spatial density
    transition_sigma: Transition probability decay parameter
    min_hits_to_confirm: Hits needed to confirm track
    max_misses_to_delete: Consecutive misses to delete track
  """

  window_size: int = 5
  max_tracks: int = 32
  gate_threshold: float = 9.21  # Chi-squared 99% with 2 degrees of freedom
  miss_prob: float = 0.1
  false_alarm_density: float = 1e-6
  transition_sigma: float = 2.0
  min_hits_to_confirm: int = 3
  max_misses_to_delete: int = 5


class ViterbiTracker:
  """Multi-object tracker using Viterbi algorithm.

  Uses a Hidden Markov Model where:
  - Hidden states: Track-to-detection assignments
  - Observations: Detection measurements
  - Transition: Assignment consistency over time
  - Emission: Mahalanobis likelihood of measurement

  The Viterbi algorithm finds the globally optimal assignment
  sequence over the temporal window.

  Usage:
    config = ViterbiConfig(window_size=5)
    tracker = ViterbiTracker(config)

    # Update with new detections
    for frame_detections in detection_stream:
      tracks = tracker.update(frame_detections)
      for track in tracks:
        print(f"Track {track.id}: {track.state[:2]}")
  """

  def __init__(self, config: ViterbiConfig | None = None):
    """Initialize tracker.

    Args:
      config: Tracker configuration
    """
    self.config = config or ViterbiConfig()

    # Active tracks
    self.tracks: list[TrackState] = []

    # Detection history for temporal window
    self._detection_history: list[list[Detection]] = []

    # Assignment history
    self._assignment_history: list[np.ndarray] = []

    # Track ID counter
    self._next_id = 0

    # Frame counter
    self._frame_count = 0

  def update(
    self,
    detections: list[Detection],
    dt: float = 0.05,
  ) -> list[TrackState]:
    """Update tracks with new detections.

    Args:
      detections: List of new detections
      dt: Time step since last update

    Returns:
      List of confirmed tracks
    """
    self._frame_count += 1

    # Predict all tracks forward
    self._predict_tracks(dt)

    # Add detections to history
    self._detection_history.append(detections)
    if len(self._detection_history) > self.config.window_size:
      self._detection_history.pop(0)

    # Compute assignment using Viterbi if we have enough history
    if len(self._detection_history) >= 2:
      assignments = self._viterbi_decode()
    else:
      assignments = self._greedy_assign(detections)

    # Apply assignments to update tracks
    self._apply_assignments(detections, assignments)

    # Manage track lifecycle
    self._manage_tracks(detections)

    # Return confirmed tracks
    return [t for t in self.tracks if t.hits >= self.config.min_hits_to_confirm]

  def _predict_tracks(self, dt: float) -> None:
    """Predict track states forward in time.

    Uses constant velocity model.

    Args:
      dt: Time step
    """
    # State transition matrix for constant velocity
    n = 4  # [x, vx, y, vy]
    F = np.eye(n)
    F[0, 1] = dt  # x += vx * dt
    F[2, 3] = dt  # y += vy * dt

    # Process noise
    q = 0.1
    Q = (
      np.array(
        [
          [dt**3 / 3, dt**2 / 2, 0, 0],
          [dt**2 / 2, dt, 0, 0],
          [0, 0, dt**3 / 3, dt**2 / 2],
          [0, 0, dt**2 / 2, dt],
        ]
      )
      * q
    )

    for track in self.tracks:
      if len(track.state) >= n:
        track.state[:n] = F @ track.state[:n]
        track.covariance[:n, :n] = F @ track.covariance[:n, :n] @ F.T + Q

  def _greedy_assign(self, detections: list[Detection]) -> np.ndarray:
    """Greedy assignment for first frame.

    Args:
      detections: Current detections

    Returns:
      Assignment array (detection_idx -> track_idx, -1 for unassigned)
    """
    n_det = len(detections)
    assignments = np.full(n_det, -1, dtype=np.int32)

    if not self.tracks or n_det == 0:
      return assignments

    # Compute cost matrix
    cost_matrix = self._compute_cost_matrix(detections)

    # Greedy assignment
    used_tracks = set()
    for d_idx in range(n_det):
      best_t_idx = -1
      best_cost = self.config.gate_threshold

      for t_idx in range(len(self.tracks)):
        if t_idx in used_tracks:
          continue
        if cost_matrix[d_idx, t_idx] < best_cost:
          best_cost = cost_matrix[d_idx, t_idx]
          best_t_idx = t_idx

      if best_t_idx >= 0:
        assignments[d_idx] = best_t_idx
        used_tracks.add(best_t_idx)

    return assignments

  def _compute_cost_matrix(self, detections: list[Detection]) -> np.ndarray:
    """Compute cost matrix between detections and tracks.

    Uses Mahalanobis distance as cost.

    Args:
      detections: Current detections

    Returns:
      Cost matrix [n_detections x n_tracks]
    """
    n_det = len(detections)
    n_tracks = len(self.tracks)

    cost = np.full((n_det, n_tracks), np.inf)

    # Measurement matrix (observe x, y from state)
    H = np.array(
      [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
      ]
    )

    for d_idx, det in enumerate(detections):
      for t_idx, track in enumerate(self.tracks):
        # Predicted measurement
        z_pred = H @ track.state[:4]

        # Innovation covariance
        det_cov = det.covariance if det.covariance is not None else np.eye(2)
        S = H @ track.covariance[:4, :4] @ H.T + det_cov

        # Mahalanobis distance
        try:
          innovation = det.measurement[:2] - z_pred
          S_inv = np.linalg.inv(S)
          dist = mahalanobis_distance(innovation, np.zeros(2), S_inv)
          dist_sq = dist**2
        except (np.linalg.LinAlgError, ValueError):
          dist_sq = np.inf

        cost[d_idx, t_idx] = dist_sq

    return cost

  def _compute_transition_matrix(self) -> np.ndarray:
    """Compute HMM transition probability matrix.

    Models probability of track assignment transitions between frames.

    Returns:
      Transition matrix [n_states x n_states]
    """
    n_tracks = len(self.tracks) + 1  # +1 for "unassigned" state
    n_states = n_tracks

    # Initialize with small base probability
    trans = np.full((n_states, n_states), 0.01)

    # High probability of staying assigned to same track
    for i in range(n_states):
      trans[i, i] = 1.0 - self.config.miss_prob

    # Probability of becoming unassigned (missed detection)
    for i in range(n_states - 1):
      trans[i, n_states - 1] = self.config.miss_prob

    # Probability of new track (from unassigned)
    sigma = self.config.transition_sigma
    for j in range(n_states - 1):
      trans[n_states - 1, j] = np.exp(-j / sigma) * 0.1

    # Normalize rows
    trans = trans / trans.sum(axis=1, keepdims=True)

    return trans

  def _compute_emission_prob(
    self,
    detections: list[Detection],
  ) -> np.ndarray:
    """Compute HMM emission probabilities.

    Emission probability is the likelihood of observing a detection
    given track assignment.

    Args:
      detections: Current detections

    Returns:
      Emission probability matrix [n_detections x n_states]
    """
    n_det = len(detections)
    n_tracks = len(self.tracks)
    n_states = n_tracks + 1  # +1 for unassigned

    emission = np.zeros((n_det, n_states))

    # Cost matrix gives squared Mahalanobis distances
    cost = self._compute_cost_matrix(detections)

    for d_idx in range(n_det):
      for t_idx in range(n_tracks):
        dist_sq = cost[d_idx, t_idx]

        # Gating
        if dist_sq > self.config.gate_threshold:
          emission[d_idx, t_idx] = 1e-10
        else:
          # Gaussian likelihood (negative log likelihood ~ dist_sq/2)
          emission[d_idx, t_idx] = np.exp(-dist_sq / 2)

      # Unassigned state (false alarm)
      emission[d_idx, n_states - 1] = self.config.false_alarm_density

    # Normalize
    row_sums = emission.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    emission = emission / row_sums

    return emission

  def _viterbi_decode(self) -> np.ndarray:
    """Run Viterbi algorithm on detection history.

    Finds globally optimal assignment sequence over temporal window.

    Returns:
      Assignment array for most recent frame
    """
    if len(self._detection_history) < 2 or not self.tracks:
      if self._detection_history:
        return self._greedy_assign(self._detection_history[-1])
      return np.array([], dtype=np.int32)

    n_tracks = len(self.tracks)
    n_states = n_tracks + 1

    # Transition matrix
    trans = self._compute_transition_matrix()
    log_trans = np.log(trans + 1e-10)

    # Forward pass: compute Viterbi scores
    # For each detection in each frame, track the best path
    viterbi_scores: list[np.ndarray] = []
    backpointers: list[np.ndarray] = []

    for frame_idx, detections in enumerate(self._detection_history):
      n_det = len(detections)

      if n_det == 0:
        viterbi_scores.append(np.zeros((0, n_states)))
        backpointers.append(np.zeros((0, n_states), dtype=np.int32))
        continue

      emission = self._compute_emission_prob(detections)
      log_emission = np.log(emission + 1e-10)

      scores = np.zeros((n_det, n_states))
      bp = np.zeros((n_det, n_states), dtype=np.int32)

      if frame_idx == 0:
        # Initialize with emission only
        scores = log_emission
      else:
        prev_scores = viterbi_scores[-1]
        if prev_scores.size == 0:
          scores = log_emission
        else:
          # For each detection, find best previous state
          for d_idx in range(n_det):
            for s_idx in range(n_states):
              # Best score to reach state s_idx
              candidates = prev_scores.mean(axis=0) + log_trans[:, s_idx]
              best_prev = np.argmax(candidates)
              scores[d_idx, s_idx] = candidates[best_prev] + log_emission[d_idx, s_idx]
              bp[d_idx, s_idx] = best_prev

      viterbi_scores.append(scores)
      backpointers.append(bp)

    # Backward pass: trace best assignments
    final_scores = viterbi_scores[-1]
    if final_scores.size == 0:
      return np.array([], dtype=np.int32)

    # Find best state for each detection in final frame
    n_det_final = len(self._detection_history[-1])
    assignments = np.full(n_det_final, -1, dtype=np.int32)

    # Assign detections to tracks (avoiding duplicates)
    used_tracks = set()
    det_order = np.argsort(final_scores.max(axis=1))[::-1]

    for d_idx in det_order:
      best_state = np.argmax(final_scores[d_idx])

      # If it's a track (not unassigned state)
      if best_state < n_tracks and best_state not in used_tracks:
        # Check if within gate
        if final_scores[d_idx, best_state] > np.log(1e-5):
          assignments[d_idx] = best_state
          used_tracks.add(best_state)

    return assignments

  def _apply_assignments(
    self,
    detections: list[Detection],
    assignments: np.ndarray,
  ) -> None:
    """Apply assignments to update tracks.

    Args:
      detections: Current detections
      assignments: Assignment array (det_idx -> track_idx)
    """
    # Measurement matrix
    H = np.array(
      [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
      ]
    )

    # Track which tracks were updated
    updated_tracks = set()

    for d_idx, t_idx in enumerate(assignments):
      if t_idx < 0 or t_idx >= len(self.tracks):
        continue

      det = detections[d_idx]
      track = self.tracks[t_idx]

      # Kalman update
      det_cov = det.covariance if det.covariance is not None else np.eye(2)
      S = H @ track.covariance[:4, :4] @ H.T + det_cov
      K = track.covariance[:4, :4] @ H.T @ np.linalg.inv(S)

      innovation = det.measurement[:2] - H @ track.state[:4]
      track.state[:4] = track.state[:4] + K @ innovation
      track.covariance[:4, :4] = (np.eye(4) - K @ H) @ track.covariance[:4, :4]

      track.hits += 1
      track.misses = 0
      updated_tracks.add(t_idx)

    # Mark unassigned tracks as missed
    for t_idx, track in enumerate(self.tracks):
      if t_idx not in updated_tracks:
        track.misses += 1

  def _manage_tracks(self, detections: list[Detection]) -> None:
    """Manage track creation and deletion.

    Args:
      detections: Current detections
    """
    # Delete tracks with too many misses
    self.tracks = [t for t in self.tracks if t.misses < self.config.max_misses_to_delete]

    # Age all tracks
    for track in self.tracks:
      track.age += 1

    # Create new tracks for unassigned detections
    if len(self.tracks) < self.config.max_tracks:
      # Find unassigned detections
      assigned = set()
      for det_history in self._assignment_history[-1:]:
        assigned.update(d for d in det_history if d >= 0)

      current_assignments = self._greedy_assign(detections)
      for d_idx, t_idx in enumerate(current_assignments):
        if t_idx < 0 and len(self.tracks) < self.config.max_tracks:
          # Create new track
          det = detections[d_idx]
          state = np.zeros(4)
          state[0] = det.measurement[0]  # x
          state[2] = det.measurement[1] if len(det.measurement) > 1 else 0.0  # y

          cov = np.eye(4) * 10.0
          cov[0, 0] = 1.0  # x variance
          cov[2, 2] = 1.0  # y variance

          new_track = TrackState(
            id=self._next_id,
            state=state,
            covariance=cov,
            hits=1,  # Creation from detection counts as first hit
          )
          self._next_id += 1
          self.tracks.append(new_track)

  def get_all_tracks(self) -> list[TrackState]:
    """Get all tracks including unconfirmed ones.

    Returns:
      List of all tracks
    """
    return self.tracks.copy()

  def reset(self) -> None:
    """Reset tracker state."""
    self.tracks.clear()
    self._detection_history.clear()
    self._assignment_history.clear()
    self._frame_count = 0

  @property
  def frame_count(self) -> int:
    """Number of frames processed."""
    return self._frame_count
