"""Multi-target tracking comparison for openpilot vs Stone Soup."""
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.dataassociator.probability import JPDA
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.initiator.simple import MultiMeasurementInitiator, SimpleMeasurementInitiator
from stonesoup.measures import Mahalanobis
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.updater.kalman import KalmanUpdater


@dataclass
class GroundTruthTrack:
  """Ground truth for a single target."""
  track_id: int
  timestamps: np.ndarray
  positions: np.ndarray  # [n_steps, 2] for x, y
  velocities: np.ndarray  # [n_steps, 2] for vx, vy


@dataclass
class MultiTargetScenario:
  """Scenario with multiple targets."""
  name: str
  dt: float
  timestamps: np.ndarray
  ground_truth: list[GroundTruthTrack]
  detections_by_time: dict[float, list[Detection]]  # timestamp -> detections
  description: str = ""


@dataclass
class TrackingMetrics:
  """Multi-object tracking metrics."""
  tracker_name: str
  mota: float  # Multiple Object Tracking Accuracy
  motp: float  # Multiple Object Tracking Precision
  id_switches: int
  false_positives: int
  false_negatives: int
  n_ground_truth: int
  n_hypotheses: int


def create_highway_scenario(
  dt: float = 0.05,
  duration: float = 10.0,
  n_vehicles: int = 3,
  measurement_noise_std: float = 1.0,
  detection_probability: float = 0.95,
  seed: int = 42,
) -> MultiTargetScenario:
  """Create highway following scenario with multiple vehicles.

  Vehicles are ahead of ego, moving at slightly different speeds.
  """
  rng = np.random.default_rng(seed)
  n_steps = int(duration / dt)
  timestamps = np.arange(n_steps) * dt
  epoch = datetime(2000, 1, 1)

  ground_truth = []
  detections_by_time: dict[float, list[Detection]] = {t: [] for t in timestamps}

  # Create vehicles at different distances and speeds
  for i in range(n_vehicles):
    initial_x = 30.0 + i * 25.0  # 30m, 55m, 80m ahead
    initial_y = (i - 1) * 2.0  # Spread across lanes: -2m, 0m, 2m
    velocity_x = -3.0 - i * 1.0  # Relative velocities: -3, -4, -5 m/s

    # Generate ground truth trajectory
    positions = np.zeros((n_steps, 2))
    velocities = np.zeros((n_steps, 2))

    for step in range(n_steps):
      t = timestamps[step]
      positions[step, 0] = initial_x + velocity_x * t
      positions[step, 1] = initial_y
      velocities[step, 0] = velocity_x
      velocities[step, 1] = 0.0

    ground_truth.append(GroundTruthTrack(
      track_id=i,
      timestamps=timestamps,
      positions=positions,
      velocities=velocities,
    ))

    # Generate detections with noise and missed detections
    for step, t in enumerate(timestamps):
      if rng.random() < detection_probability:
        noisy_pos = positions[step] + rng.normal(0, measurement_noise_std, 2)
        detection = Detection(
          state_vector=StateVector([noisy_pos[0], noisy_pos[1]]),
          timestamp=epoch + timedelta(seconds=t),
          metadata={'ground_truth_id': i},
        )
        detections_by_time[t].append(detection)

  return MultiTargetScenario(
    name="highway_following",
    dt=dt,
    timestamps=timestamps,
    ground_truth=ground_truth,
    detections_by_time=detections_by_time,
    description=f"{n_vehicles} vehicles, detection_prob={detection_probability}",
  )


def create_cut_in_scenario(
  dt: float = 0.05,
  duration: float = 10.0,
  measurement_noise_std: float = 1.0,
  detection_probability: float = 0.95,
  seed: int = 42,
) -> MultiTargetScenario:
  """Create cut-in scenario where a vehicle enters from adjacent lane."""
  rng = np.random.default_rng(seed)
  n_steps = int(duration / dt)
  timestamps = np.arange(n_steps) * dt
  epoch = datetime(2000, 1, 1)

  ground_truth = []
  detections_by_time: dict[float, list[Detection]] = {t: [] for t in timestamps}

  # Lead vehicle (constant in lane)
  lead_x, lead_y = 50.0, 0.0
  lead_vx = -2.0

  positions_lead = np.zeros((n_steps, 2))
  velocities_lead = np.zeros((n_steps, 2))
  for step, t in enumerate(timestamps):
    positions_lead[step] = [lead_x + lead_vx * t, lead_y]
    velocities_lead[step] = [lead_vx, 0.0]

  ground_truth.append(GroundTruthTrack(
    track_id=0,
    timestamps=timestamps,
    positions=positions_lead,
    velocities=velocities_lead,
  ))

  # Cut-in vehicle (starts in adjacent lane, moves into ego lane)
  cutin_start_x, cutin_start_y = 40.0, 3.5  # Adjacent lane
  cutin_vx = -4.0  # Faster than lead
  cutin_time = 3.0  # Starts cut-in at t=3s
  cutin_duration = 2.0  # Takes 2s to complete

  positions_cutin = np.zeros((n_steps, 2))
  velocities_cutin = np.zeros((n_steps, 2))
  for step, t in enumerate(timestamps):
    x = cutin_start_x + cutin_vx * t
    if t < cutin_time:
      y = cutin_start_y
      vy = 0.0
    elif t < cutin_time + cutin_duration:
      # Linear interpolation during cut-in
      progress = (t - cutin_time) / cutin_duration
      y = cutin_start_y * (1 - progress)
      vy = -cutin_start_y / cutin_duration
    else:
      y = 0.0
      vy = 0.0
    positions_cutin[step] = [x, y]
    velocities_cutin[step] = [cutin_vx, vy]

  ground_truth.append(GroundTruthTrack(
    track_id=1,
    timestamps=timestamps,
    positions=positions_cutin,
    velocities=velocities_cutin,
  ))

  # Generate detections
  for gt in ground_truth:
    for step, t in enumerate(timestamps):
      if rng.random() < detection_probability:
        noisy_pos = gt.positions[step] + rng.normal(0, measurement_noise_std, 2)
        detection = Detection(
          state_vector=StateVector([noisy_pos[0], noisy_pos[1]]),
          timestamp=epoch + timedelta(seconds=t),
          metadata={'ground_truth_id': gt.track_id},
        )
        detections_by_time[t].append(detection)

  return MultiTargetScenario(
    name="cut_in",
    dt=dt,
    timestamps=timestamps,
    ground_truth=ground_truth,
    detections_by_time=detections_by_time,
    description="Lead vehicle + cut-in vehicle",
  )


class MultiTargetTrackerWrapper:
  """Base class for multi-target tracker wrappers."""

  def __init__(self, dt: float):
    self.dt = dt
    self.epoch = datetime(2000, 1, 1)
    self.tracks: list[Track] = []

  def process_detections(self, timestamp: float, detections: list[Detection]) -> list[Track]:
    """Process detections at a single timestamp and return current tracks."""
    raise NotImplementedError


class JPDATrackerWrapper(MultiTargetTrackerWrapper):
  """Wrapper for Stone Soup JPDA tracker."""

  def __init__(self, dt: float, prob_detect: float = 0.95, clutter_density: float = 1e-6):
    super().__init__(dt)

    # Models
    q = 5.0  # process noise
    self.transition_model = CombinedLinearGaussianTransitionModel([
      ConstantVelocity(q),
      ConstantVelocity(q),
    ])

    r = 1.0  # measurement noise
    self.measurement_model = LinearGaussian(
      ndim_state=4,
      mapping=(0, 2),  # Observe x and y positions
      noise_covar=np.diag([r, r]),
    )

    # Predictor and updater
    self.predictor = KalmanPredictor(self.transition_model)
    self.updater = KalmanUpdater(self.measurement_model)

    # JPDA hypothesiser
    self.hypothesiser = PDAHypothesiser(
      predictor=self.predictor,
      updater=self.updater,
      clutter_spatial_density=clutter_density,
      prob_detect=prob_detect,
    )

    # JPDA data associator
    self.data_associator = JPDA(hypothesiser=self.hypothesiser)

    # Simple track initiator - creates track from each unassociated detection
    self.initiator = SimpleMeasurementInitiator(
      prior_state=GaussianState(
        state_vector=StateVector([0, 0, 0, 0]),
        covar=CovarianceMatrix(np.diag([100, 10, 100, 10])),
      ),
      measurement_model=self.measurement_model,
    )

    # Track deleter
    self.deleter = UpdateTimeStepsDeleter(time_steps_since_update=5)

    self.tracks = set()
    self.associated_detections = set()

  def process_detections(self, timestamp: float, detections: list[Detection]) -> list[Track]:
    """Process detections and return current tracks."""
    time = self.epoch + timedelta(seconds=timestamp)

    # Update detection timestamps
    for det in detections:
      det.timestamp = time

    detection_set = set(detections)
    self.associated_detections = set()

    # Associate detections with tracks
    if self.tracks:
      hypotheses = self.data_associator.associate(self.tracks, detection_set, time)

      # Update tracks - JPDA returns MultipleHypothesis, need to handle specially
      for track, hyp in hypotheses.items():
        # Get the best single hypothesis from the multiple hypotheses
        if hasattr(hyp, 'single_hypotheses') and hyp.single_hypotheses:
          # Find best hypothesis by weight
          best_hyp = max(hyp.single_hypotheses, key=lambda h: h.weight if hasattr(h, 'weight') else 0)
          if best_hyp.measurement:
            post = self.updater.update(best_hyp)
            track.append(post)
            self.associated_detections.add(best_hyp.measurement)
          else:
            pred = self.predictor.predict(track.state, timestamp=time)
            track.append(pred)
        elif hyp.measurement:
          post = self.updater.update(hyp)
          track.append(post)
          self.associated_detections.add(hyp.measurement)
        else:
          pred = self.predictor.predict(track.state, timestamp=time)
          track.append(pred)

    # Initiate new tracks from unassociated detections
    unassociated = detection_set - self.associated_detections
    new_tracks = self.initiator.initiate(unassociated, time)
    self.tracks |= new_tracks

    # Delete old tracks
    self.tracks -= self.deleter.delete_tracks(self.tracks)

    return list(self.tracks)


class GNNTrackerWrapper(MultiTargetTrackerWrapper):
  """Wrapper for Stone Soup GNN (Global Nearest Neighbor) tracker."""

  def __init__(self, dt: float, gate_threshold: float = 10.0):
    super().__init__(dt)

    # Models (same as JPDA)
    q = 5.0
    self.transition_model = CombinedLinearGaussianTransitionModel([
      ConstantVelocity(q),
      ConstantVelocity(q),
    ])

    r = 1.0
    self.measurement_model = LinearGaussian(
      ndim_state=4,
      mapping=(0, 2),
      noise_covar=np.diag([r, r]),
    )

    self.predictor = KalmanPredictor(self.transition_model)
    self.updater = KalmanUpdater(self.measurement_model)

    # GNN with Mahalanobis distance
    self.hypothesiser = DistanceHypothesiser(
      predictor=self.predictor,
      updater=self.updater,
      measure=Mahalanobis(),
      missed_distance=gate_threshold,
    )

    self.data_associator = GNNWith2DAssignment(hypothesiser=self.hypothesiser)

    # Track initiator
    self.initiator = MultiMeasurementInitiator(
      prior_state=GaussianState(
        state_vector=StateVector([0, 0, 0, 0]),
        covar=CovarianceMatrix(np.diag([100, 10, 100, 10])),
      ),
      measurement_model=self.measurement_model,
      deleter=UpdateTimeStepsDeleter(time_steps_since_update=3),
      data_associator=self.data_associator,
      updater=self.updater,
      min_points=2,
    )

    self.deleter = UpdateTimeStepsDeleter(time_steps_since_update=5)
    self.tracks = set()

  def process_detections(self, timestamp: float, detections: list[Detection]) -> list[Track]:
    """Process detections and return current tracks."""
    time = self.epoch + timedelta(seconds=timestamp)

    for det in detections:
      det.timestamp = time

    if self.tracks:
      hypotheses = self.data_associator.associate(self.tracks, detections, time)

      for track, hyp in hypotheses.items():
        if hyp.measurement:
          post = self.updater.update(hyp)
          track.append(post)
        else:
          pred = self.predictor.predict(track.state, timestamp=time)
          track.append(pred)

    new_tracks = self.initiator.initiate(set(detections), time)
    self.tracks |= new_tracks
    self.tracks -= self.deleter.delete_tracks(self.tracks)

    return list(self.tracks)


def compute_tracking_metrics(
  tracker_name: str,
  tracks: list[list[Track]],  # tracks at each timestep
  scenario: MultiTargetScenario,
  distance_threshold: float = 5.0,
) -> TrackingMetrics:
  """Compute MOTA, MOTP, and ID switches.

  MOTA = 1 - (FN + FP + ID_switches) / n_ground_truth
  MOTP = sum(distances) / n_matches
  """
  total_fn = 0  # False negatives (missed detections)
  total_fp = 0  # False positives
  id_switches = 0
  total_distance = 0.0
  total_matches = 0
  total_gt = 0

  # Track ID mappings from previous frame
  prev_gt_to_track: dict[int, int] = {}

  for step, _t in enumerate(scenario.timestamps):
    current_tracks = tracks[step] if step < len(tracks) else []

    # Get ground truth positions at this timestep
    gt_positions = {}
    for gt in scenario.ground_truth:
      gt_positions[gt.track_id] = gt.positions[step]

    total_gt += len(gt_positions)

    # Match tracks to ground truth using Hungarian algorithm (greedy here for simplicity)
    matched_gt = set()
    matched_tracks = set()
    gt_to_track: dict[int, int] = {}

    for track_idx, track in enumerate(current_tracks):
      if not track:
        continue
      track_pos = np.array([track.state.state_vector[0], track.state.state_vector[2]])

      best_gt_id = None
      best_dist = float('inf')

      for gt_id, gt_pos in gt_positions.items():
        if gt_id in matched_gt:
          continue
        dist = np.linalg.norm(track_pos - gt_pos)
        if dist < best_dist and dist < distance_threshold:
          best_dist = dist
          best_gt_id = gt_id

      if best_gt_id is not None:
        matched_gt.add(best_gt_id)
        matched_tracks.add(track_idx)
        gt_to_track[best_gt_id] = track_idx
        total_distance += best_dist
        total_matches += 1

    # Count false negatives (unmatched ground truth)
    total_fn += len(gt_positions) - len(matched_gt)

    # Count false positives (unmatched tracks)
    total_fp += len(current_tracks) - len(matched_tracks)

    # Count ID switches
    for gt_id, track_idx in gt_to_track.items():
      if gt_id in prev_gt_to_track and prev_gt_to_track[gt_id] != track_idx:
        id_switches += 1

    prev_gt_to_track = gt_to_track

  # Compute MOTA and MOTP
  mota = 1.0 - (total_fn + total_fp + id_switches) / max(total_gt, 1)
  motp = total_distance / max(total_matches, 1)

  return TrackingMetrics(
    tracker_name=tracker_name,
    mota=mota,
    motp=motp,
    id_switches=id_switches,
    false_positives=total_fp,
    false_negatives=total_fn,
    n_ground_truth=total_gt,
    n_hypotheses=total_matches,
  )


def run_tracker_on_scenario(
  tracker: MultiTargetTrackerWrapper,
  scenario: MultiTargetScenario,
) -> list[list[Track]]:
  """Run tracker on scenario and return tracks at each timestep."""
  all_tracks = []

  for t in scenario.timestamps:
    detections = scenario.detections_by_time.get(t, [])
    tracks = tracker.process_detections(t, detections)
    all_tracks.append(list(tracks))

  return all_tracks


def compare_multi_target_trackers(
  scenario: MultiTargetScenario,
) -> dict[str, TrackingMetrics]:
  """Compare JPDA and GNN trackers on a scenario."""
  results = {}

  # JPDA tracker
  jpda = JPDATrackerWrapper(dt=scenario.dt)
  jpda_tracks = run_tracker_on_scenario(jpda, scenario)
  results["JPDA"] = compute_tracking_metrics("JPDA", jpda_tracks, scenario)

  # GNN tracker
  gnn = GNNTrackerWrapper(dt=scenario.dt)
  gnn_tracks = run_tracker_on_scenario(gnn, scenario)
  results["GNN"] = compute_tracking_metrics("GNN", gnn_tracks, scenario)

  return results


def format_tracking_report(
  scenario: MultiTargetScenario,
  metrics: dict[str, TrackingMetrics],
) -> str:
  """Format tracking comparison results as markdown."""
  lines = [
    "# Multi-Target Tracking Comparison",
    "",
    f"## Scenario: {scenario.name}",
    f"{scenario.description}",
    "",
    f"- Duration: {scenario.timestamps[-1]:.1f}s",
    f"- Time step: {scenario.dt*1000:.0f}ms",
    f"- Ground truth tracks: {len(scenario.ground_truth)}",
    "",
    "## Results",
    "",
    "| Tracker | MOTA | MOTP (m) | ID Switches | FP | FN |",
    "|---------|------|----------|-------------|----|----|",
  ]

  for name, m in sorted(metrics.items()):
    lines.append(
      f"| {name} | {m.mota:.3f} | {m.motp:.3f} | {m.id_switches} | "
      + f"{m.false_positives} | {m.false_negatives} |"
    )

  lines.extend([
    "",
    "MOTA = Multiple Object Tracking Accuracy (higher is better)",
    "MOTP = Multiple Object Tracking Precision in meters (lower is better)",
  ])

  return "\n".join(lines)
