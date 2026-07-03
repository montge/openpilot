"""Track-to-Track Fusion using Covariance Intersection.

Implements sensor fusion for combining radar and vision tracks with
unknown cross-correlation between estimates.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import minimize_scalar


class TrackState(Protocol):
  """Protocol for track states that can be fused."""
  position: np.ndarray
  velocity: np.ndarray
  covariance: np.ndarray


@dataclass
class FusedState:
  """Result of track fusion."""
  position: np.ndarray
  velocity: np.ndarray
  covariance: np.ndarray
  omega: float  # Optimal fusion weight


@dataclass
class SensorTrack:
  """Track estimate from a single sensor."""
  sensor_name: str
  position: np.ndarray  # [x, y] or [x, y, z]
  velocity: np.ndarray  # [vx, vy] or [vx, vy, vz]
  covariance: np.ndarray  # Full state covariance
  timestamp: float


@dataclass
class RadarTrack(SensorTrack):
  """Track from radar sensor."""
  range_rate: float = 0.0
  rcs: float = 0.0  # Radar cross-section


@dataclass
class VisionTrack(SensorTrack):
  """Track from vision/camera sensor."""
  confidence: float = 1.0
  bbox: tuple[float, float, float, float] = (0, 0, 0, 0)  # x, y, w, h


def covariance_intersection(
  x1: np.ndarray,
  P1: np.ndarray,
  x2: np.ndarray,
  P2: np.ndarray,
  optimize: str = "trace"
) -> tuple[np.ndarray, np.ndarray, float]:
  """Fuse two estimates using Covariance Intersection.

  CI is optimal for fusing estimates with unknown correlation.
  It provides a consistent (non-overconfident) fused estimate.

  Args:
    x1: First state estimate
    P1: First covariance matrix
    x2: Second state estimate
    P2: Second covariance matrix
    optimize: "trace" or "det" - optimization criterion

  Returns:
    x_fused: Fused state estimate
    P_fused: Fused covariance matrix
    omega: Optimal fusion weight [0, 1]
  """
  def compute_fused(omega: float) -> tuple[np.ndarray, np.ndarray]:
    P1_inv = np.linalg.inv(P1)
    P2_inv = np.linalg.inv(P2)

    P_fused_inv = omega * P1_inv + (1 - omega) * P2_inv
    P_fused = np.linalg.inv(P_fused_inv)

    x_fused = P_fused @ (omega * P1_inv @ x1 + (1 - omega) * P2_inv @ x2)
    return x_fused, P_fused

  def objective(omega: float) -> float:
    _, P_fused = compute_fused(omega)
    if optimize == "trace":
      return np.trace(P_fused)
    else:  # det
      return np.linalg.det(P_fused)

  result = minimize_scalar(objective, bounds=(0.01, 0.99), method='bounded')
  omega_opt = result.x

  x_fused, P_fused = compute_fused(omega_opt)
  return x_fused, P_fused, omega_opt


def fast_covariance_intersection(
  x1: np.ndarray,
  P1: np.ndarray,
  x2: np.ndarray,
  P2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
  """Fast approximate CI using closed-form solution.

  Uses trace-based omega approximation that's faster than optimization.
  """
  trace1 = np.trace(P1)
  trace2 = np.trace(P2)

  omega = trace2 / (trace1 + trace2)
  omega = np.clip(omega, 0.01, 0.99)

  P1_inv = np.linalg.inv(P1)
  P2_inv = np.linalg.inv(P2)

  P_fused_inv = omega * P1_inv + (1 - omega) * P2_inv
  P_fused = np.linalg.inv(P_fused_inv)

  x_fused = P_fused @ (omega * P1_inv @ x1 + (1 - omega) * P2_inv @ x2)

  return x_fused, P_fused, omega


@dataclass
class FusionScenario:
  """Test scenario for track-to-track fusion."""
  name: str
  dt: float
  timestamps: np.ndarray
  ground_truth_position: np.ndarray
  ground_truth_velocity: np.ndarray
  radar_tracks: list[RadarTrack]
  vision_tracks: list[VisionTrack]


@dataclass
class FusionResult:
  """Result of fusing tracks at a timestep."""
  timestamp: float
  fused_state: FusedState
  radar_track: RadarTrack | None
  vision_track: VisionTrack | None


@dataclass
class FusionMetrics:
  """Metrics for evaluating fusion performance."""
  method_name: str
  rmse_position: float
  rmse_velocity: float
  mean_omega: float
  fusion_rate: float  # Fraction of timesteps with fusion
  n_samples: int


class TrackFusionEngine:
  """Engine for fusing radar and vision tracks."""

  def __init__(self, use_fast_ci: bool = False):
    self.use_fast_ci = use_fast_ci
    self.fusion_results: list[FusionResult] = []

  def fuse_tracks(
    self,
    radar: RadarTrack | None,
    vision: VisionTrack | None
  ) -> FusedState | None:
    """Fuse radar and vision tracks at a single timestep."""
    if radar is None and vision is None:
      return None

    if radar is None:
      # Vision only
      return FusedState(
        position=vision.position,
        velocity=vision.velocity,
        covariance=vision.covariance,
        omega=0.0
      )

    if vision is None:
      # Radar only
      return FusedState(
        position=radar.position,
        velocity=radar.velocity,
        covariance=radar.covariance,
        omega=1.0
      )

    # Both available - use CI
    radar_state = np.concatenate([radar.position, radar.velocity])
    vision_state = np.concatenate([vision.position, vision.velocity])

    ci_func = fast_covariance_intersection if self.use_fast_ci else covariance_intersection

    fused_state, fused_cov, omega = ci_func(
      radar_state, radar.covariance,
      vision_state, vision.covariance
    )

    dim = len(radar.position)
    return FusedState(
      position=fused_state[:dim],
      velocity=fused_state[dim:],
      covariance=fused_cov,
      omega=omega
    )

  def process_scenario(self, scenario: FusionScenario) -> list[FusionResult]:
    """Process a fusion scenario, returning results at each timestep."""
    self.fusion_results = []

    for i, t in enumerate(scenario.timestamps):
      radar = scenario.radar_tracks[i] if i < len(scenario.radar_tracks) else None
      vision = scenario.vision_tracks[i] if i < len(scenario.vision_tracks) else None

      fused = self.fuse_tracks(radar, vision)
      if fused is not None:
        self.fusion_results.append(FusionResult(
          timestamp=t,
          fused_state=fused,
          radar_track=radar,
          vision_track=vision
        ))

    return self.fusion_results


def create_fusion_scenario(
  dt: float = 0.05,
  duration: float = 5.0,
  radar_noise_std: float = 0.5,
  vision_noise_std: float = 1.0,
  radar_dropout_prob: float = 0.1,
  vision_dropout_prob: float = 0.2,
  seed: int | None = None
) -> FusionScenario:
  """Create a test scenario with radar and vision tracks.

  Simulates a lead vehicle with noisy sensor measurements.
  """
  if seed is not None:
    np.random.seed(seed)

  n_steps = int(duration / dt)
  timestamps = np.arange(n_steps) * dt

  # Ground truth: vehicle decelerating
  initial_pos = np.array([50.0, 0.0])  # x=50m ahead, y=0 lateral
  initial_vel = np.array([-5.0, 0.0])  # Approaching at 5 m/s
  accel = np.array([2.0, 0.0])  # Decelerating (relative to ego)

  gt_positions = np.zeros((n_steps, 2))
  gt_velocities = np.zeros((n_steps, 2))

  for i in range(n_steps):
    t = timestamps[i]
    gt_positions[i] = initial_pos + initial_vel * t + 0.5 * accel * t**2
    gt_velocities[i] = initial_vel + accel * t

  # Generate noisy sensor tracks
  radar_tracks = []
  vision_tracks = []

  radar_cov = np.diag([radar_noise_std**2, radar_noise_std**2,
                       (radar_noise_std * 0.5)**2, (radar_noise_std * 0.5)**2])
  vision_cov = np.diag([vision_noise_std**2, vision_noise_std**2,
                        vision_noise_std**2, vision_noise_std**2])

  for i in range(n_steps):
    t = timestamps[i]
    gt_pos = gt_positions[i]
    gt_vel = gt_velocities[i]

    # Radar track (with dropout)
    if np.random.random() > radar_dropout_prob:
      pos_noise = np.random.randn(2) * radar_noise_std
      vel_noise = np.random.randn(2) * radar_noise_std * 0.5
      radar_tracks.append(RadarTrack(
        sensor_name="radar",
        position=gt_pos + pos_noise,
        velocity=gt_vel + vel_noise,
        covariance=radar_cov,
        timestamp=t,
        range_rate=np.linalg.norm(gt_vel) + np.random.randn() * 0.1
      ))
    else:
      radar_tracks.append(None)

    # Vision track (with dropout)
    if np.random.random() > vision_dropout_prob:
      pos_noise = np.random.randn(2) * vision_noise_std
      vel_noise = np.random.randn(2) * vision_noise_std
      vision_tracks.append(VisionTrack(
        sensor_name="vision",
        position=gt_pos + pos_noise,
        velocity=gt_vel + vel_noise,
        covariance=vision_cov,
        timestamp=t,
        confidence=0.8 + np.random.random() * 0.2
      ))
    else:
      vision_tracks.append(None)

  return FusionScenario(
    name="radar_vision_fusion",
    dt=dt,
    timestamps=timestamps,
    ground_truth_position=gt_positions,
    ground_truth_velocity=gt_velocities,
    radar_tracks=radar_tracks,
    vision_tracks=vision_tracks
  )


def create_occlusion_scenario(
  dt: float = 0.05,
  duration: float = 5.0,
  occlusion_start: float = 2.0,
  occlusion_duration: float = 1.0,
  seed: int | None = None
) -> FusionScenario:
  """Create scenario where vision is occluded for part of the time.

  Radar continues to track during occlusion but vision drops out.
  """
  if seed is not None:
    np.random.seed(seed)

  n_steps = int(duration / dt)
  timestamps = np.arange(n_steps) * dt

  # Ground truth
  initial_pos = np.array([40.0, 0.0])
  initial_vel = np.array([-3.0, 0.0])

  gt_positions = np.zeros((n_steps, 2))
  gt_velocities = np.zeros((n_steps, 2))

  for i in range(n_steps):
    t = timestamps[i]
    gt_positions[i] = initial_pos + initial_vel * t
    gt_velocities[i] = initial_vel

  # Sensor tracks
  radar_tracks = []
  vision_tracks = []

  radar_cov = np.diag([0.3**2, 0.3**2, 0.1**2, 0.1**2])
  vision_cov = np.diag([0.8**2, 0.8**2, 0.5**2, 0.5**2])

  for i in range(n_steps):
    t = timestamps[i]
    gt_pos = gt_positions[i]
    gt_vel = gt_velocities[i]

    # Radar always available (with small noise)
    pos_noise = np.random.randn(2) * 0.3
    vel_noise = np.random.randn(2) * 0.1
    radar_tracks.append(RadarTrack(
      sensor_name="radar",
      position=gt_pos + pos_noise,
      velocity=gt_vel + vel_noise,
      covariance=radar_cov,
      timestamp=t
    ))

    # Vision drops out during occlusion
    in_occlusion = occlusion_start <= t < (occlusion_start + occlusion_duration)
    if not in_occlusion:
      pos_noise = np.random.randn(2) * 0.8
      vel_noise = np.random.randn(2) * 0.5
      vision_tracks.append(VisionTrack(
        sensor_name="vision",
        position=gt_pos + pos_noise,
        velocity=gt_vel + vel_noise,
        covariance=vision_cov,
        timestamp=t
      ))
    else:
      vision_tracks.append(None)

  return FusionScenario(
    name="occlusion",
    dt=dt,
    timestamps=timestamps,
    ground_truth_position=gt_positions,
    ground_truth_velocity=gt_velocities,
    radar_tracks=radar_tracks,
    vision_tracks=vision_tracks
  )


def compute_fusion_metrics(
  method_name: str,
  results: list[FusionResult],
  scenario: FusionScenario
) -> FusionMetrics:
  """Compute accuracy metrics for fusion results."""
  position_errors = []
  velocity_errors = []
  omegas = []

  for result in results:
    # Find corresponding ground truth
    idx = np.searchsorted(scenario.timestamps, result.timestamp)
    if idx >= len(scenario.timestamps):
      idx = len(scenario.timestamps) - 1

    gt_pos = scenario.ground_truth_position[idx]
    gt_vel = scenario.ground_truth_velocity[idx]

    pos_err = np.linalg.norm(result.fused_state.position - gt_pos)
    vel_err = np.linalg.norm(result.fused_state.velocity - gt_vel)

    position_errors.append(pos_err)
    velocity_errors.append(vel_err)
    omegas.append(result.fused_state.omega)

  n_fused = sum(1 for r in results
                if r.radar_track is not None and r.vision_track is not None)

  return FusionMetrics(
    method_name=method_name,
    rmse_position=np.sqrt(np.mean(np.array(position_errors)**2)),
    rmse_velocity=np.sqrt(np.mean(np.array(velocity_errors)**2)),
    mean_omega=np.mean(omegas),
    fusion_rate=n_fused / len(results) if results else 0.0,
    n_samples=len(results)
  )


def compare_fusion_methods(scenario: FusionScenario) -> dict[str, FusionMetrics]:
  """Compare different fusion methods on a scenario."""
  methods = {
    "CI_Optimized": TrackFusionEngine(use_fast_ci=False),
    "CI_Fast": TrackFusionEngine(use_fast_ci=True),
    "Radar_Only": None,  # Special case
    "Vision_Only": None,  # Special case
  }

  metrics = {}

  for name, engine in methods.items():
    if name == "Radar_Only":
      # Use only radar tracks
      results = []
      for i, t in enumerate(scenario.timestamps):
        radar = scenario.radar_tracks[i] if i < len(scenario.radar_tracks) else None
        if radar is not None:
          results.append(FusionResult(
            timestamp=t,
            fused_state=FusedState(
              position=radar.position,
              velocity=radar.velocity,
              covariance=radar.covariance,
              omega=1.0
            ),
            radar_track=radar,
            vision_track=None
          ))
    elif name == "Vision_Only":
      # Use only vision tracks
      results = []
      for i, t in enumerate(scenario.timestamps):
        vision = scenario.vision_tracks[i] if i < len(scenario.vision_tracks) else None
        if vision is not None:
          results.append(FusionResult(
            timestamp=t,
            fused_state=FusedState(
              position=vision.position,
              velocity=vision.velocity,
              covariance=vision.covariance,
              omega=0.0
            ),
            radar_track=None,
            vision_track=vision
          ))
    else:
      results = engine.process_scenario(scenario)

    metrics[name] = compute_fusion_metrics(name, results, scenario)

  return metrics


def format_fusion_report(
  scenario: FusionScenario,
  metrics: dict[str, FusionMetrics]
) -> str:
  """Format fusion comparison as markdown report."""
  lines = [
    "# Track-to-Track Fusion Comparison",
    "",
    f"## Scenario: {scenario.name}",
    f"- Duration: {scenario.timestamps[-1]:.1f}s",
    f"- Time step: {int(scenario.dt * 1000)}ms",
    "",
    "## Results",
    "",
    "| Method | RMSE Pos (m) | RMSE Vel (m/s) | Mean ω | Fusion Rate | Samples |",
    "|--------|-------------|----------------|--------|-------------|---------|",
  ]

  for name, m in sorted(metrics.items(), key=lambda x: x[1].rmse_position):
    lines.append(
      f"| {name} | {m.rmse_position:.3f} | {m.rmse_velocity:.3f} | "
      + f"{m.mean_omega:.3f} | {m.fusion_rate:.2%} | {m.n_samples} |"
    )

  lines.extend([
    "",
    "ω = weight given to radar (1.0 = radar only, 0.0 = vision only)",
    "RMSE = Root Mean Square Error vs ground truth",
    ""
  ])

  return "\n".join(lines)


if __name__ == "__main__":
  # Demo the fusion comparison
  scenario = create_fusion_scenario(seed=42)
  metrics = compare_fusion_methods(scenario)
  report = format_fusion_report(scenario, metrics)
  print(report)
