"""Comparison harness for openpilot KF1D vs Stone Soup filters."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.predictor.kalman import (
  CubatureKalmanPredictor,
  ExtendedKalmanPredictor,
  KalmanPredictor,
  UnscentedKalmanPredictor,
)
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.types.array import CovarianceMatrix, StateVector, StateVectors
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import GaussianState, ParticleState
from stonesoup.updater.kalman import (
  CubatureKalmanUpdater,
  ExtendedKalmanUpdater,
  KalmanUpdater,
  UnscentedKalmanUpdater,
)
from stonesoup.updater.particle import ParticleUpdater

from openpilot.common.simple_kalman import KF1D


@dataclass
class FilterResult:
  """Result from a single filter update."""
  timestamp: float
  state: np.ndarray  # [position, velocity]
  measurement: float | None = None
  ground_truth: np.ndarray | None = None


@dataclass
class ComparisonMetrics:
  """Metrics comparing filter performance."""
  filter_name: str
  rmse_position: float
  rmse_velocity: float
  mae_position: float
  mae_velocity: float
  max_error_position: float
  max_error_velocity: float
  n_samples: int


class FilterWrapper(ABC):
  """Abstract base class for filter wrappers."""

  @abstractmethod
  def predict(self, dt: float) -> np.ndarray:
    """Predict state forward by dt seconds."""

  @abstractmethod
  def update(self, measurement: float) -> np.ndarray:
    """Update state with measurement."""

  @abstractmethod
  def get_state(self) -> np.ndarray:
    """Get current state as [position, velocity]."""

  @abstractmethod
  def reset(self, x0: np.ndarray) -> None:
    """Reset filter to initial state."""


class KF1DWrapper(FilterWrapper):
  """Wrapper for openpilot's KF1D filter."""

  def __init__(self, dt: float, x0: np.ndarray | None = None):
    self.dt = dt
    self.x0 = x0 if x0 is not None else np.array([[0.0], [0.0]])

    # Standard KF1D parameters from radard.py
    self.A = [[1.0, dt], [0.0, 1.0]]
    self.C = [1.0, 0.0]

    # Pre-computed Kalman gain (from radard.py lookup table, interpolated for dt)
    dts = [i * 0.01 for i in range(1, 21)]
    K0 = [0.12287673, 0.14556536, 0.16522756, 0.18281627, 0.1988689, 0.21372394,
          0.22761098, 0.24069424, 0.253096, 0.26491023, 0.27621103, 0.28705801,
          0.29750003, 0.30757767, 0.31732515, 0.32677158, 0.33594201, 0.34485814,
          0.35353899, 0.36200124]
    K1 = [0.29666309, 0.29330885, 0.29042818, 0.28787125, 0.28555364, 0.28342219,
          0.28144091, 0.27958406, 0.27783249, 0.27617149, 0.27458948, 0.27307714,
          0.27162685, 0.27023228, 0.26888809, 0.26758976, 0.26633338, 0.26511557,
          0.26393339, 0.26278425]
    self.K = [[np.interp(dt, dts, K0)], [np.interp(dt, dts, K1)]]

    self.kf = KF1D(self.x0.tolist(), self.A, self.C, self.K)

  def predict(self, dt: float) -> np.ndarray:
    # KF1D combines predict+update, so predict is implicit
    return self.get_state()

  def update(self, measurement: float) -> np.ndarray:
    self.kf.update(measurement)
    return self.get_state()

  def get_state(self) -> np.ndarray:
    return np.array([self.kf.x[0][0], self.kf.x[1][0]])

  def reset(self, x0: np.ndarray) -> None:
    self.kf.set_x([[x0[0]], [x0[1]]])


class StoneSoupKalmanWrapper(FilterWrapper):
  """Wrapper for Stone Soup Kalman filter."""

  def __init__(self, dt: float, x0: np.ndarray | None = None, filter_type: str = "kalman"):
    self.dt = dt
    self.filter_type = filter_type
    self.epoch = datetime(2000, 1, 1)
    self.current_time = 0.0

    # Process noise (tuned to match KF1D behavior)
    q = 10.0  # process noise intensity
    self.transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q)])

    # Measurement noise
    r = 1000.0  # measurement noise variance (matches KF1D R=1e3)
    self.measurement_model = LinearGaussian(
      ndim_state=2,
      mapping=(0,),
      noise_covar=np.array([[r]])
    )

    # Create predictor and updater based on filter type
    if filter_type == "kalman":
      self.predictor = KalmanPredictor(self.transition_model)
      self.updater = KalmanUpdater(self.measurement_model)
    elif filter_type == "ekf":
      self.predictor = ExtendedKalmanPredictor(self.transition_model)
      self.updater = ExtendedKalmanUpdater(self.measurement_model)
    elif filter_type == "ukf":
      self.predictor = UnscentedKalmanPredictor(self.transition_model)
      self.updater = UnscentedKalmanUpdater(self.measurement_model)
    elif filter_type == "ckf":
      self.predictor = CubatureKalmanPredictor(self.transition_model)
      self.updater = CubatureKalmanUpdater(self.measurement_model)
    else:
      raise ValueError(f"Unknown filter type: {filter_type}")

    # Initialize state
    x0 = x0 if x0 is not None else np.array([0.0, 0.0])
    self.state = GaussianState(
      state_vector=StateVector([x0[0], x0[1]]),
      covar=CovarianceMatrix(np.diag([100.0, 100.0])),
      timestamp=self.epoch
    )

  def _time_to_datetime(self, t: float) -> datetime:
    return self.epoch + timedelta(seconds=t)

  def predict(self, dt: float) -> np.ndarray:
    self.current_time += dt
    self.state = self.predictor.predict(
      self.state,
      timestamp=self._time_to_datetime(self.current_time)
    )
    return self.get_state()

  def update(self, measurement: float) -> np.ndarray:
    # Create detection
    detection = Detection(
      state_vector=StateVector([measurement]),
      timestamp=self._time_to_datetime(self.current_time),
      measurement_model=self.measurement_model
    )
    # Predict to detection time
    prediction = self.predictor.predict(self.state, timestamp=detection.timestamp)
    # Create hypothesis and update
    hypothesis = SingleHypothesis(prediction=prediction, measurement=detection)
    self.state = self.updater.update(hypothesis)
    return self.get_state()

  def get_state(self) -> np.ndarray:
    return np.array([float(self.state.state_vector[0]), float(self.state.state_vector[1])])

  def reset(self, x0: np.ndarray) -> None:
    self.current_time = 0.0
    self.state = GaussianState(
      state_vector=StateVector([x0[0], x0[1]]),
      covar=CovarianceMatrix(np.diag([100.0, 100.0])),
      timestamp=self.epoch
    )


class ParticleFilterWrapper(FilterWrapper):
  """Wrapper for Stone Soup Particle filter."""

  def __init__(self, dt: float, x0: np.ndarray | None = None, n_particles: int = 500):
    self.dt = dt
    self.n_particles = n_particles
    self.epoch = datetime(2000, 1, 1)
    self.current_time = 0.0

    # Process noise
    q = 10.0
    self.transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q)])

    # Measurement noise
    r = 1000.0
    self.measurement_model = LinearGaussian(
      ndim_state=2,
      mapping=(0,),
      noise_covar=np.array([[r]])
    )

    # Particle filter components
    self.resampler = SystematicResampler()
    self.predictor = ParticlePredictor(self.transition_model)
    self.updater = ParticleUpdater(self.measurement_model, self.resampler)

    # Initialize particles
    x0 = x0 if x0 is not None else np.array([0.0, 0.0])
    self._init_particles(x0)

  def _init_particles(self, x0: np.ndarray) -> None:
    """Initialize particle distribution around x0."""
    # Sample particles from Gaussian around initial state
    rng = np.random.default_rng(42)
    particles = rng.multivariate_normal(
      mean=x0,
      cov=np.diag([100.0, 100.0]),
      size=self.n_particles
    )
    weights = np.full(self.n_particles, 1.0 / self.n_particles)

    # StateVectors expects shape (ndim, n_particles)
    self.state = ParticleState(
      state_vector=StateVectors(particles.T),
      weight=weights,
      timestamp=self.epoch
    )

  def _time_to_datetime(self, t: float) -> datetime:
    return self.epoch + timedelta(seconds=t)

  def predict(self, dt: float) -> np.ndarray:
    self.current_time += dt
    self.state = self.predictor.predict(
      self.state,
      timestamp=self._time_to_datetime(self.current_time)
    )
    return self.get_state()

  def update(self, measurement: float) -> np.ndarray:
    detection = Detection(
      state_vector=StateVector([measurement]),
      timestamp=self._time_to_datetime(self.current_time),
      measurement_model=self.measurement_model
    )
    predicted = self.predictor.predict(self.state, timestamp=detection.timestamp)
    hypothesis = SingleHypothesis(prediction=predicted, measurement=detection)
    self.state = self.updater.update(hypothesis)
    return self.get_state()

  def get_state(self) -> np.ndarray:
    # Return weighted mean of particles
    mean = np.average(self.state.state_vector, weights=self.state.weight, axis=1)
    return np.array([float(mean[0]), float(mean[1])])

  def reset(self, x0: np.ndarray) -> None:
    self.current_time = 0.0
    self._init_particles(x0)


@dataclass
class Scenario:
  """Test scenario with ground truth trajectory."""
  name: str
  dt: float
  timestamps: np.ndarray
  ground_truth_position: np.ndarray
  ground_truth_velocity: np.ndarray
  measurements: np.ndarray  # noisy position measurements
  description: str = ""


def create_constant_velocity_scenario(
  dt: float = 0.05,
  duration: float = 10.0,
  initial_position: float = 50.0,
  velocity: float = -5.0,
  measurement_noise_std: float = 2.0,
  seed: int = 42
) -> Scenario:
  """Create scenario with constant velocity motion."""
  rng = np.random.default_rng(seed)
  n_steps = int(duration / dt)
  timestamps = np.arange(n_steps) * dt

  # Ground truth
  gt_position = initial_position + velocity * timestamps
  gt_velocity = np.full(n_steps, velocity)

  # Noisy measurements
  measurements = gt_position + rng.normal(0, measurement_noise_std, n_steps)

  return Scenario(
    name="constant_velocity",
    dt=dt,
    timestamps=timestamps,
    ground_truth_position=gt_position,
    ground_truth_velocity=gt_velocity,
    measurements=measurements,
    description=f"Constant velocity={velocity} m/s, noise_std={measurement_noise_std}"
  )


def create_acceleration_scenario(
  dt: float = 0.05,
  duration: float = 10.0,
  initial_position: float = 50.0,
  initial_velocity: float = -5.0,
  acceleration: float = -2.0,
  measurement_noise_std: float = 2.0,
  seed: int = 42
) -> Scenario:
  """Create scenario with constant acceleration motion."""
  rng = np.random.default_rng(seed)
  n_steps = int(duration / dt)
  timestamps = np.arange(n_steps) * dt

  # Ground truth (constant acceleration)
  gt_position = initial_position + initial_velocity * timestamps + 0.5 * acceleration * timestamps**2
  gt_velocity = initial_velocity + acceleration * timestamps

  # Noisy measurements
  measurements = gt_position + rng.normal(0, measurement_noise_std, n_steps)

  return Scenario(
    name="constant_acceleration",
    dt=dt,
    timestamps=timestamps,
    ground_truth_position=gt_position,
    ground_truth_velocity=gt_velocity,
    measurements=measurements,
    description=f"Constant accel={acceleration} m/s², noise_std={measurement_noise_std}"
  )


def run_filter_on_scenario(
  filter_wrapper: FilterWrapper,
  scenario: Scenario
) -> list[FilterResult]:
  """Run a filter on a scenario and return results."""
  results = []
  filter_wrapper.reset(np.array([scenario.measurements[0], 0.0]))

  for t, meas, gt_pos, gt_vel in zip(
    scenario.timestamps,
    scenario.measurements,
    scenario.ground_truth_position,
    scenario.ground_truth_velocity,
    strict=True,
  ):
    state = filter_wrapper.update(meas)
    results.append(FilterResult(
      timestamp=t,
      state=state,
      measurement=meas,
      ground_truth=np.array([gt_pos, gt_vel])
    ))

  return results


def compute_metrics(filter_name: str, results: list[FilterResult]) -> ComparisonMetrics:
  """Compute comparison metrics from filter results."""
  # Skip first few samples for filter warmup
  warmup = min(10, len(results) // 5)
  results = results[warmup:]

  position_errors = []
  velocity_errors = []

  for r in results:
    if r.ground_truth is not None:
      position_errors.append(r.state[0] - r.ground_truth[0])
      velocity_errors.append(r.state[1] - r.ground_truth[1])

  pos_err = np.array(position_errors)
  vel_err = np.array(velocity_errors)

  return ComparisonMetrics(
    filter_name=filter_name,
    rmse_position=float(np.sqrt(np.mean(pos_err**2))),
    rmse_velocity=float(np.sqrt(np.mean(vel_err**2))),
    mae_position=float(np.mean(np.abs(pos_err))),
    mae_velocity=float(np.mean(np.abs(vel_err))),
    max_error_position=float(np.max(np.abs(pos_err))),
    max_error_velocity=float(np.max(np.abs(vel_err))),
    n_samples=len(results)
  )


def compare_filters(
  scenario: Scenario,
  filter_configs: dict[str, FilterWrapper] | None = None
) -> dict[str, ComparisonMetrics]:
  """Compare multiple filters on a scenario."""
  if filter_configs is None:
    # Default filter configurations - all supported filters
    filter_configs = {
      "KF1D": KF1DWrapper(dt=scenario.dt),
      "StoneSoup_Kalman": StoneSoupKalmanWrapper(dt=scenario.dt, filter_type="kalman"),
      "StoneSoup_EKF": StoneSoupKalmanWrapper(dt=scenario.dt, filter_type="ekf"),
      "StoneSoup_UKF": StoneSoupKalmanWrapper(dt=scenario.dt, filter_type="ukf"),
      "StoneSoup_CKF": StoneSoupKalmanWrapper(dt=scenario.dt, filter_type="ckf"),
      "StoneSoup_Particle": ParticleFilterWrapper(dt=scenario.dt),
    }

  metrics = {}
  for name, filter_wrapper in filter_configs.items():
    results = run_filter_on_scenario(filter_wrapper, scenario)
    metrics[name] = compute_metrics(name, results)

  return metrics


def format_comparison_report(
  scenario: Scenario,
  metrics: dict[str, ComparisonMetrics]
) -> str:
  """Format comparison results as markdown report."""
  lines = [
    "# Filter Comparison Report",
    "",
    f"## Scenario: {scenario.name}",
    f"{scenario.description}",
    "",
    f"- Duration: {scenario.timestamps[-1]:.1f}s",
    f"- Time step: {scenario.dt*1000:.0f}ms",
    f"- Samples: {len(scenario.timestamps)}",
    "",
    "## Results",
    "",
    "| Filter | RMSE Pos | RMSE Vel | MAE Pos | MAE Vel | Max Err Pos | Max Err Vel |",
    "|--------|----------|----------|---------|---------|-------------|-------------|",
  ]

  for name, m in sorted(metrics.items()):
    line = (
      f"| {name} | {m.rmse_position:.3f} | {m.rmse_velocity:.3f} | "
      + f"{m.mae_position:.3f} | {m.mae_velocity:.3f} | "
      + f"{m.max_error_position:.3f} | {m.max_error_velocity:.3f} |"
    )
    lines.append(line)

  lines.extend([
    "",
    "Units: position (m), velocity (m/s)",
  ])

  return "\n".join(lines)
