"""Benchmark scenarios for tracking algorithm comparison.

Provides standardized test scenarios for evaluating tracking algorithms:
- Highway following: Simple lead vehicle tracking
- Cut-in: Vehicle enters from adjacent lane
- Cut-out: Lead vehicle leaves to adjacent lane
- Multi-vehicle: Multiple leads in view
- Occlusion: Lead vehicle temporarily occluded
- Noisy: Simulated adverse weather conditions

Each scenario generates ground truth trajectories and simulated
sensor measurements with configurable noise levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class VehicleState:
  """State of a vehicle at a time instant.

  Attributes:
    x: Longitudinal position (meters, forward positive)
    y: Lateral position (meters, left positive)
    vx: Longitudinal velocity (m/s)
    vy: Lateral velocity (m/s)
    ax: Longitudinal acceleration (m/s^2)
    ay: Lateral acceleration (m/s^2)
  """

  x: float
  y: float
  vx: float = 0.0
  vy: float = 0.0
  ax: float = 0.0
  ay: float = 0.0


@dataclass
class Detection:
  """Simulated sensor detection.

  Attributes:
    d_rel: Relative distance to ego (meters)
    v_rel: Relative velocity (m/s)
    y_rel: Lateral offset (meters)
    timestamp: Time of detection (seconds)
    valid: Whether detection is valid (False for missed detections)
    noise: Added measurement noise
  """

  d_rel: float
  v_rel: float
  y_rel: float
  timestamp: float
  valid: bool = True
  noise: dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkScenario:
  """A complete benchmark scenario with ground truth and measurements.

  Attributes:
    name: Scenario name
    description: Scenario description
    duration: Total duration in seconds
    dt: Time step
    ego_trajectory: List of ego vehicle states over time
    vehicle_trajectories: Dict of vehicle_id -> list of states
    detections: Dict of vehicle_id -> list of detections
    ground_truth: Dict of vehicle_id -> list of ground truth states
    metadata: Additional scenario metadata
  """

  name: str
  description: str
  duration: float
  dt: float
  ego_trajectory: list[VehicleState]
  vehicle_trajectories: dict[int, list[VehicleState]]
  detections: dict[int, list[Detection]]
  ground_truth: dict[int, list[VehicleState]]
  metadata: dict[str, Any] = field(default_factory=dict)

  @property
  def n_frames(self) -> int:
    """Number of frames in scenario."""
    return len(self.ego_trajectory)

  @property
  def timestamps(self) -> np.ndarray:
    """Array of timestamps."""
    return np.arange(0, self.duration, self.dt)

  def get_all_detections_at(self, frame_idx: int) -> list[Detection]:
    """Get all valid detections at a frame.

    Args:
      frame_idx: Frame index

    Returns:
      List of valid detections
    """
    result = []
    for _vehicle_id, dets in self.detections.items():
      if frame_idx < len(dets) and dets[frame_idx].valid:
        result.append(dets[frame_idx])
    return result


def create_highway_scenario(
  duration: float = 30.0,
  dt: float = 0.05,
  lead_distance: float = 50.0,
  lead_velocity: float = 30.0,
  ego_velocity: float = 28.0,
  noise_std: float = 0.5,
) -> BenchmarkScenario:
  """Create highway following scenario.

  Simple scenario with one lead vehicle maintaining constant distance.

  Args:
    duration: Scenario duration (seconds)
    dt: Time step (seconds)
    lead_distance: Initial lead distance (meters)
    lead_velocity: Lead vehicle speed (m/s)
    ego_velocity: Ego vehicle speed (m/s)
    noise_std: Measurement noise standard deviation

  Returns:
    BenchmarkScenario instance
  """
  n_frames = int(duration / dt)
  timestamps = np.arange(0, duration, dt)

  # Ego trajectory (constant velocity)
  ego_trajectory = []
  ego_x = 0.0
  for _t in timestamps:
    ego_trajectory.append(VehicleState(x=ego_x, y=0.0, vx=ego_velocity))
    ego_x += ego_velocity * dt

  # Lead vehicle trajectory
  lead_trajectory = []
  lead_x = lead_distance
  for _t in timestamps:
    lead_trajectory.append(VehicleState(x=lead_x, y=0.0, vx=lead_velocity))
    lead_x += lead_velocity * dt

  # Generate detections with noise
  detections = []
  np.random.seed(42)

  for i in range(n_frames):
    ego = ego_trajectory[i]
    lead = lead_trajectory[i]

    d_rel = lead.x - ego.x
    v_rel = lead.vx - ego.vx
    y_rel = lead.y - ego.y

    # Add noise
    d_noise = np.random.normal(0, noise_std)
    v_noise = np.random.normal(0, noise_std * 0.1)
    y_noise = np.random.normal(0, noise_std * 0.2)

    det = Detection(
      d_rel=d_rel + d_noise,
      v_rel=v_rel + v_noise,
      y_rel=y_rel + y_noise,
      timestamp=timestamps[i],
      noise={"d": d_noise, "v": v_noise, "y": y_noise},
    )
    detections.append(det)

  return BenchmarkScenario(
    name="highway_following",
    description="Simple highway following with constant distance lead",
    duration=duration,
    dt=dt,
    ego_trajectory=ego_trajectory,
    vehicle_trajectories={1: lead_trajectory},
    detections={1: detections},
    ground_truth={1: lead_trajectory},
    metadata={
      "lead_distance": lead_distance,
      "lead_velocity": lead_velocity,
      "ego_velocity": ego_velocity,
      "noise_std": noise_std,
    },
  )


def create_cut_in_scenario(
  duration: float = 20.0,
  dt: float = 0.05,
  cut_in_time: float = 5.0,
  cut_in_duration: float = 3.0,
  initial_lateral: float = 3.5,
  final_distance: float = 30.0,
  ego_velocity: float = 28.0,
  noise_std: float = 0.5,
) -> BenchmarkScenario:
  """Create cut-in scenario.

  Vehicle from adjacent lane cuts in front of ego.

  Args:
    duration: Scenario duration (seconds)
    dt: Time step (seconds)
    cut_in_time: Time when cut-in starts (seconds)
    cut_in_duration: Duration of lane change (seconds)
    initial_lateral: Initial lateral offset (meters)
    final_distance: Final distance after cut-in (meters)
    ego_velocity: Ego vehicle speed (m/s)
    noise_std: Measurement noise standard deviation

  Returns:
    BenchmarkScenario instance
  """
  n_frames = int(duration / dt)
  timestamps = np.arange(0, duration, dt)

  # Ego trajectory
  ego_trajectory = []
  ego_x = 0.0
  for _t in timestamps:
    ego_trajectory.append(VehicleState(x=ego_x, y=0.0, vx=ego_velocity))
    ego_x += ego_velocity * dt

  # Cut-in vehicle trajectory
  cut_in_trajectory = []
  cut_in_x = final_distance + ego_velocity * cut_in_time
  cut_in_velocity = ego_velocity + 2.0  # Slightly faster

  for i, t in enumerate(timestamps):
    # Calculate lateral position (sigmoid transition)
    if t < cut_in_time:
      y = initial_lateral
    elif t > cut_in_time + cut_in_duration:
      y = 0.0
    else:
      progress = (t - cut_in_time) / cut_in_duration
      # Smooth S-curve
      y = initial_lateral * (1 - (3 * progress**2 - 2 * progress**3))

    # Calculate longitudinal position
    ego = ego_trajectory[i]
    cut_in_x - ego.x
    cut_in_x += cut_in_velocity * dt

    cut_in_trajectory.append(
      VehicleState(
        x=cut_in_x,
        y=y,
        vx=cut_in_velocity,
        vy=-initial_lateral / cut_in_duration if cut_in_time <= t < cut_in_time + cut_in_duration else 0,
      )
    )

  # Generate detections with noise
  detections = []
  np.random.seed(43)

  for i in range(n_frames):
    ego = ego_trajectory[i]
    vehicle = cut_in_trajectory[i]

    d_rel = vehicle.x - ego.x
    v_rel = vehicle.vx - ego.vx
    y_rel = vehicle.y

    # Add noise
    d_noise = np.random.normal(0, noise_std)
    v_noise = np.random.normal(0, noise_std * 0.1)
    y_noise = np.random.normal(0, noise_std * 0.2)

    # Before cut-in, vehicle might not be detected (outside FOV)
    valid = timestamps[i] >= cut_in_time - 1.0 or abs(y_rel) < 4.0

    det = Detection(
      d_rel=d_rel + d_noise,
      v_rel=v_rel + v_noise,
      y_rel=y_rel + y_noise,
      timestamp=timestamps[i],
      valid=valid,
      noise={"d": d_noise, "v": v_noise, "y": y_noise},
    )
    detections.append(det)

  return BenchmarkScenario(
    name="cut_in",
    description="Vehicle cuts in from adjacent lane",
    duration=duration,
    dt=dt,
    ego_trajectory=ego_trajectory,
    vehicle_trajectories={1: cut_in_trajectory},
    detections={1: detections},
    ground_truth={1: cut_in_trajectory},
    metadata={
      "cut_in_time": cut_in_time,
      "cut_in_duration": cut_in_duration,
      "initial_lateral": initial_lateral,
      "final_distance": final_distance,
      "ego_velocity": ego_velocity,
      "noise_std": noise_std,
    },
  )


def create_cut_out_scenario(
  duration: float = 20.0,
  dt: float = 0.05,
  cut_out_time: float = 5.0,
  cut_out_duration: float = 3.0,
  initial_distance: float = 30.0,
  final_lateral: float = 3.5,
  ego_velocity: float = 28.0,
  noise_std: float = 0.5,
) -> BenchmarkScenario:
  """Create cut-out scenario.

  Lead vehicle changes to adjacent lane, revealing vehicle ahead.

  Args:
    duration: Scenario duration (seconds)
    dt: Time step (seconds)
    cut_out_time: Time when cut-out starts (seconds)
    cut_out_duration: Duration of lane change (seconds)
    initial_distance: Initial distance to lead (meters)
    final_lateral: Final lateral offset (meters)
    ego_velocity: Ego vehicle speed (m/s)
    noise_std: Measurement noise standard deviation

  Returns:
    BenchmarkScenario instance
  """
  n_frames = int(duration / dt)
  timestamps = np.arange(0, duration, dt)

  # Ego trajectory
  ego_trajectory = []
  ego_x = 0.0
  for _t in timestamps:
    ego_trajectory.append(VehicleState(x=ego_x, y=0.0, vx=ego_velocity))
    ego_x += ego_velocity * dt

  # Cut-out vehicle trajectory (originally lead)
  cut_out_trajectory = []
  cut_out_x = initial_distance
  cut_out_velocity = ego_velocity

  for _i, t in enumerate(timestamps):
    # Calculate lateral position (sigmoid transition)
    if t < cut_out_time:
      y = 0.0
    elif t > cut_out_time + cut_out_duration:
      y = final_lateral
    else:
      progress = (t - cut_out_time) / cut_out_duration
      y = final_lateral * (3 * progress**2 - 2 * progress**3)

    cut_out_trajectory.append(
      VehicleState(
        x=cut_out_x,
        y=y,
        vx=cut_out_velocity,
        vy=final_lateral / cut_out_duration if cut_out_time <= t < cut_out_time + cut_out_duration else 0,
      )
    )
    cut_out_x += cut_out_velocity * dt

  # New lead vehicle (revealed after cut-out)
  new_lead_trajectory = []
  new_lead_x = initial_distance + 40.0  # 40m ahead of cut-out vehicle
  new_lead_velocity = ego_velocity - 3.0  # Slower (reason for cut-out)

  for _t in timestamps:
    new_lead_trajectory.append(
      VehicleState(
        x=new_lead_x,
        y=0.0,
        vx=new_lead_velocity,
      )
    )
    new_lead_x += new_lead_velocity * dt

  # Generate detections for both vehicles
  np.random.seed(44)
  detections_cut_out = []
  detections_new_lead = []

  for i in range(n_frames):
    ego = ego_trajectory[i]
    cut_out = cut_out_trajectory[i]
    new_lead = new_lead_trajectory[i]

    # Cut-out vehicle detection
    d_rel = cut_out.x - ego.x
    v_rel = cut_out.vx - ego.vx
    y_rel = cut_out.y

    d_noise = np.random.normal(0, noise_std)
    v_noise = np.random.normal(0, noise_std * 0.1)
    y_noise = np.random.normal(0, noise_std * 0.2)

    # After cut-out, vehicle might leave sensor FOV
    valid = abs(y_rel) < 4.0

    det1 = Detection(
      d_rel=d_rel + d_noise,
      v_rel=v_rel + v_noise,
      y_rel=y_rel + y_noise,
      timestamp=timestamps[i],
      valid=valid,
    )
    detections_cut_out.append(det1)

    # New lead detection (only visible after cut-out starts)
    d_rel = new_lead.x - ego.x
    v_rel = new_lead.vx - ego.vx
    y_rel = 0.0

    d_noise = np.random.normal(0, noise_std)
    v_noise = np.random.normal(0, noise_std * 0.1)
    y_noise = np.random.normal(0, noise_std * 0.2)

    # New lead only visible after cut-out vehicle moves
    visible = timestamps[i] >= cut_out_time + cut_out_duration * 0.5

    det2 = Detection(
      d_rel=d_rel + d_noise,
      v_rel=v_rel + v_noise,
      y_rel=y_rel + y_noise,
      timestamp=timestamps[i],
      valid=visible,
    )
    detections_new_lead.append(det2)

  return BenchmarkScenario(
    name="cut_out",
    description="Lead vehicle cuts out revealing slower vehicle ahead",
    duration=duration,
    dt=dt,
    ego_trajectory=ego_trajectory,
    vehicle_trajectories={
      1: cut_out_trajectory,
      2: new_lead_trajectory,
    },
    detections={
      1: detections_cut_out,
      2: detections_new_lead,
    },
    ground_truth={
      1: cut_out_trajectory,
      2: new_lead_trajectory,
    },
    metadata={
      "cut_out_time": cut_out_time,
      "cut_out_duration": cut_out_duration,
      "initial_distance": initial_distance,
      "final_lateral": final_lateral,
      "ego_velocity": ego_velocity,
      "noise_std": noise_std,
    },
  )


def create_multi_vehicle_scenario(
  duration: float = 30.0,
  dt: float = 0.05,
  n_vehicles: int = 3,
  base_distance: float = 30.0,
  ego_velocity: float = 28.0,
  noise_std: float = 0.5,
) -> BenchmarkScenario:
  """Create multi-vehicle scenario.

  Multiple vehicles in sensor field of view.

  Args:
    duration: Scenario duration (seconds)
    dt: Time step (seconds)
    n_vehicles: Number of vehicles
    base_distance: Base distance to first vehicle
    ego_velocity: Ego vehicle speed (m/s)
    noise_std: Measurement noise standard deviation

  Returns:
    BenchmarkScenario instance
  """
  n_frames = int(duration / dt)
  timestamps = np.arange(0, duration, dt)

  # Ego trajectory
  ego_trajectory = []
  ego_x = 0.0
  for _t in timestamps:
    ego_trajectory.append(VehicleState(x=ego_x, y=0.0, vx=ego_velocity))
    ego_x += ego_velocity * dt

  # Vehicle configurations
  vehicle_configs = [
    {"distance": base_distance, "lateral": 0.0, "velocity_delta": -2.0},  # Lead, slower
    {"distance": base_distance + 20, "lateral": 3.5, "velocity_delta": 0.0},  # Adjacent lane
    {"distance": base_distance - 10, "lateral": -3.5, "velocity_delta": 3.0},  # Other lane, faster
  ][:n_vehicles]

  # Generate trajectories
  vehicle_trajectories = {}
  all_detections = {}

  np.random.seed(45)

  for vid, config in enumerate(vehicle_configs, start=1):
    v_x = config["distance"]
    v_velocity = ego_velocity + config["velocity_delta"]

    trajectory = []
    detections = []

    for i in range(n_frames):
      trajectory.append(
        VehicleState(
          x=v_x,
          y=config["lateral"],
          vx=v_velocity,
        )
      )

      # Detection
      ego = ego_trajectory[i]
      d_rel = v_x - ego.x
      v_rel = v_velocity - ego.vx
      y_rel = config["lateral"]

      d_noise = np.random.normal(0, noise_std)
      v_noise = np.random.normal(0, noise_std * 0.1)
      y_noise = np.random.normal(0, noise_std * 0.2)

      det = Detection(
        d_rel=d_rel + d_noise,
        v_rel=v_rel + v_noise,
        y_rel=y_rel + y_noise,
        timestamp=timestamps[i],
        valid=True,
      )
      detections.append(det)

      v_x += v_velocity * dt

    vehicle_trajectories[vid] = trajectory
    all_detections[vid] = detections

  return BenchmarkScenario(
    name="multi_vehicle",
    description=f"Multiple vehicles ({n_vehicles}) in sensor view",
    duration=duration,
    dt=dt,
    ego_trajectory=ego_trajectory,
    vehicle_trajectories=vehicle_trajectories,
    detections=all_detections,
    ground_truth=vehicle_trajectories,
    metadata={
      "n_vehicles": n_vehicles,
      "base_distance": base_distance,
      "ego_velocity": ego_velocity,
      "noise_std": noise_std,
    },
  )


def create_occlusion_scenario(
  duration: float = 30.0,
  dt: float = 0.05,
  lead_distance: float = 40.0,
  occlusion_start: float = 10.0,
  occlusion_duration: float = 5.0,
  ego_velocity: float = 28.0,
  noise_std: float = 0.5,
) -> BenchmarkScenario:
  """Create occlusion scenario.

  Lead vehicle is temporarily occluded (missed detections).

  Args:
    duration: Scenario duration (seconds)
    dt: Time step (seconds)
    lead_distance: Initial lead distance (meters)
    occlusion_start: Time when occlusion starts (seconds)
    occlusion_duration: Duration of occlusion (seconds)
    ego_velocity: Ego vehicle speed (m/s)
    noise_std: Measurement noise standard deviation

  Returns:
    BenchmarkScenario instance
  """
  n_frames = int(duration / dt)
  timestamps = np.arange(0, duration, dt)

  # Ego trajectory
  ego_trajectory = []
  ego_x = 0.0
  for _t in timestamps:
    ego_trajectory.append(VehicleState(x=ego_x, y=0.0, vx=ego_velocity))
    ego_x += ego_velocity * dt

  # Lead vehicle trajectory
  lead_trajectory = []
  lead_x = lead_distance
  lead_velocity = ego_velocity - 1.0  # Slightly slower

  for _t in timestamps:
    lead_trajectory.append(VehicleState(x=lead_x, y=0.0, vx=lead_velocity))
    lead_x += lead_velocity * dt

  # Generate detections with occlusion
  detections = []
  np.random.seed(46)

  for i in range(n_frames):
    ego = ego_trajectory[i]
    lead = lead_trajectory[i]
    t = timestamps[i]

    d_rel = lead.x - ego.x
    v_rel = lead.vx - ego.vx
    y_rel = lead.y

    d_noise = np.random.normal(0, noise_std)
    v_noise = np.random.normal(0, noise_std * 0.1)
    y_noise = np.random.normal(0, noise_std * 0.2)

    # Determine if occluded
    occluded = occlusion_start <= t < occlusion_start + occlusion_duration

    det = Detection(
      d_rel=d_rel + d_noise,
      v_rel=v_rel + v_noise,
      y_rel=y_rel + y_noise,
      timestamp=t,
      valid=not occluded,
    )
    detections.append(det)

  return BenchmarkScenario(
    name="occlusion",
    description=f"Lead vehicle occluded for {occlusion_duration}s",
    duration=duration,
    dt=dt,
    ego_trajectory=ego_trajectory,
    vehicle_trajectories={1: lead_trajectory},
    detections={1: detections},
    ground_truth={1: lead_trajectory},
    metadata={
      "lead_distance": lead_distance,
      "occlusion_start": occlusion_start,
      "occlusion_duration": occlusion_duration,
      "ego_velocity": ego_velocity,
      "noise_std": noise_std,
    },
  )


def create_noisy_scenario(
  duration: float = 30.0,
  dt: float = 0.05,
  lead_distance: float = 50.0,
  ego_velocity: float = 28.0,
  base_noise_std: float = 0.5,
  noise_multiplier: float = 3.0,
  false_alarm_rate: float = 0.1,
  miss_rate: float = 0.1,
) -> BenchmarkScenario:
  """Create noisy scenario simulating adverse weather.

  Higher measurement noise, false alarms, and missed detections.

  Args:
    duration: Scenario duration (seconds)
    dt: Time step (seconds)
    lead_distance: Initial lead distance (meters)
    ego_velocity: Ego vehicle speed (m/s)
    base_noise_std: Base noise standard deviation
    noise_multiplier: Multiplier for noise (simulating bad weather)
    false_alarm_rate: Probability of false alarm per frame
    miss_rate: Probability of missed detection per frame

  Returns:
    BenchmarkScenario instance
  """
  n_frames = int(duration / dt)
  timestamps = np.arange(0, duration, dt)
  noise_std = base_noise_std * noise_multiplier

  # Ego trajectory
  ego_trajectory = []
  ego_x = 0.0
  for _t in timestamps:
    ego_trajectory.append(VehicleState(x=ego_x, y=0.0, vx=ego_velocity))
    ego_x += ego_velocity * dt

  # Lead vehicle trajectory
  lead_trajectory = []
  lead_x = lead_distance
  lead_velocity = ego_velocity - 2.0

  for _t in timestamps:
    lead_trajectory.append(VehicleState(x=lead_x, y=0.0, vx=lead_velocity))
    lead_x += lead_velocity * dt

  # Generate noisy detections
  detections = []
  false_alarms = []
  np.random.seed(47)

  for i in range(n_frames):
    ego = ego_trajectory[i]
    lead = lead_trajectory[i]

    d_rel = lead.x - ego.x
    v_rel = lead.vx - ego.vx
    y_rel = lead.y

    # Higher noise
    d_noise = np.random.normal(0, noise_std)
    v_noise = np.random.normal(0, noise_std * 0.2)
    y_noise = np.random.normal(0, noise_std * 0.4)

    # Random missed detection
    missed = np.random.random() < miss_rate

    det = Detection(
      d_rel=d_rel + d_noise,
      v_rel=v_rel + v_noise,
      y_rel=y_rel + y_noise,
      timestamp=timestamps[i],
      valid=not missed,
      noise={"d": d_noise, "v": v_noise, "y": y_noise},
    )
    detections.append(det)

    # Generate false alarms
    if np.random.random() < false_alarm_rate:
      false_d = np.random.uniform(20, 100)
      false_y = np.random.uniform(-4, 4)
      false_v = np.random.uniform(-5, 5)

      fa = Detection(
        d_rel=false_d,
        v_rel=false_v,
        y_rel=false_y,
        timestamp=timestamps[i],
        valid=True,
      )
      false_alarms.append((i, fa))

  return BenchmarkScenario(
    name="noisy",
    description="Simulated adverse weather with high noise and false alarms",
    duration=duration,
    dt=dt,
    ego_trajectory=ego_trajectory,
    vehicle_trajectories={1: lead_trajectory},
    detections={1: detections},
    ground_truth={1: lead_trajectory},
    metadata={
      "lead_distance": lead_distance,
      "ego_velocity": ego_velocity,
      "noise_std": noise_std,
      "noise_multiplier": noise_multiplier,
      "false_alarm_rate": false_alarm_rate,
      "miss_rate": miss_rate,
      "false_alarms": false_alarms,
    },
  )
