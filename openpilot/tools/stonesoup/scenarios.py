"""Benchmark Scenarios for tracking algorithm comparison.

Provides standardized test scenarios for evaluating tracking algorithms
including highway following, cut-in/cut-out, occlusion, and adverse weather.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class VehicleTrajectory:
  """Ground truth trajectory for a vehicle."""
  vehicle_id: int
  positions: np.ndarray  # Nx2 or Nx3 array
  velocities: np.ndarray
  timestamps: np.ndarray


@dataclass
class SensorDetection:
  """Detection from a sensor at a single timestep."""
  position: np.ndarray
  velocity: np.ndarray | None = None
  confidence: float = 1.0
  sensor: str = "radar"


@dataclass
class BenchmarkScenario:
  """Complete benchmark scenario."""
  name: str
  description: str
  dt: float
  timestamps: np.ndarray
  ground_truth: list[VehicleTrajectory]
  detections_by_time: dict[float, list[SensorDetection]]
  metadata: dict


def create_highway_following(
  dt: float = 0.1,
  duration: float = 30.0,
  n_vehicles: int = 3,
  detection_prob: float = 0.95,
  position_noise_std: float = 0.3,
  seed: int | None = None
) -> BenchmarkScenario:
  """Create highway following scenario.

  Multiple vehicles traveling in same direction at different speeds.
  Tests steady-state tracking accuracy.
  """
  if seed is not None:
    np.random.seed(seed)

  n_steps = int(duration / dt)
  timestamps = np.arange(n_steps) * dt

  ground_truth = []
  for i in range(n_vehicles):
    initial_pos = np.array([30.0 + i * 20.0, 0.0])  # Staggered positions
    velocity = np.array([-3.0 - i * 0.5, 0.0])  # Slightly different speeds

    positions = np.zeros((n_steps, 2))
    velocities = np.zeros((n_steps, 2))

    for t in range(n_steps):
      positions[t] = initial_pos + velocity * timestamps[t]
      velocities[t] = velocity

    ground_truth.append(VehicleTrajectory(
      vehicle_id=i,
      positions=positions,
      velocities=velocities,
      timestamps=timestamps.copy()
    ))

  # Generate detections
  detections_by_time: dict[float, list[SensorDetection]] = {}
  for t_idx, t in enumerate(timestamps):
    dets = []
    for gt in ground_truth:
      if np.random.random() < detection_prob:
        noise = np.random.randn(2) * position_noise_std
        dets.append(SensorDetection(
          position=gt.positions[t_idx] + noise,
          velocity=gt.velocities[t_idx] + np.random.randn(2) * 0.1,
          confidence=0.8 + np.random.random() * 0.2
        ))
    detections_by_time[t] = dets

  return BenchmarkScenario(
    name="highway_following",
    description="Multiple vehicles in steady highway driving",
    dt=dt,
    timestamps=timestamps,
    ground_truth=ground_truth,
    detections_by_time=detections_by_time,
    metadata={"n_vehicles": n_vehicles, "detection_prob": detection_prob}
  )


def create_cut_in(
  dt: float = 0.1,
  duration: float = 20.0,
  cut_in_time: float = 5.0,
  detection_prob: float = 0.95,
  seed: int | None = None
) -> BenchmarkScenario:
  """Create cut-in scenario.

  A vehicle cuts in from adjacent lane. Tests track initiation
  and handling of sudden trajectory changes.
  """
  if seed is not None:
    np.random.seed(seed)

  n_steps = int(duration / dt)
  timestamps = np.arange(n_steps) * dt

  # Lead vehicle (constant lane)
  lead_positions = np.zeros((n_steps, 2))
  lead_velocities = np.zeros((n_steps, 2))
  initial_lead = np.array([50.0, 0.0])
  lead_vel = np.array([-2.0, 0.0])

  for t_idx, t in enumerate(timestamps):
    lead_positions[t_idx] = initial_lead + lead_vel * t
    lead_velocities[t_idx] = lead_vel

  # Cut-in vehicle (starts in adjacent lane, moves to ego lane)
  cutin_positions = np.zeros((n_steps, 2))
  cutin_velocities = np.zeros((n_steps, 2))
  initial_cutin = np.array([40.0, 3.5])  # Adjacent lane (y=3.5m)
  cutin_vel_x = -4.0  # Faster than lead

  for t_idx, t in enumerate(timestamps):
    x_pos = initial_cutin[0] + cutin_vel_x * t

    # Lane change dynamics
    if t < cut_in_time:
      y_pos = 3.5  # In adjacent lane
      vy = 0.0
    elif t < cut_in_time + 2.0:
      # Smooth lane change over 2 seconds
      progress = (t - cut_in_time) / 2.0
      y_pos = 3.5 * (1 - progress)  # Linear transition to y=0
      vy = -3.5 / 2.0
    else:
      y_pos = 0.0
      vy = 0.0

    cutin_positions[t_idx] = np.array([x_pos, y_pos])
    cutin_velocities[t_idx] = np.array([cutin_vel_x, vy])

  ground_truth = [
    VehicleTrajectory(0, lead_positions, lead_velocities, timestamps.copy()),
    VehicleTrajectory(1, cutin_positions, cutin_velocities, timestamps.copy())
  ]

  # Generate detections
  detections_by_time: dict[float, list[SensorDetection]] = {}
  for t_idx, t in enumerate(timestamps):
    dets = []
    for gt in ground_truth:
      if np.random.random() < detection_prob:
        noise = np.random.randn(2) * 0.3
        dets.append(SensorDetection(
          position=gt.positions[t_idx] + noise,
          velocity=gt.velocities[t_idx],
          confidence=0.85 + np.random.random() * 0.15
        ))
    detections_by_time[t] = dets

  return BenchmarkScenario(
    name="cut_in",
    description="Vehicle cuts in from adjacent lane",
    dt=dt,
    timestamps=timestamps,
    ground_truth=ground_truth,
    detections_by_time=detections_by_time,
    metadata={"cut_in_time": cut_in_time}
  )


def create_cut_out(
  dt: float = 0.1,
  duration: float = 20.0,
  cut_out_time: float = 5.0,
  seed: int | None = None
) -> BenchmarkScenario:
  """Create cut-out scenario.

  Lead vehicle moves to adjacent lane, revealing vehicle behind.
  Tests track deletion and new track initiation.
  """
  if seed is not None:
    np.random.seed(seed)

  n_steps = int(duration / dt)
  timestamps = np.arange(n_steps) * dt

  # Lead vehicle (cuts out)
  lead_positions = np.zeros((n_steps, 2))
  lead_velocities = np.zeros((n_steps, 2))
  initial_lead = np.array([40.0, 0.0])
  lead_vel_x = -3.0

  for t_idx, t in enumerate(timestamps):
    x_pos = initial_lead[0] + lead_vel_x * t

    if t < cut_out_time:
      y_pos = 0.0
      vy = 0.0
    elif t < cut_out_time + 2.0:
      progress = (t - cut_out_time) / 2.0
      y_pos = 3.5 * progress
      vy = 3.5 / 2.0
    else:
      y_pos = 3.5
      vy = 0.0

    lead_positions[t_idx] = np.array([x_pos, y_pos])
    lead_velocities[t_idx] = np.array([lead_vel_x, vy])

  # Revealed vehicle (was occluded by lead)
  revealed_positions = np.zeros((n_steps, 2))
  revealed_velocities = np.zeros((n_steps, 2))
  initial_revealed = np.array([70.0, 0.0])
  revealed_vel = np.array([-2.5, 0.0])

  for t_idx, t in enumerate(timestamps):
    revealed_positions[t_idx] = initial_revealed + revealed_vel * t
    revealed_velocities[t_idx] = revealed_vel

  ground_truth = [
    VehicleTrajectory(0, lead_positions, lead_velocities, timestamps.copy()),
    VehicleTrajectory(1, revealed_positions, revealed_velocities, timestamps.copy())
  ]

  # Generate detections (revealed vehicle not detected until after cut-out)
  detections_by_time: dict[float, list[SensorDetection]] = {}
  for t_idx, t in enumerate(timestamps):
    dets = []

    # Lead always detected
    if np.random.random() < 0.95:
      noise = np.random.randn(2) * 0.3
      dets.append(SensorDetection(
        position=ground_truth[0].positions[t_idx] + noise,
        velocity=ground_truth[0].velocities[t_idx]
      ))

    # Revealed vehicle only detected after cut-out completes
    if t > cut_out_time + 2.0 and np.random.random() < 0.95:
      noise = np.random.randn(2) * 0.3
      dets.append(SensorDetection(
        position=ground_truth[1].positions[t_idx] + noise,
        velocity=ground_truth[1].velocities[t_idx]
      ))

    detections_by_time[t] = dets

  return BenchmarkScenario(
    name="cut_out",
    description="Lead vehicle cuts out revealing vehicle behind",
    dt=dt,
    timestamps=timestamps,
    ground_truth=ground_truth,
    detections_by_time=detections_by_time,
    metadata={"cut_out_time": cut_out_time}
  )


def create_multi_vehicle(
  dt: float = 0.1,
  duration: float = 30.0,
  n_vehicles: int = 5,
  seed: int | None = None
) -> BenchmarkScenario:
  """Create complex multi-vehicle scenario.

  Multiple vehicles with varying speeds and lane positions.
  Tests data association in cluttered environments.
  """
  if seed is not None:
    np.random.seed(seed)

  n_steps = int(duration / dt)
  timestamps = np.arange(n_steps) * dt

  ground_truth = []
  for i in range(n_vehicles):
    # Random initial position and velocity
    lane = (i % 2) * 3.5  # Alternate between lanes
    initial_pos = np.array([
      20.0 + np.random.uniform(0, 60),
      lane + np.random.uniform(-0.5, 0.5)
    ])
    velocity = np.array([
      -2.0 - np.random.uniform(0, 3),
      np.random.uniform(-0.1, 0.1)
    ])

    positions = np.zeros((n_steps, 2))
    velocities = np.zeros((n_steps, 2))

    for t_idx, t in enumerate(timestamps):
      positions[t_idx] = initial_pos + velocity * t
      velocities[t_idx] = velocity

    ground_truth.append(VehicleTrajectory(
      vehicle_id=i,
      positions=positions,
      velocities=velocities,
      timestamps=timestamps.copy()
    ))

  # Generate detections with some clutter
  detections_by_time: dict[float, list[SensorDetection]] = {}
  for t_idx, t in enumerate(timestamps):
    dets = []

    for gt in ground_truth:
      if np.random.random() < 0.93:
        noise = np.random.randn(2) * 0.4
        dets.append(SensorDetection(
          position=gt.positions[t_idx] + noise,
          velocity=gt.velocities[t_idx],
          confidence=0.7 + np.random.random() * 0.3
        ))

    # Add some false alarms
    if np.random.random() < 0.1:
      dets.append(SensorDetection(
        position=np.array([
          np.random.uniform(10, 100),
          np.random.uniform(-5, 8)
        ]),
        confidence=0.3 + np.random.random() * 0.3
      ))

    detections_by_time[t] = dets

  return BenchmarkScenario(
    name="multi_vehicle",
    description="Complex traffic with multiple vehicles",
    dt=dt,
    timestamps=timestamps,
    ground_truth=ground_truth,
    detections_by_time=detections_by_time,
    metadata={"n_vehicles": n_vehicles}
  )


def create_occlusion(
  dt: float = 0.1,
  duration: float = 20.0,
  occlusion_start: float = 5.0,
  occlusion_duration: float = 3.0,
  seed: int | None = None
) -> BenchmarkScenario:
  """Create occlusion scenario.

  Vehicle gets temporarily occluded (no detections).
  Tests track maintenance during detection gaps.
  """
  if seed is not None:
    np.random.seed(seed)

  n_steps = int(duration / dt)
  timestamps = np.arange(n_steps) * dt

  initial_pos = np.array([50.0, 0.0])
  velocity = np.array([-3.0, 0.0])

  positions = np.zeros((n_steps, 2))
  velocities = np.zeros((n_steps, 2))

  for t_idx, t in enumerate(timestamps):
    positions[t_idx] = initial_pos + velocity * t
    velocities[t_idx] = velocity

  ground_truth = [VehicleTrajectory(
    vehicle_id=0,
    positions=positions,
    velocities=velocities,
    timestamps=timestamps.copy()
  )]

  # Generate detections with occlusion gap
  detections_by_time: dict[float, list[SensorDetection]] = {}
  for t_idx, t in enumerate(timestamps):
    is_occluded = occlusion_start <= t < (occlusion_start + occlusion_duration)

    if is_occluded:
      detections_by_time[t] = []
    elif np.random.random() < 0.95:
      noise = np.random.randn(2) * 0.3
      detections_by_time[t] = [SensorDetection(
        position=positions[t_idx] + noise,
        velocity=velocities[t_idx]
      )]
    else:
      detections_by_time[t] = []

  return BenchmarkScenario(
    name="occlusion",
    description="Vehicle temporarily occluded",
    dt=dt,
    timestamps=timestamps,
    ground_truth=ground_truth,
    detections_by_time=detections_by_time,
    metadata={
      "occlusion_start": occlusion_start,
      "occlusion_duration": occlusion_duration
    }
  )


def create_adverse_weather(
  dt: float = 0.1,
  duration: float = 20.0,
  n_vehicles: int = 2,
  detection_prob: float = 0.7,  # Lower in bad weather
  position_noise_std: float = 1.0,  # Higher noise
  clutter_rate: float = 0.3,  # More false alarms
  seed: int | None = None
) -> BenchmarkScenario:
  """Create adverse weather scenario.

  Simulates degraded sensor performance with:
  - Lower detection probability
  - Higher position noise
  - More false alarms
  """
  if seed is not None:
    np.random.seed(seed)

  n_steps = int(duration / dt)
  timestamps = np.arange(n_steps) * dt

  ground_truth = []
  for i in range(n_vehicles):
    initial_pos = np.array([30.0 + i * 25.0, 0.0])
    velocity = np.array([-2.5 - i * 0.5, 0.0])

    positions = np.zeros((n_steps, 2))
    velocities = np.zeros((n_steps, 2))

    for t_idx, t in enumerate(timestamps):
      positions[t_idx] = initial_pos + velocity * t
      velocities[t_idx] = velocity

    ground_truth.append(VehicleTrajectory(
      vehicle_id=i,
      positions=positions,
      velocities=velocities,
      timestamps=timestamps.copy()
    ))

  # Generate noisy detections
  detections_by_time: dict[float, list[SensorDetection]] = {}
  for t_idx, t in enumerate(timestamps):
    dets = []

    for gt in ground_truth:
      if np.random.random() < detection_prob:
        noise = np.random.randn(2) * position_noise_std
        dets.append(SensorDetection(
          position=gt.positions[t_idx] + noise,
          velocity=gt.velocities[t_idx] + np.random.randn(2) * 0.5,
          confidence=0.4 + np.random.random() * 0.4
        ))

    # Add clutter (false alarms)
    n_clutter = np.random.poisson(clutter_rate)
    for _ in range(n_clutter):
      dets.append(SensorDetection(
        position=np.array([
          np.random.uniform(5, 80),
          np.random.uniform(-5, 5)
        ]),
        confidence=0.2 + np.random.random() * 0.3
      ))

    detections_by_time[t] = dets

  return BenchmarkScenario(
    name="adverse_weather",
    description="Degraded sensor performance (rain/fog)",
    dt=dt,
    timestamps=timestamps,
    ground_truth=ground_truth,
    detections_by_time=detections_by_time,
    metadata={
      "detection_prob": detection_prob,
      "noise_std": position_noise_std,
      "clutter_rate": clutter_rate
    }
  )


def get_all_scenarios(seed: int | None = 42) -> dict[str, BenchmarkScenario]:
  """Get all standard benchmark scenarios."""
  return {
    "highway_following": create_highway_following(seed=seed),
    "cut_in": create_cut_in(seed=seed),
    "cut_out": create_cut_out(seed=seed),
    "multi_vehicle": create_multi_vehicle(seed=seed),
    "occlusion": create_occlusion(seed=seed),
    "adverse_weather": create_adverse_weather(seed=seed),
  }


def format_scenario_summary(scenarios: dict[str, BenchmarkScenario]) -> str:
  """Format summary of all scenarios."""
  lines = [
    "# Benchmark Scenarios Summary",
    "",
    "| Scenario | Duration | Vehicles | Description |",
    "|----------|----------|----------|-------------|",
  ]

  for name, scenario in scenarios.items():
    n_vehicles = len(scenario.ground_truth)
    duration = scenario.timestamps[-1]
    lines.append(
      f"| {name} | {duration:.1f}s | {n_vehicles} | {scenario.description} |"
    )

  return "\n".join(lines)


if __name__ == "__main__":
  scenarios = get_all_scenarios()
  print(format_scenario_summary(scenarios))

  print("\n\nDetailed scenario info:")
  for name, scenario in scenarios.items():
    print(f"\n{name}:")
    print(f"  Duration: {scenario.timestamps[-1]:.1f}s")
    print(f"  Vehicles: {len(scenario.ground_truth)}")
    total_dets = sum(len(d) for d in scenario.detections_by_time.values())
    print(f"  Total detections: {total_dets}")
