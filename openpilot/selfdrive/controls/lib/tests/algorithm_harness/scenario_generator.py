"""
Scenario generation utilities for algorithm testing.

This module provides functions to generate synthetic driving scenarios
for testing control algorithms. Scenarios can be created from scratch
or derived from route log data.
"""

import numpy as np

from openpilot.selfdrive.controls.lib.tests.algorithm_harness.interface import (
  LateralAlgorithmState,
  LongitudinalAlgorithmState,
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.runner import Scenario
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenario_schema import (
  ScenarioMetadata,
  ScenarioType,
  DifficultyLevel,
)


def generate_highway_straight(
  duration_s: float = 30.0,
  dt_s: float = 0.01,
  v_ego: float = 30.0,  # ~108 km/h = ~67 mph
  lane_offset_noise: float = 0.0005,
) -> tuple[Scenario, ScenarioMetadata]:
  """
  Generate a highway straight driving scenario (baseline).

  This is the simplest scenario - steady speed on a straight road.
  Used as a baseline to verify algorithms work in ideal conditions.

  Args:
    duration_s: Duration in seconds
    dt_s: Time step
    v_ego: Vehicle speed (m/s)
    lane_offset_noise: Small noise in desired curvature to simulate lane keeping

  Returns:
    Tuple of (Scenario, ScenarioMetadata)
  """
  num_steps = int(duration_s / dt_s)
  states = []
  ground_truth = []

  np.random.seed(42)  # Reproducible

  for i in range(num_steps):
    t = i * dt_s
    timestamp_ns = int(t * 1e9)

    # Small random curvature noise (lane keeping)
    curvature = np.random.normal(0, lane_offset_noise)

    state = LateralAlgorithmState(
      timestamp_ns=timestamp_ns,
      v_ego=v_ego,
      a_ego=0.0,
      active=True,
      steering_angle_deg=0.0,
      yaw_rate=0.0,
      desired_curvature=curvature,
    )
    states.append(state)

    # Ground truth: small steering to track the curvature
    gt_steer = curvature * 100  # Simple proportional
    ground_truth.append(np.clip(gt_steer, -1.0, 1.0))

  metadata = ScenarioMetadata(
    name="highway_straight_baseline",
    description="Highway straight driving at constant speed. Baseline scenario for algorithm validation.",
    scenario_type=ScenarioType.HIGHWAY_STRAIGHT,
    difficulty=DifficultyLevel.EASY,
    duration_s=duration_s,
    dt_s=dt_s,
    num_steps=num_steps,
    road_type="highway",
  )

  scenario = Scenario(
    name=metadata.name,
    description=metadata.description,
    states=states,
    ground_truth=ground_truth,
  )

  return scenario, metadata


def generate_tight_s_curve(
  duration_s: float = 20.0,
  dt_s: float = 0.01,
  v_ego: float = 20.0,  # ~72 km/h
  curve_amplitude: float = 0.01,  # 1/m curvature (100m radius)
  curve_frequency: float = 0.1,  # Hz
) -> tuple[Scenario, ScenarioMetadata]:
  """
  Generate a tight S-curve scenario.

  Tests lateral control tracking through varying curvature.

  Args:
    duration_s: Duration in seconds
    dt_s: Time step
    v_ego: Vehicle speed (m/s)
    curve_amplitude: Maximum curvature (1/m)
    curve_frequency: S-curve frequency (Hz)

  Returns:
    Tuple of (Scenario, ScenarioMetadata)
  """
  num_steps = int(duration_s / dt_s)
  states = []
  ground_truth = []

  for i in range(num_steps):
    t = i * dt_s
    timestamp_ns = int(t * 1e9)

    # S-curve curvature profile
    curvature = curve_amplitude * np.sin(2 * np.pi * curve_frequency * t)

    # Compute expected steering angle (simplified model)
    wheelbase = 2.7
    steering_angle_rad = np.arctan(curvature * wheelbase)
    steering_angle_deg = np.degrees(steering_angle_rad)

    # Yaw rate from curvature and speed: yaw_rate = v * curvature
    yaw_rate = v_ego * curvature

    state = LateralAlgorithmState(
      timestamp_ns=timestamp_ns,
      v_ego=v_ego,
      a_ego=0.0,
      active=True,
      steering_angle_deg=steering_angle_deg * 0.9,  # Slight lag
      yaw_rate=yaw_rate,
      desired_curvature=curvature,
    )
    states.append(state)

    # Ground truth: steering command to achieve curvature
    gt_steer = np.clip(curvature * 50, -1.0, 1.0)
    ground_truth.append(gt_steer)

  metadata = ScenarioMetadata(
    name="tight_s_curve",
    description="Tight S-curve with varying curvature. Tests lateral control tracking ability.",
    scenario_type=ScenarioType.HIGHWAY_CURVE,
    difficulty=DifficultyLevel.MEDIUM,
    duration_s=duration_s,
    dt_s=dt_s,
    num_steps=num_steps,
    road_type="highway",
  )

  scenario = Scenario(
    name=metadata.name,
    description=metadata.description,
    states=states,
    ground_truth=ground_truth,
  )

  return scenario, metadata


def generate_highway_lane_change(
  duration_s: float = 15.0,
  dt_s: float = 0.01,
  v_ego: float = 30.0,
  lane_change_time: float = 5.0,  # When lane change starts
  lane_change_duration: float = 4.0,  # Duration of lane change
) -> tuple[Scenario, ScenarioMetadata]:
  """
  Generate a highway lane change scenario.

  Tests lateral control during a planned lane change maneuver.

  Args:
    duration_s: Total duration
    dt_s: Time step
    v_ego: Vehicle speed
    lane_change_time: Time when lane change starts
    lane_change_duration: Duration of the lane change

  Returns:
    Tuple of (Scenario, ScenarioMetadata)
  """
  num_steps = int(duration_s / dt_s)
  states = []
  ground_truth = []

  for i in range(num_steps):
    t = i * dt_s
    timestamp_ns = int(t * 1e9)

    # Lane change curvature profile (smooth S-curve)
    if t < lane_change_time:
      # Before lane change - straight
      curvature = 0.0
    elif t < lane_change_time + lane_change_duration:
      # During lane change - sinusoidal curvature
      lc_progress = (t - lane_change_time) / lane_change_duration
      # S-curve shape for smooth lane change
      curvature = 0.005 * np.sin(2 * np.pi * lc_progress)
    else:
      # After lane change - straight
      curvature = 0.0

    yaw_rate = v_ego * curvature

    state = LateralAlgorithmState(
      timestamp_ns=timestamp_ns,
      v_ego=v_ego,
      a_ego=0.0,
      active=True,
      steering_angle_deg=np.degrees(np.arctan(curvature * 2.7)),
      yaw_rate=yaw_rate,
      desired_curvature=curvature,
    )
    states.append(state)

    gt_steer = np.clip(curvature * 100, -1.0, 1.0)
    ground_truth.append(gt_steer)

  metadata = ScenarioMetadata(
    name="highway_lane_change",
    description="Highway lane change maneuver. Tests lateral control during planned lane changes.",
    scenario_type=ScenarioType.HIGHWAY_LANE_CHANGE,
    difficulty=DifficultyLevel.MEDIUM,
    duration_s=duration_s,
    dt_s=dt_s,
    num_steps=num_steps,
    road_type="highway",
  )

  scenario = Scenario(
    name=metadata.name,
    description=metadata.description,
    states=states,
    ground_truth=ground_truth,
  )

  return scenario, metadata


def generate_low_speed_maneuver(
  duration_s: float = 30.0,
  dt_s: float = 0.01,
  max_v: float = 5.0,  # ~18 km/h
  max_curvature: float = 0.05,  # 20m radius turns
) -> tuple[Scenario, ScenarioMetadata]:
  """
  Generate a low-speed parking lot maneuver scenario.

  Tests control at low speeds with tight turns (parking lot, driveway).

  Args:
    duration_s: Duration in seconds
    dt_s: Time step
    max_v: Maximum speed (m/s)
    max_curvature: Maximum curvature for tight turns

  Returns:
    Tuple of (Scenario, ScenarioMetadata)
  """
  num_steps = int(duration_s / dt_s)
  states = []
  ground_truth = []

  for i in range(num_steps):
    t = i * dt_s
    timestamp_ns = int(t * 1e9)

    # Speed profile: accelerate, cruise, decelerate
    if t < 5:
      v = max_v * (t / 5)  # Accelerate
    elif t < duration_s - 5:
      v = max_v  # Cruise
    else:
      v = max_v * ((duration_s - t) / 5)  # Decelerate

    v = max(0.1, v)  # Minimum speed

    # Curvature: various turns
    if t < 10:
      curvature = 0.0  # Straight
    elif t < 15:
      curvature = max_curvature * np.sin(np.pi * (t - 10) / 5)  # Right turn
    elif t < 20:
      curvature = 0.0  # Straight
    elif t < 25:
      curvature = -max_curvature * np.sin(np.pi * (t - 20) / 5)  # Left turn
    else:
      curvature = 0.0  # Straight

    yaw_rate = v * curvature

    state = LateralAlgorithmState(
      timestamp_ns=timestamp_ns,
      v_ego=v,
      a_ego=0.0 if 5 < t < duration_s - 5 else (max_v / 5 if t < 5 else -max_v / 5),
      active=True,
      steering_angle_deg=np.degrees(np.arctan(curvature * 2.7)),
      yaw_rate=yaw_rate,
      desired_curvature=curvature,
    )
    states.append(state)

    gt_steer = np.clip(curvature * 20, -1.0, 1.0)
    ground_truth.append(gt_steer)

  metadata = ScenarioMetadata(
    name="low_speed_maneuver",
    description="Low-speed parking lot maneuver with tight turns. Tests control at low speeds.",
    scenario_type=ScenarioType.LOW_SPEED_MANEUVER,
    difficulty=DifficultyLevel.HARD,
    duration_s=duration_s,
    dt_s=dt_s,
    num_steps=num_steps,
    road_type="urban",
  )

  scenario = Scenario(
    name=metadata.name,
    description=metadata.description,
    states=states,
    ground_truth=ground_truth,
  )

  return scenario, metadata


def generate_emergency_stop(
  duration_s: float = 10.0,
  dt_s: float = 0.01,
  initial_v: float = 25.0,  # ~90 km/h
  stop_time: float = 3.0,  # When emergency stop starts
  decel_rate: float = 6.0,  # m/s^2 (typical emergency braking)
) -> tuple[Scenario, ScenarioMetadata]:
  """
  Generate an emergency stop scenario.

  Tests longitudinal control during hard braking.

  Args:
    duration_s: Total duration
    dt_s: Time step
    initial_v: Initial speed (m/s)
    stop_time: Time when emergency stop starts
    decel_rate: Deceleration rate (m/s^2)

  Returns:
    Tuple of (Scenario, ScenarioMetadata)
  """
  num_steps = int(duration_s / dt_s)
  states = []
  ground_truth = []

  v = initial_v

  for i in range(num_steps):
    t = i * dt_s
    timestamp_ns = int(t * 1e9)

    if t < stop_time:
      # Normal driving
      a_target = 0.0
      should_stop = False
      v = initial_v
    else:
      # Emergency braking
      elapsed = t - stop_time
      v = max(0, initial_v - decel_rate * elapsed)
      a_target = -decel_rate if v > 0 else 0.0
      should_stop = True

    state = LongitudinalAlgorithmState(
      timestamp_ns=timestamp_ns,
      v_ego=v,
      a_ego=a_target,
      active=True,
      a_target=a_target,
      should_stop=should_stop,
      brake_pressed=False,
      cruise_standstill=v < 0.1,
      accel_limits=(-8.0, 2.0),  # Wider limits for emergency
    )
    states.append(state)

    gt_accel = np.clip(a_target, -8.0, 2.0)
    ground_truth.append(gt_accel)

  metadata = ScenarioMetadata(
    name="emergency_stop",
    description="Emergency stop scenario with hard braking. Tests longitudinal control response.",
    scenario_type=ScenarioType.EMERGENCY_STOP,
    difficulty=DifficultyLevel.HARD,
    duration_s=duration_s,
    dt_s=dt_s,
    num_steps=num_steps,
    road_type="highway",
  )

  scenario = Scenario(
    name=metadata.name,
    description=metadata.description,
    states=states,
    ground_truth=ground_truth,
  )

  return scenario, metadata


def generate_all_seed_scenarios() -> dict[str, tuple[Scenario, ScenarioMetadata]]:
  """
  Generate all 5 seed scenarios.

  Returns:
    Dictionary mapping scenario names to (Scenario, ScenarioMetadata) tuples
  """
  return {
    'highway_straight': generate_highway_straight(),
    'tight_s_curve': generate_tight_s_curve(),
    'highway_lane_change': generate_highway_lane_change(),
    'low_speed_maneuver': generate_low_speed_maneuver(),
    'emergency_stop': generate_emergency_stop(),
  }


def save_seed_scenarios(output_dir: str) -> list[str]:
  """
  Generate and save all seed scenarios to a directory.

  Args:
    output_dir: Output directory path

  Returns:
    List of saved file paths
  """
  from pathlib import Path
  from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenarios import save_scenario

  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  saved_files = []
  for name, (scenario, metadata) in generate_all_seed_scenarios().items():
    file_path = output_path / f"{name}.parquet"
    save_scenario(scenario, file_path, metadata)
    saved_files.append(str(file_path))

  return saved_files
