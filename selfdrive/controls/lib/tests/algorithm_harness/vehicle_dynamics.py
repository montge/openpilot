"""
Vehicle dynamics configuration for algorithm testing.

This module provides configurable vehicle dynamics models for testing
control algorithms without hardware. It includes presets for common
vehicle types (sedan, SUV, truck) and utilities for computing steering
and acceleration responses.

The bicycle model is used as the primary dynamics representation,
matching openpilot's internal vehicle model.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class VehicleType(Enum):
  """Predefined vehicle types with typical parameters."""

  SEDAN = "sedan"
  SUV = "suv"
  TRUCK = "truck"
  COMPACT = "compact"
  SPORTS = "sports"
  CUSTOM = "custom"


@dataclass
class VehicleDynamicsConfig:
  """
  Vehicle dynamics configuration parameters.

  Based on openpilot's CarParams structure but simplified for testing.
  Uses bicycle model assumptions.

  Attributes:
    name: Human-readable vehicle name
    vehicle_type: Enumerated vehicle type
    wheelbase: Distance between front and rear axles (meters)
    steer_ratio: Steering wheel angle to road wheel angle ratio
    mass: Vehicle mass (kg)
    rotational_inertia: Moment of inertia about vertical axis (kg*m^2)
    center_to_front: Distance from CG to front axle (meters)
    tire_stiffness_front: Front tire cornering stiffness (N/rad)
    tire_stiffness_rear: Rear tire cornering stiffness (N/rad)
    max_steer_angle_deg: Maximum steering wheel angle (degrees)
    max_lateral_accel: Maximum lateral acceleration (m/s^2)
    max_longitudinal_accel: Maximum longitudinal acceleration (m/s^2)
    max_longitudinal_decel: Maximum deceleration (m/s^2, positive value)
    actuator_delay_steer: Steering actuator delay (seconds)
    actuator_delay_accel: Acceleration actuator delay (seconds)
  """

  name: str = "Generic Sedan"
  vehicle_type: VehicleType = VehicleType.SEDAN
  wheelbase: float = 2.7  # meters
  steer_ratio: float = 15.0
  mass: float = 1500.0  # kg
  rotational_inertia: float = 2500.0  # kg*m^2
  center_to_front: float = 1.35  # meters (typically ~wheelbase/2)
  tire_stiffness_front: float = 200000.0  # N/rad
  tire_stiffness_rear: float = 200000.0  # N/rad
  max_steer_angle_deg: float = 540.0  # degrees
  max_lateral_accel: float = 3.0  # m/s^2
  max_longitudinal_accel: float = 2.0  # m/s^2
  max_longitudinal_decel: float = 4.0  # m/s^2

  # Actuator delays
  actuator_delay_steer: float = 0.1  # seconds
  actuator_delay_accel: float = 0.2  # seconds

  # Optional: friction coefficient (used for tire slip estimation)
  friction_coefficient: float = 1.0

  def __post_init__(self):
    """Validate configuration parameters."""
    if self.wheelbase <= 0:
      raise ValueError(f"Wheelbase must be positive, got {self.wheelbase}")
    if self.steer_ratio <= 0:
      raise ValueError(f"Steer ratio must be positive, got {self.steer_ratio}")
    if self.mass <= 0:
      raise ValueError(f"Mass must be positive, got {self.mass}")
    if self.center_to_front < 0 or self.center_to_front > self.wheelbase:
      raise ValueError("Center to front must be between 0 and wheelbase")

  @property
  def center_to_rear(self) -> float:
    """Distance from center of gravity to rear axle."""
    return self.wheelbase - self.center_to_front

  @property
  def max_road_wheel_angle_deg(self) -> float:
    """Maximum road wheel angle (front wheel angle)."""
    return self.max_steer_angle_deg / self.steer_ratio

  @property
  def understeer_gradient(self) -> float:
    """
    Understeer gradient (deg/g).

    Positive = understeering, negative = oversteering.
    Typical values: 1-5 deg/g for understeering vehicles.
    """
    # Simplified calculation based on tire stiffness and weight distribution
    weight_front = (self.center_to_rear / self.wheelbase) * self.mass
    weight_rear = (self.center_to_front / self.wheelbase) * self.mass

    # Understeer gradient = Wf/Cf - Wr/Cr (radians/m/s^2)
    # Convert to deg/g
    K = weight_front / self.tire_stiffness_front - weight_rear / self.tire_stiffness_rear
    return math.degrees(K) * 9.81  # deg/g


@dataclass
class BicycleModelState:
  """State representation for bicycle model dynamics."""

  x: float = 0.0  # Position in global X (meters)
  y: float = 0.0  # Position in global Y (meters)
  yaw: float = 0.0  # Heading angle (radians)
  v: float = 0.0  # Longitudinal velocity (m/s)
  yaw_rate: float = 0.0  # Yaw rate (rad/s)
  slip_angle: float = 0.0  # Sideslip angle at CG (radians)
  steer_angle: float = 0.0  # Front wheel steer angle (radians)


class BicycleModel:
  """
  Kinematic bicycle model for vehicle dynamics simulation.

  Provides a simple but accurate model for low-speed and normal driving
  conditions. Does not model tire slip at high lateral accelerations.
  """

  def __init__(self, config: VehicleDynamicsConfig):
    self.config = config
    self.state = BicycleModelState()

  def reset(self, initial_state: Optional[BicycleModelState] = None):
    """Reset the model to initial state."""
    self.state = initial_state or BicycleModelState()

  def curvature_to_steer_angle(self, curvature: float) -> float:
    """
    Convert desired curvature to steering wheel angle.

    Args:
      curvature: Desired path curvature (1/m)

    Returns:
      Steering wheel angle (radians)
    """
    # Road wheel angle from curvature: delta = atan(L * kappa)
    road_wheel_angle = math.atan(self.config.wheelbase * curvature)
    # Steering wheel angle = road wheel angle * steer ratio
    return road_wheel_angle * self.config.steer_ratio

  def steer_angle_to_curvature(self, steer_angle_rad: float) -> float:
    """
    Convert steering wheel angle to path curvature.

    Args:
      steer_angle_rad: Steering wheel angle (radians)

    Returns:
      Path curvature (1/m)
    """
    road_wheel_angle = steer_angle_rad / self.config.steer_ratio
    return math.tan(road_wheel_angle) / self.config.wheelbase

  def yaw_rate_from_curvature(self, curvature: float, v_ego: float) -> float:
    """
    Compute yaw rate from curvature and speed.

    Args:
      curvature: Path curvature (1/m)
      v_ego: Vehicle speed (m/s)

    Returns:
      Yaw rate (rad/s)
    """
    return curvature * v_ego

  def lateral_accel_from_curvature(self, curvature: float, v_ego: float) -> float:
    """
    Compute lateral acceleration from curvature and speed.

    Args:
      curvature: Path curvature (1/m)
      v_ego: Vehicle speed (m/s)

    Returns:
      Lateral acceleration (m/s^2)
    """
    return curvature * v_ego * v_ego

  def max_curvature_at_speed(self, v_ego: float) -> float:
    """
    Compute maximum safe curvature at given speed.

    Limited by lateral acceleration and maximum steering angle.

    Args:
      v_ego: Vehicle speed (m/s)

    Returns:
      Maximum curvature (1/m)
    """
    if v_ego < 0.1:
      # At very low speeds, limited by max steer angle
      max_road_angle_rad = math.radians(self.config.max_road_wheel_angle_deg)
      return math.tan(max_road_angle_rad) / self.config.wheelbase

    # Limited by lateral acceleration
    curvature_accel = self.config.max_lateral_accel / (v_ego * v_ego)

    # Limited by steering geometry
    max_road_angle_rad = math.radians(self.config.max_road_wheel_angle_deg)
    curvature_steer = math.tan(max_road_angle_rad) / self.config.wheelbase

    return min(curvature_accel, curvature_steer)

  def step(self, dt: float, steer_cmd: float, accel_cmd: float) -> BicycleModelState:
    """
    Advance the model by one time step.

    Args:
      dt: Time step (seconds)
      steer_cmd: Steering command (normalized -1 to 1)
      accel_cmd: Acceleration command (m/s^2)

    Returns:
      Updated state
    """
    # Convert normalized steering to road wheel angle
    max_road_angle_rad = math.radians(self.config.max_road_wheel_angle_deg)
    delta = steer_cmd * max_road_angle_rad

    # Clamp acceleration
    accel = max(-self.config.max_longitudinal_decel, min(self.config.max_longitudinal_accel, accel_cmd))

    # Update velocity
    v_new = max(0, self.state.v + accel * dt)

    # Kinematic bicycle model update
    if abs(self.state.v) > 0.01:
      # Curvature from steering angle
      curvature = math.tan(delta) / self.config.wheelbase

      # Yaw rate
      yaw_rate = curvature * self.state.v

      # Update position and heading
      if abs(curvature) > 1e-6:
        # Curved path
        radius = 1.0 / curvature
        beta = yaw_rate * dt
        dx = radius * (math.sin(self.state.yaw + beta) - math.sin(self.state.yaw))
        dy = radius * (math.cos(self.state.yaw) - math.cos(self.state.yaw + beta))
      else:
        # Straight line
        dx = self.state.v * dt * math.cos(self.state.yaw)
        dy = self.state.v * dt * math.sin(self.state.yaw)
        beta = 0

      self.state.x += dx
      self.state.y += dy
      self.state.yaw += beta
      self.state.yaw_rate = yaw_rate
    else:
      self.state.yaw_rate = 0

    self.state.v = v_new
    self.state.steer_angle = delta

    return self.state


# ============================================================================
# Vehicle Presets
# ============================================================================


def get_sedan_config(name: str = "Generic Sedan") -> VehicleDynamicsConfig:
  """Get configuration for a typical sedan (e.g., Honda Civic, Toyota Camry)."""
  return VehicleDynamicsConfig(
    name=name,
    vehicle_type=VehicleType.SEDAN,
    wheelbase=2.7,
    steer_ratio=15.0,
    mass=1500.0,
    rotational_inertia=2500.0,
    center_to_front=1.35,
    tire_stiffness_front=200000.0,
    tire_stiffness_rear=200000.0,
    max_steer_angle_deg=540.0,
    max_lateral_accel=3.0,
    max_longitudinal_accel=2.0,
    max_longitudinal_decel=4.0,
    actuator_delay_steer=0.1,
    actuator_delay_accel=0.2,
  )


def get_suv_config(name: str = "Generic SUV") -> VehicleDynamicsConfig:
  """Get configuration for a typical SUV (e.g., Toyota RAV4, Honda CR-V)."""
  return VehicleDynamicsConfig(
    name=name,
    vehicle_type=VehicleType.SUV,
    wheelbase=2.85,
    steer_ratio=16.0,
    mass=1900.0,
    rotational_inertia=3500.0,
    center_to_front=1.4,
    tire_stiffness_front=180000.0,
    tire_stiffness_rear=180000.0,
    max_steer_angle_deg=540.0,
    max_lateral_accel=2.5,
    max_longitudinal_accel=1.8,
    max_longitudinal_decel=3.5,
    actuator_delay_steer=0.12,
    actuator_delay_accel=0.25,
  )


def get_truck_config(name: str = "Generic Truck") -> VehicleDynamicsConfig:
  """Get configuration for a typical truck (e.g., Ford F-150, Chevy Silverado)."""
  return VehicleDynamicsConfig(
    name=name,
    vehicle_type=VehicleType.TRUCK,
    wheelbase=3.7,
    steer_ratio=18.0,
    mass=2500.0,
    rotational_inertia=5000.0,
    center_to_front=1.8,
    tire_stiffness_front=160000.0,
    tire_stiffness_rear=220000.0,
    max_steer_angle_deg=540.0,
    max_lateral_accel=2.0,
    max_longitudinal_accel=1.5,
    max_longitudinal_decel=3.0,
    actuator_delay_steer=0.15,
    actuator_delay_accel=0.3,
  )


def get_compact_config(name: str = "Generic Compact") -> VehicleDynamicsConfig:
  """Get configuration for a compact car (e.g., VW Golf, Honda Fit)."""
  return VehicleDynamicsConfig(
    name=name,
    vehicle_type=VehicleType.COMPACT,
    wheelbase=2.55,
    steer_ratio=14.0,
    mass=1200.0,
    rotational_inertia=1800.0,
    center_to_front=1.25,
    tire_stiffness_front=180000.0,
    tire_stiffness_rear=180000.0,
    max_steer_angle_deg=540.0,
    max_lateral_accel=3.5,
    max_longitudinal_accel=2.5,
    max_longitudinal_decel=4.5,
    actuator_delay_steer=0.08,
    actuator_delay_accel=0.15,
  )


def get_sports_config(name: str = "Generic Sports") -> VehicleDynamicsConfig:
  """Get configuration for a sports car (e.g., Mazda Miata, Porsche 911)."""
  return VehicleDynamicsConfig(
    name=name,
    vehicle_type=VehicleType.SPORTS,
    wheelbase=2.45,
    steer_ratio=12.0,
    mass=1300.0,
    rotational_inertia=2000.0,
    center_to_front=1.2,
    tire_stiffness_front=220000.0,
    tire_stiffness_rear=240000.0,
    max_steer_angle_deg=450.0,
    max_lateral_accel=4.5,
    max_longitudinal_accel=3.0,
    max_longitudinal_decel=5.0,
    actuator_delay_steer=0.05,
    actuator_delay_accel=0.1,
  )


def get_vehicle_config(vehicle_type: VehicleType, name: Optional[str] = None) -> VehicleDynamicsConfig:
  """
  Get vehicle configuration by type.

  Args:
    vehicle_type: Type of vehicle
    name: Optional custom name

  Returns:
    VehicleDynamicsConfig for the specified vehicle type
  """
  configs = {
    VehicleType.SEDAN: get_sedan_config,
    VehicleType.SUV: get_suv_config,
    VehicleType.TRUCK: get_truck_config,
    VehicleType.COMPACT: get_compact_config,
    VehicleType.SPORTS: get_sports_config,
  }

  if vehicle_type not in configs:
    return get_sedan_config(name or "Custom Vehicle")

  config_fn = configs[vehicle_type]
  return config_fn(name) if name else config_fn()


# Dictionary mapping vehicle type strings to configs
VEHICLE_PRESETS = {
  "sedan": get_sedan_config(),
  "suv": get_suv_config(),
  "truck": get_truck_config(),
  "compact": get_compact_config(),
  "sports": get_sports_config(),
}
