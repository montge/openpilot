"""
Algorithm interface protocol and data classes for the test harness.

This module defines the standard interface that all algorithms must implement
to be compatible with the test harness, along with the data structures for
algorithm inputs and outputs.
"""

from dataclasses import dataclass, field
from typing import Protocol, Any, runtime_checkable


@dataclass
class AlgorithmState:
  """Base state provided to algorithms at each timestep."""
  timestamp_ns: int  # Monotonic timestamp in nanoseconds
  v_ego: float  # Vehicle speed (m/s)
  a_ego: float  # Vehicle acceleration (m/s^2)
  active: bool = True  # Whether control is active


@dataclass
class AlgorithmOutput:
  """Base output from algorithms at each timestep."""
  output: float  # Primary control output
  saturated: bool = False  # Whether output was clipped
  metadata: dict[str, Any] = field(default_factory=dict)  # Algorithm-specific debug info


@dataclass
class LateralAlgorithmState(AlgorithmState):
  """State for lateral (steering) control algorithms."""
  steering_angle_deg: float = 0.0  # Current steering angle (degrees)
  steering_rate_deg: float = 0.0  # Steering rate (degrees/s)
  yaw_rate: float = 0.0  # Vehicle yaw rate (rad/s)
  desired_curvature: float = 0.0  # Target path curvature (1/m)
  roll: float = 0.0  # Road roll angle (rad)
  steering_pressed: bool = False  # Driver override
  steer_limited_by_safety: bool = False  # Safety system limiting
  curvature_limited: bool = False  # Curvature limit active


@dataclass
class LateralAlgorithmOutput(AlgorithmOutput):
  """Output from lateral control algorithms."""
  steering_angle_desired_deg: float = 0.0  # Desired steering angle
  angle_error_deg: float = 0.0  # Steering angle error


@dataclass
class LongitudinalAlgorithmState(AlgorithmState):
  """State for longitudinal (speed) control algorithms."""
  a_target: float = 0.0  # Target acceleration (m/s^2)
  should_stop: bool = False  # Vehicle should come to stop
  brake_pressed: bool = False  # Brake pedal pressed
  cruise_standstill: bool = False  # Cruise control in standstill
  accel_limits: tuple[float, float] = (-3.5, 2.0)  # (min, max) acceleration limits


@dataclass
class LongitudinalAlgorithmOutput(AlgorithmOutput):
  """Output from longitudinal control algorithms."""
  accel: float = 0.0  # Commanded acceleration (m/s^2)
  control_state: str = "off"  # State machine state


@runtime_checkable
class AlgorithmInterface(Protocol):
  """
  Protocol defining the interface for algorithms compatible with the test harness.

  All algorithms must implement:
  - reset(): Reset internal state
  - update(state): Process state and return output

  Example usage:
    class MyAlgorithm:
      def reset(self) -> None:
        self.internal_state = 0.0

      def update(self, state: AlgorithmState) -> AlgorithmOutput:
        # Process state and compute output
        output = self.compute(state)
        return AlgorithmOutput(output=output, saturated=False)
  """

  def reset(self) -> None:
    """Reset algorithm internal state."""
    ...

  def update(self, state: AlgorithmState) -> AlgorithmOutput:
    """
    Process current state and return control output.

    Args:
      state: Current algorithm state (vehicle state, targets, etc.)

    Returns:
      AlgorithmOutput with control command and metadata
    """
    ...


@runtime_checkable
class LateralAlgorithmInterface(Protocol):
  """Protocol for lateral control algorithms."""

  def reset(self) -> None:
    """Reset algorithm internal state."""
    ...

  def update(self, state: LateralAlgorithmState) -> LateralAlgorithmOutput:
    """Process state and return lateral control output."""
    ...


@runtime_checkable
class LongitudinalAlgorithmInterface(Protocol):
  """Protocol for longitudinal control algorithms."""

  def reset(self) -> None:
    """Reset algorithm internal state."""
    ...

  def update(self, state: LongitudinalAlgorithmState) -> LongitudinalAlgorithmOutput:
    """Process state and return longitudinal control output."""
    ...
