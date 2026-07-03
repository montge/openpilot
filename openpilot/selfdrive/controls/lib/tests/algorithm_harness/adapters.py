"""
Adapter wrappers for existing openpilot controllers.

These adapters wrap openpilot's native controllers (LatControlPID, LatControlTorque,
LongControl) to implement the AlgorithmInterface protocol for use with the test harness.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from openpilot.selfdrive.controls.lib.tests.algorithm_harness.interface import (
  LateralAlgorithmState,
  LateralAlgorithmOutput,
  LongitudinalAlgorithmState,
  LongitudinalAlgorithmOutput,
)


class Mock:
  """Simple mock object that auto-creates nested attributes."""

  def __init__(self, return_value: Any = None):
    super().__setattr__('_return_value', return_value)
    super().__setattr__('_children', {})

  def __getattr__(self, name: str) -> Any:
    if name == 'return_value':
      return self._return_value
    if name not in self._children:
      self._children[name] = Mock()
    return self._children[name]

  def __setattr__(self, name: str, value: Any) -> None:
    if name == 'return_value':
      super().__setattr__('_return_value', value)
    else:
      self._children[name] = value

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    return self._return_value


@dataclass
class LateralControlConfig:
  """Configuration for lateral control adapters."""

  # PID tuning
  kp_bp: list[float] = field(default_factory=lambda: [0.0])
  kp_v: list[float] = field(default_factory=lambda: [0.5])
  ki_bp: list[float] = field(default_factory=lambda: [0.0])
  ki_v: list[float] = field(default_factory=lambda: [0.1])
  kf: float = 1.0

  # Limits
  steer_limit_timer: float = 4.0
  steer_max: float = 1.0

  # Time step
  dt: float = 0.01

  # Feedforward
  feedforward_value: float = 0.0


@dataclass
class LongitudinalControlConfig:
  """Configuration for longitudinal control adapters."""

  # PID tuning
  kp_bp: list[float] = field(default_factory=lambda: [0.0])
  kp_v: list[float] = field(default_factory=lambda: [1.0])
  ki_bp: list[float] = field(default_factory=lambda: [0.0])
  ki_v: list[float] = field(default_factory=lambda: [0.1])

  # Vehicle params
  v_ego_starting: float = 0.3
  starting_state: bool = True
  stop_accel: float = -2.0
  stopping_decel_rate: float = 0.8
  start_accel: float = 1.2

  # Time step
  dt: float = 0.01


def create_mock_lateral_dependencies(
  config: LateralControlConfig,
  steer_from_curvature_fn: Optional[callable] = None,
) -> tuple[Any, Any]:
  """
  Create mock CarParams and CarInterface for lateral controllers.

  Args:
    config: Lateral control configuration
    steer_from_curvature_fn: Optional function to compute steer from curvature

  Returns:
    Tuple of (mock_CP, mock_CI)
  """
  CP = Mock()
  CP.steerLimitTimer = config.steer_limit_timer
  CP.lateralTuning.pid.kpBP = config.kp_bp
  CP.lateralTuning.pid.kpV = config.kp_v
  CP.lateralTuning.pid.kiBP = config.ki_bp
  CP.lateralTuning.pid.kiV = config.ki_v
  CP.lateralTuning.pid.kf = config.kf

  # Torque-specific params
  CP.lateralTuning.torque.kp = config.kp_v[0] if config.kp_v else 0.5
  CP.lateralTuning.torque.ki = config.ki_v[0] if config.ki_v else 0.1
  CP.lateralTuning.torque.kf = config.kf
  CP.lateralTuning.torque.friction = 0.0
  CP.lateralTuning.torque.latAccelFactor = 1.0
  CP.lateralTuning.torque.latAccelOffset = 0.0
  # For LatControlTorque.as_builder() - return self so chained attributes work
  CP.lateralTuning.torque.as_builder.return_value = CP.lateralTuning.torque

  CI = Mock()
  feedforward_fn = Mock(return_value=config.feedforward_value)
  CI.get_steer_feedforward_function.return_value = feedforward_fn

  # For LatControlTorque: provide mock torque conversion functions
  def torque_to_accel(torque: float, _params: Any) -> float:
    return torque  # Simple 1:1 mapping for testing

  def accel_to_torque(accel: float, _params: Any) -> float:
    return accel  # Simple 1:1 mapping for testing

  CI.torque_from_lateral_accel.return_value = accel_to_torque
  CI.lateral_accel_from_torque.return_value = torque_to_accel

  return CP, CI


def create_mock_car_state(state: LateralAlgorithmState) -> Any:
  """Convert LateralAlgorithmState to mock CarState."""
  CS = Mock()
  CS.vEgo = float(state.v_ego)
  CS.aEgo = float(state.a_ego)
  CS.steeringAngleDeg = float(state.steering_angle_deg)
  CS.steeringRateDeg = float(state.steering_rate_deg)
  CS.steeringPressed = bool(state.steering_pressed)
  return CS


def create_mock_vehicle_model(steer_from_curvature: float = 0.0) -> Any:
  """Create mock VehicleModel."""
  VM = Mock()
  VM.get_steer_from_curvature.return_value = steer_from_curvature
  return VM


def create_mock_params(state: LateralAlgorithmState) -> Any:
  """Create mock calibration params from state."""
  params = Mock()
  params.angleOffsetDeg = 0.0
  params.roll = state.roll
  return params


class LatControlPIDAdapter:
  """
  Adapter wrapping LatControlPID for the test harness.

  Usage:
    adapter = LatControlPIDAdapter(config)
    output = adapter.update(state)
  """

  def __init__(self, config: Optional[LateralControlConfig] = None):
    """Initialize with optional configuration."""
    self.config = config or LateralControlConfig()
    self._controller = None
    self._vm = None
    self._initialize()

  def _initialize(self) -> None:
    """Initialize the underlying controller."""
    from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID

    CP, CI = create_mock_lateral_dependencies(self.config)
    self._controller = LatControlPID(CP, CI, self.config.dt)

  def reset(self) -> None:
    """Reset controller state."""
    if self._controller is not None:
      self._controller.reset()
      self._controller.pid.reset()

  def update(self, state: LateralAlgorithmState) -> LateralAlgorithmOutput:
    """Process state and return lateral control output."""
    # Compute steer angle from desired curvature
    # This is a simplified model: steer = atan(curvature * wheelbase)
    wheelbase = 2.7  # meters, typical value
    steer_from_curvature = math.atan(-state.desired_curvature * wheelbase)

    CS = create_mock_car_state(state)
    VM = create_mock_vehicle_model(steer_from_curvature)
    params = create_mock_params(state)

    torque, angle_desired, log = self._controller.update(
      active=state.active,
      CS=CS,
      VM=VM,
      params=params,
      steer_limited_by_safety=state.steer_limited_by_safety,
      desired_curvature=state.desired_curvature,
      curvature_limited=state.curvature_limited,
      lat_delay=0.1,
    )

    return LateralAlgorithmOutput(
      output=torque,
      saturated=log.saturated if hasattr(log, 'saturated') else False,
      steering_angle_desired_deg=angle_desired,
      angle_error_deg=log.angleError if hasattr(log, 'angleError') else 0.0,
      metadata={
        'p': log.p if hasattr(log, 'p') else 0.0,
        'i': log.i if hasattr(log, 'i') else 0.0,
        'f': log.f if hasattr(log, 'f') else 0.0,
      },
    )


class LatControlTorqueAdapter:
  """Adapter wrapping LatControlTorque for the test harness."""

  def __init__(self, config: Optional[LateralControlConfig] = None):
    """Initialize with optional configuration."""
    self.config = config or LateralControlConfig()
    self._controller = None
    self._initialize()

  def _initialize(self) -> None:
    """Initialize the underlying controller."""
    from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque

    CP, CI = create_mock_lateral_dependencies(self.config)
    self._controller = LatControlTorque(CP, CI, self.config.dt)

  def reset(self) -> None:
    """Reset controller state."""
    if self._controller is not None:
      self._controller.reset()
      self._controller.pid.reset()

  def update(self, state: LateralAlgorithmState) -> LateralAlgorithmOutput:
    """Process state and return lateral control output."""
    wheelbase = 2.7
    steer_from_curvature = math.atan(-state.desired_curvature * wheelbase)

    CS = create_mock_car_state(state)
    VM = create_mock_vehicle_model(steer_from_curvature)
    params = create_mock_params(state)

    torque, angle_desired, log = self._controller.update(
      active=state.active,
      CS=CS,
      VM=VM,
      params=params,
      steer_limited_by_safety=state.steer_limited_by_safety,
      desired_curvature=state.desired_curvature,
      curvature_limited=state.curvature_limited,
      lat_delay=0.1,
    )

    return LateralAlgorithmOutput(
      output=torque,
      saturated=log.saturated if hasattr(log, 'saturated') else False,
      steering_angle_desired_deg=angle_desired,
      angle_error_deg=log.angleError if hasattr(log, 'angleError') else 0.0,
      metadata={
        'actual_lateral_accel': getattr(log, 'actualLateralAccel', 0.0),
        'desired_lateral_accel': getattr(log, 'desiredLateralAccel', 0.0),
      },
    )


class LongControlAdapter:
  """Adapter wrapping LongControl for the test harness."""

  def __init__(self, config: Optional[LongitudinalControlConfig] = None):
    """Initialize with optional configuration."""
    self.config = config or LongitudinalControlConfig()
    self._controller = None
    self._initialize()

  def _initialize(self) -> None:
    """Initialize the underlying controller."""
    from openpilot.selfdrive.controls.lib.longcontrol import LongControl

    CP = Mock()
    CP.longitudinalTuning.kpBP = self.config.kp_bp
    CP.longitudinalTuning.kpV = self.config.kp_v
    CP.longitudinalTuning.kiBP = self.config.ki_bp
    CP.longitudinalTuning.kiV = self.config.ki_v
    CP.vEgoStarting = self.config.v_ego_starting
    CP.startingState = self.config.starting_state
    CP.stopAccel = self.config.stop_accel
    CP.stoppingDecelRate = self.config.stopping_decel_rate
    CP.startAccel = self.config.start_accel

    self._controller = LongControl(CP)

  def reset(self) -> None:
    """Reset controller state."""
    if self._controller is not None:
      self._controller.reset()

  def update(self, state: LongitudinalAlgorithmState) -> LongitudinalAlgorithmOutput:
    """Process state and return longitudinal control output."""
    CS = Mock()
    CS.vEgo = float(state.v_ego)
    CS.aEgo = float(state.a_ego)
    CS.brakePressed = bool(state.brake_pressed)
    CS.cruiseState.standstill = bool(state.cruise_standstill)

    accel = self._controller.update(
      active=state.active,
      CS=CS,
      a_target=state.a_target,
      should_stop=state.should_stop,
      accel_limits=state.accel_limits,
    )

    # Get state name from Cap'n Proto enum integer value
    from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
    state_map = {
      LongCtrlState.off: 'off',
      LongCtrlState.pid: 'pid',
      LongCtrlState.stopping: 'stopping',
      LongCtrlState.starting: 'starting',
    }
    state_name = state_map.get(self._controller.long_control_state, 'unknown')

    return LongitudinalAlgorithmOutput(
      output=accel,
      accel=accel,
      saturated=abs(accel - state.accel_limits[0]) < 0.01 or abs(accel - state.accel_limits[1]) < 0.01,
      control_state=state_name,
      metadata={
        'long_control_state': self._controller.long_control_state,
      },
    )


# Convenience aliases
LatPIDAdapter = LatControlPIDAdapter
LatTorqueAdapter = LatControlTorqueAdapter
LongAdapter = LongControlAdapter
