"""Adapters for converting between openpilot and Stone Soup types.

Provides type conversion for:
- Radar detections → Stone Soup Detection
- Stone Soup state estimates → openpilot LeadData
- Pose states for localization comparison
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

# Check Stone Soup availability
try:
  from stonesoup.types.detection import Detection
  from stonesoup.types.groundtruth import GroundTruthState
  from stonesoup.types.state import GaussianState, StateVector

  STONESOUP_AVAILABLE = True
except ImportError:
  STONESOUP_AVAILABLE = False


@dataclass
class RadarDetection:
  """Simplified radar detection for conversion.

  Represents a radar measurement in vehicle frame.
  """

  d_rel: float  # Relative distance (m)
  v_rel: float  # Relative velocity (m/s)
  a_rel: float  # Relative acceleration (m/s^2)
  y_rel: float  # Lateral offset (m)
  timestamp: float  # Monotonic timestamp


@dataclass
class LeadData:
  """Openpilot lead vehicle data structure.

  Represents filtered/tracked lead vehicle state.
  """

  d_rel: float  # Relative distance (m)
  v_rel: float  # Relative velocity (m/s)
  a_rel: float  # Relative acceleration (m/s^2)
  y_rel: float  # Lateral offset (m)
  d_std: float  # Distance standard deviation
  v_std: float  # Velocity standard deviation
  prob: float  # Lead probability (0-1)


@dataclass
class PoseState:
  """Vehicle pose state for localization.

  Position and velocity in world frame.
  """

  x: float  # X position (m)
  y: float  # Y position (m)
  z: float  # Z position (m)
  vx: float  # X velocity (m/s)
  vy: float  # Y velocity (m/s)
  vz: float  # Z velocity (m/s)
  roll: float  # Roll angle (rad)
  pitch: float  # Pitch angle (rad)
  yaw: float  # Yaw/heading angle (rad)
  timestamp: float


class OpenpilotAdapter:
  """Adapter for converting between openpilot and Stone Soup types.

  Usage:
    adapter = OpenpilotAdapter()

    # Convert radar detection to Stone Soup
    ss_detection = adapter.radar_to_stonesoup(radar_det, timestamp)

    # Convert Stone Soup state back to openpilot format
    lead_data = adapter.stonesoup_to_lead(ss_state, timestamp)
  """

  def __init__(self):
    """Initialize the adapter."""
    if not STONESOUP_AVAILABLE:
      raise ImportError("Stone Soup is required for this adapter. Install with: pip install stonesoup")

  def radar_to_stonesoup(
    self,
    detection: RadarDetection,
    timestamp: datetime | None = None,
  ) -> Detection:
    """Convert openpilot radar detection to Stone Soup Detection.

    Args:
      detection: Radar detection from openpilot
      timestamp: Optional datetime timestamp

    Returns:
      Stone Soup Detection object
    """
    # Create state vector [d_rel, v_rel, y_rel]
    state_vector = StateVector([[detection.d_rel], [detection.v_rel], [detection.y_rel]])

    # Create detection with metadata
    metadata = {
      "a_rel": detection.a_rel,
      "source": "radar",
    }

    if timestamp is None:
      timestamp = datetime.fromtimestamp(detection.timestamp)

    return Detection(
      state_vector=state_vector,
      timestamp=timestamp,
      metadata=metadata,
    )

  def stonesoup_to_lead(
    self,
    state: GaussianState,
    timestamp: float | None = None,
  ) -> LeadData:
    """Convert Stone Soup state estimate to openpilot LeadData.

    Args:
      state: Stone Soup GaussianState (filtered state estimate)
      timestamp: Optional monotonic timestamp

    Returns:
      LeadData compatible with openpilot
    """
    # Extract state values
    sv = state.state_vector
    d_rel = float(sv[0, 0])
    v_rel = float(sv[1, 0]) if len(sv) > 1 else 0.0
    y_rel = float(sv[2, 0]) if len(sv) > 2 else 0.0
    a_rel = float(sv[3, 0]) if len(sv) > 3 else 0.0

    # Extract uncertainties from covariance
    cov = state.covar
    d_std = float(np.sqrt(cov[0, 0]))
    v_std = float(np.sqrt(cov[1, 1])) if cov.shape[0] > 1 else 0.1

    return LeadData(
      d_rel=d_rel,
      v_rel=v_rel,
      a_rel=a_rel,
      y_rel=y_rel,
      d_std=d_std,
      v_std=v_std,
      prob=1.0,  # Tracked object has high probability
    )

  def lead_to_groundtruth(
    self,
    lead: LeadData,
    timestamp: datetime,
  ) -> GroundTruthState:
    """Convert LeadData to Stone Soup GroundTruthState for evaluation.

    Args:
      lead: Openpilot lead data (ground truth)
      timestamp: Datetime timestamp

    Returns:
      Stone Soup GroundTruthState
    """
    state_vector = StateVector(
      [
        [lead.d_rel],
        [lead.v_rel],
        [lead.y_rel],
        [lead.a_rel],
      ]
    )

    return GroundTruthState(
      state_vector=state_vector,
      timestamp=timestamp,
    )

  def pose_to_stonesoup(
    self,
    pose: PoseState,
    timestamp: datetime | None = None,
  ) -> GaussianState:
    """Convert openpilot pose state to Stone Soup GaussianState.

    Args:
      pose: Vehicle pose state
      timestamp: Optional datetime timestamp

    Returns:
      Stone Soup GaussianState
    """
    # State vector: [x, vx, y, vy, z, vz, yaw]
    state_vector = StateVector(
      [
        [pose.x],
        [pose.vx],
        [pose.y],
        [pose.vy],
        [pose.z],
        [pose.vz],
        [pose.yaw],
      ]
    )

    # Default covariance (can be overridden)
    covar = np.diag([1.0, 0.1, 1.0, 0.1, 0.5, 0.05, 0.01])

    if timestamp is None:
      timestamp = datetime.fromtimestamp(pose.timestamp)

    return GaussianState(
      state_vector=state_vector,
      covar=covar,
      timestamp=timestamp,
    )

  def stonesoup_to_pose(
    self,
    state: GaussianState,
    timestamp: float | None = None,
  ) -> PoseState:
    """Convert Stone Soup GaussianState to openpilot PoseState.

    Args:
      state: Stone Soup state estimate
      timestamp: Optional monotonic timestamp

    Returns:
      PoseState compatible with openpilot
    """
    sv = state.state_vector
    ts = timestamp or state.timestamp.timestamp()

    return PoseState(
      x=float(sv[0, 0]),
      vx=float(sv[1, 0]) if len(sv) > 1 else 0.0,
      y=float(sv[2, 0]) if len(sv) > 2 else 0.0,
      vy=float(sv[3, 0]) if len(sv) > 3 else 0.0,
      z=float(sv[4, 0]) if len(sv) > 4 else 0.0,
      vz=float(sv[5, 0]) if len(sv) > 5 else 0.0,
      roll=0.0,
      pitch=0.0,
      yaw=float(sv[6, 0]) if len(sv) > 6 else 0.0,
      timestamp=ts,
    )


def create_constant_velocity_model(
  noise_diffusion: float = 0.1,
) -> Any:
  """Create a constant velocity transition model for Stone Soup.

  Args:
    noise_diffusion: Process noise diffusion coefficient

  Returns:
    Stone Soup LinearGaussianTransitionModel
  """
  if not STONESOUP_AVAILABLE:
    raise ImportError("Stone Soup is required")

  from stonesoup.models.transition.linear import ConstantVelocity, CombinedLinearGaussianTransitionModel

  # 2D constant velocity model (x, vx, y, vy)
  model = CombinedLinearGaussianTransitionModel(
    [
      ConstantVelocity(noise_diffusion),
      ConstantVelocity(noise_diffusion),
    ]
  )

  return model


def create_position_measurement_model(
  noise_covar: np.ndarray | None = None,
) -> Any:
  """Create a position-only measurement model for Stone Soup.

  Args:
    noise_covar: Measurement noise covariance (default: identity)

  Returns:
    Stone Soup LinearGaussian measurement model
  """
  if not STONESOUP_AVAILABLE:
    raise ImportError("Stone Soup is required")

  from stonesoup.models.measurement.linear import LinearGaussian

  if noise_covar is None:
    noise_covar = np.diag([1.0, 1.0])

  # Measure position only (x, y) from state (x, vx, y, vy)
  model = LinearGaussian(
    ndim_state=4,
    mapping=(0, 2),  # Map state indices 0 and 2 to measurements
    noise_covar=noise_covar,
  )

  return model
