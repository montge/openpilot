"""Comparison logger for shadow device testing.

Captures model outputs, trajectories, and control commands for offline
comparison between shadow and production devices.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FrameData:
  """Data captured for a single frame."""

  frame_id: int
  timestamp_mono: float  # time.monotonic() for internal sync
  timestamp_gps: float | None = None  # GPS time for cross-device sync

  # Model outputs
  model_outputs: dict[str, Any] = field(default_factory=dict)

  # Planned trajectory
  trajectory: dict[str, Any] = field(default_factory=dict)

  # Control commands (what would have been sent)
  controls: dict[str, float] = field(default_factory=dict)

  # Events and state
  events: list[str] = field(default_factory=list)
  state: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict:
    """Convert to dictionary for serialization."""
    return {
      "frame_id": self.frame_id,
      "timestamp_mono": self.timestamp_mono,
      "timestamp_gps": self.timestamp_gps,
      "model_outputs": self.model_outputs,
      "trajectory": self.trajectory,
      "controls": self.controls,
      "events": self.events,
      "state": self.state,
    }

  @classmethod
  def from_dict(cls, data: dict) -> FrameData:
    """Create from dictionary."""
    return cls(
      frame_id=data["frame_id"],
      timestamp_mono=data["timestamp_mono"],
      timestamp_gps=data.get("timestamp_gps"),
      model_outputs=data.get("model_outputs", {}),
      trajectory=data.get("trajectory", {}),
      controls=data.get("controls", {}),
      events=data.get("events", []),
      state=data.get("state", {}),
    )


class ComparisonLogger:
  """Logger for capturing shadow device data for comparison analysis.

  Captures model outputs, planned trajectories, control commands, and
  state for each frame. Data is serialized using JSON and optionally
  compressed with gzip.

  Usage:
    logger = ComparisonLogger(output_dir="/data/shadow_logs")
    logger.start_segment("segment_001")

    for frame in frames:
      frame_data = FrameData(
        frame_id=frame.id,
        timestamp_mono=time.monotonic(),
        model_outputs={"desired_curvature": 0.01},
        controls={"steer": 0.5, "accel": 1.0},
      )
      logger.log_frame(frame_data)

    logger.end_segment()
  """

  # Configuration
  MAX_FRAMES_PER_FILE = 6000  # ~1 minute at 100Hz
  COMPRESSION_LEVEL = 6

  def __init__(
    self,
    output_dir: str | Path,
    device_id: str = "shadow",
    compress: bool = True,
    max_frames_per_file: int | None = None,
  ):
    """Initialize the comparison logger.

    Args:
      output_dir: Directory for log files
      device_id: Identifier for this device (for multi-device setups)
      compress: Whether to gzip compress log files
      max_frames_per_file: Override for log rotation threshold
    """
    self.output_dir = Path(output_dir)
    self.device_id = device_id
    self.compress = compress
    self.max_frames_per_file = max_frames_per_file or self.MAX_FRAMES_PER_FILE

    self._current_segment: str | None = None
    self._current_file_index = 0
    self._frame_buffer: list[dict] = []
    self._total_frames = 0
    self._segment_start_time: float | None = None

    # Ensure output directory exists
    self.output_dir.mkdir(parents=True, exist_ok=True)

  def start_segment(self, segment_id: str) -> None:
    """Start logging a new segment.

    Args:
      segment_id: Unique identifier for this segment (e.g., route name)
    """
    if self._current_segment is not None:
      self.end_segment()

    self._current_segment = segment_id
    self._current_file_index = 0
    self._frame_buffer = []
    self._total_frames = 0
    self._segment_start_time = time.monotonic()

    # Create segment directory
    segment_dir = self.output_dir / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)

  def end_segment(self) -> None:
    """End the current segment and flush remaining data."""
    if self._current_segment is None:
      return

    # Flush any remaining frames
    if self._frame_buffer:
      self._flush_buffer()

    self._current_segment = None
    self._segment_start_time = None

  def log_frame(self, frame_data: FrameData) -> None:
    """Log a single frame's data.

    Args:
      frame_data: Data captured for this frame
    """
    if self._current_segment is None:
      raise RuntimeError("Must call start_segment() before logging frames")

    self._frame_buffer.append(frame_data.to_dict())
    self._total_frames += 1

    # Check for rotation
    if len(self._frame_buffer) >= self.max_frames_per_file:
      self._flush_buffer()

  def log_model_output(
    self,
    frame_id: int,
    desired_curvature: float,
    model_confidence: float | None = None,
    lane_lines: list[list[float]] | None = None,
    lead_car: dict | None = None,
  ) -> FrameData:
    """Convenience method to log model outputs.

    Args:
      frame_id: Current frame ID
      desired_curvature: Model's desired curvature output
      model_confidence: Confidence score (0-1)
      lane_lines: Lane line polynomials
      lead_car: Lead car detection data

    Returns:
      FrameData object that was logged
    """
    frame_data = FrameData(
      frame_id=frame_id,
      timestamp_mono=time.monotonic(),
      model_outputs={
        "desired_curvature": desired_curvature,
        "confidence": model_confidence,
        "lane_lines": lane_lines,
        "lead_car": lead_car,
      },
    )
    self.log_frame(frame_data)
    return frame_data

  def log_control_command(
    self,
    frame_id: int,
    steer_torque: float,
    accel: float,
    steering_angle_deg: float | None = None,
    lat_active: bool = False,
    long_active: bool = False,
    gps_time: float | None = None,
  ) -> FrameData:
    """Convenience method to log control commands.

    Args:
      frame_id: Current frame ID
      steer_torque: Steering torque command
      accel: Acceleration command
      steering_angle_deg: Steering angle in degrees
      lat_active: Whether lateral control is active
      long_active: Whether longitudinal control is active
      gps_time: GPS timestamp for synchronization

    Returns:
      FrameData object that was logged
    """
    frame_data = FrameData(
      frame_id=frame_id,
      timestamp_mono=time.monotonic(),
      timestamp_gps=gps_time,
      controls={
        "steer_torque": steer_torque,
        "accel": accel,
        **({"steering_angle_deg": steering_angle_deg} if steering_angle_deg is not None else {}),
      },
      state={
        "lat_active": lat_active,
        "long_active": long_active,
      },
    )
    self.log_frame(frame_data)
    return frame_data

  def log_trajectory(
    self,
    frame_id: int,
    x_points: list[float],
    y_points: list[float],
    v_points: list[float] | None = None,
    a_points: list[float] | None = None,
  ) -> FrameData:
    """Convenience method to log planned trajectory.

    Args:
      frame_id: Current frame ID
      x_points: X coordinates of trajectory points
      y_points: Y coordinates of trajectory points
      v_points: Velocity at each point
      a_points: Acceleration at each point

    Returns:
      FrameData object that was logged
    """
    frame_data = FrameData(
      frame_id=frame_id,
      timestamp_mono=time.monotonic(),
      trajectory={
        "x": x_points,
        "y": y_points,
        "v": v_points,
        "a": a_points,
      },
    )
    self.log_frame(frame_data)
    return frame_data

  def _flush_buffer(self) -> None:
    """Write buffered frames to disk."""
    if not self._frame_buffer or self._current_segment is None:
      return

    segment_dir = self.output_dir / self._current_segment

    # Generate filename
    ext = ".json.gz" if self.compress else ".json"
    filename = f"{self.device_id}_{self._current_file_index:04d}{ext}"
    filepath = segment_dir / filename

    # Prepare data with metadata
    data = {
      "version": 1,
      "device_id": self.device_id,
      "segment_id": self._current_segment,
      "file_index": self._current_file_index,
      "frame_count": len(self._frame_buffer),
      "frames": self._frame_buffer,
    }

    # Serialize with JSON
    json_str = json.dumps(data, default=_serialize_numpy)
    json_bytes = json_str.encode("utf-8")

    # Write (optionally compressed)
    if self.compress:
      with gzip.open(filepath, "wb", compresslevel=self.COMPRESSION_LEVEL) as f:
        f.write(json_bytes)
    else:
      with open(filepath, "wb") as f:
        f.write(json_bytes)

    # Reset buffer and increment file index
    self._frame_buffer = []
    self._current_file_index += 1

  @property
  def total_frames(self) -> int:
    """Total frames logged in current segment."""
    return self._total_frames

  @property
  def current_segment(self) -> str | None:
    """Current segment ID."""
    return self._current_segment

  @staticmethod
  def load_segment(segment_dir: str | Path) -> list[FrameData]:
    """Load all frames from a segment directory.

    Args:
      segment_dir: Path to segment directory

    Returns:
      List of FrameData objects sorted by frame_id
    """
    segment_dir = Path(segment_dir)
    frames: list[FrameData] = []

    # Find all log files
    patterns = ["*.json", "*.json.gz"]
    files: list[Path] = []
    for pattern in patterns:
      files.extend(segment_dir.glob(pattern))

    # Sort by filename (which includes file index)
    files.sort()

    for filepath in files:
      if filepath.name.endswith(".gz"):
        content = gzip.open(filepath, "rb").read()
      else:
        content = filepath.read_bytes()

      data = json.loads(content.decode("utf-8"))
      for frame_dict in data["frames"]:
        frames.append(FrameData.from_dict(frame_dict))

    # Sort by frame_id
    frames.sort(key=lambda f: f.frame_id)
    return frames

  @staticmethod
  def load_frame_range(
    segment_dir: str | Path,
    start_frame: int,
    end_frame: int,
  ) -> list[FrameData]:
    """Load a specific range of frames from a segment.

    Args:
      segment_dir: Path to segment directory
      start_frame: First frame ID to include
      end_frame: Last frame ID to include (exclusive)

    Returns:
      List of FrameData objects in the range
    """
    all_frames = ComparisonLogger.load_segment(segment_dir)
    return [f for f in all_frames if start_frame <= f.frame_id < end_frame]


def _serialize_numpy(obj: Any) -> Any:
  """Convert numpy types to Python types for JSON serialization."""
  if isinstance(obj, np.ndarray):
    return obj.tolist()
  if isinstance(obj, (np.integer, np.floating)):
    return obj.item()
  if isinstance(obj, dict):
    return {k: _serialize_numpy(v) for k, v in obj.items()}
  if isinstance(obj, list):
    return [_serialize_numpy(v) for v in obj]
  return obj
