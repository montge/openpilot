"""Lane tracking using CoTracker-based point tracking.

Provides robust lane marking tracking through video sequences,
using techniques inspired by CoTracker for temporal consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LanePoint:
  """Single tracked lane point.

  Attributes:
    x: X coordinate in image
    y: Y coordinate in image
    confidence: Tracking confidence [0, 1]
    visible: Whether point is visible (not occluded)
    lane_id: ID of the lane this point belongs to
  """

  x: float
  y: float
  confidence: float = 1.0
  visible: bool = True
  lane_id: int = 0


@dataclass
class LaneLine:
  """Tracked lane line.

  Attributes:
    points: List of points along the lane
    lane_id: Unique lane identifier
    lane_type: Type of lane ('solid', 'dashed', 'double')
    color: Lane color ('white', 'yellow')
    confidence: Overall lane confidence
  """

  points: list[LanePoint] = field(default_factory=list)
  lane_id: int = 0
  lane_type: str = "solid"
  color: str = "white"
  confidence: float = 1.0


@dataclass
class LaneTrackingConfig:
  """Lane tracking configuration.

  Attributes:
    num_points_per_lane: Points to track per lane line
    max_lanes: Maximum number of lanes to track
    min_confidence: Minimum confidence to keep track
    temporal_smoothing: Weight for temporal smoothing [0, 1]
    search_radius: Pixel radius for point correspondence
  """

  num_points_per_lane: int = 10
  max_lanes: int = 4
  min_confidence: float = 0.3
  temporal_smoothing: float = 0.7
  search_radius: int = 20


class LaneTracker:
  """Lane tracking through video sequences.

  Uses point tracking for robust lane following across frames.
  Inspired by CoTracker's dense point tracking approach.

  Usage:
    config = LaneTrackingConfig()
    tracker = LaneTracker(config)

    # Initialize on first frame
    lanes = tracker.detect_lanes(first_frame)

    # Track through video
    for frame in video_frames:
      lanes = tracker.update(frame)
  """

  def __init__(self, config: LaneTrackingConfig | None = None):
    """Initialize lane tracker.

    Args:
      config: Lane tracking configuration
    """
    self.config = config or LaneTrackingConfig()

    # Tracking state
    self._lanes: list[LaneLine] = []
    self._prev_frame: np.ndarray | None = None
    self._initialized = False

  def detect_lanes(self, image: np.ndarray) -> list[LaneLine]:
    """Detect lane lines in an image.

    Initializes or re-initializes tracking.

    Args:
      image: RGB image [H, W, 3]

    Returns:
      Detected lane lines
    """
    # Detect lane points in current frame
    lane_points = self._detect_lane_points(image)

    # Group points into lanes
    self._lanes = self._group_into_lanes(lane_points)

    self._prev_frame = image.copy()
    self._initialized = True

    return self._lanes

  def update(self, image: np.ndarray) -> list[LaneLine]:
    """Update lane tracking with new frame.

    Args:
      image: RGB image [H, W, 3]

    Returns:
      Updated lane lines
    """
    if not self._initialized:
      return self.detect_lanes(image)

    # Track existing points
    updated_lanes = []
    for lane in self._lanes:
      updated_points = self._track_points(lane.points, image)

      # Filter low-confidence points
      updated_points = [p for p in updated_points if p.confidence >= self.config.min_confidence]

      if updated_points:
        lane.points = updated_points
        lane.confidence = np.mean([p.confidence for p in updated_points])
        updated_lanes.append(lane)

    # Detect new lanes if we lost too many
    if len(updated_lanes) < 2:  # At least need left and right
      new_detections = self._detect_lane_points(image)
      new_lanes = self._group_into_lanes(new_detections)

      # Add new lanes that don't overlap with existing
      for new_lane in new_lanes:
        if not self._overlaps_existing(new_lane, updated_lanes):
          new_lane.lane_id = len(updated_lanes)
          updated_lanes.append(new_lane)

    self._lanes = updated_lanes[: self.config.max_lanes]
    self._prev_frame = image.copy()

    return self._lanes

  def get_lane_points(self, lane_id: int) -> list[tuple[float, float]]:
    """Get points for a specific lane.

    Args:
      lane_id: Lane identifier

    Returns:
      List of (x, y) points
    """
    for lane in self._lanes:
      if lane.lane_id == lane_id:
        return [(p.x, p.y) for p in lane.points]
    return []

  def get_ego_lanes(self) -> tuple[LaneLine | None, LaneLine | None]:
    """Get left and right ego lane lines.

    Returns:
      Tuple of (left_lane, right_lane), None if not detected
    """
    if not self._lanes:
      return None, None

    # Sort lanes by average x position
    sorted_lanes = sorted(self._lanes, key=lambda l: np.mean([p.x for p in l.points]) if l.points else 0)

    # Assume image center divides left/right
    # In practice, would use camera model
    left_lane = None
    right_lane = None

    for lane in sorted_lanes:
      if not lane.points:
        continue
      avg_x = np.mean([p.x for p in lane.points])
      if avg_x < 256:  # Assuming 512 width
        left_lane = lane
      else:
        if right_lane is None:
          right_lane = lane

    return left_lane, right_lane

  def fit_polynomial(
    self,
    lane: LaneLine,
    degree: int = 2,
  ) -> np.ndarray | None:
    """Fit polynomial to lane points.

    Args:
      lane: Lane line to fit
      degree: Polynomial degree

    Returns:
      Polynomial coefficients or None if fit fails
    """
    if len(lane.points) < degree + 1:
      return None

    # Extract points (y as independent variable for lanes)
    y = np.array([p.y for p in lane.points])
    x = np.array([p.x for p in lane.points])

    try:
      coeffs = np.polyfit(y, x, degree)
      return coeffs
    except np.linalg.LinAlgError:
      return None

  def _detect_lane_points(self, image: np.ndarray) -> list[LanePoint]:
    """Detect lane points in image.

    Args:
      image: RGB image

    Returns:
      Detected lane points
    """
    # Placeholder - would use lane detection network
    # Returns dummy points for two lanes
    h, w = image.shape[:2]
    points = []

    # Left lane (dummy)
    for i in range(self.config.num_points_per_lane):
      y = h * 0.4 + i * (h * 0.55 / self.config.num_points_per_lane)
      x = w * 0.35 - i * (w * 0.1 / self.config.num_points_per_lane)
      points.append(LanePoint(x=x, y=y, lane_id=0))

    # Right lane (dummy)
    for i in range(self.config.num_points_per_lane):
      y = h * 0.4 + i * (h * 0.55 / self.config.num_points_per_lane)
      x = w * 0.65 + i * (w * 0.1 / self.config.num_points_per_lane)
      points.append(LanePoint(x=x, y=y, lane_id=1))

    return points

  def _group_into_lanes(self, points: list[LanePoint]) -> list[LaneLine]:
    """Group points into lane lines.

    Args:
      points: Detected lane points

    Returns:
      Grouped lane lines
    """
    # Group by lane_id
    lanes_dict: dict[int, list[LanePoint]] = {}
    for point in points:
      if point.lane_id not in lanes_dict:
        lanes_dict[point.lane_id] = []
      lanes_dict[point.lane_id].append(point)

    # Create lane objects
    lanes = []
    for lane_id, lane_points in lanes_dict.items():
      # Sort by y coordinate
      lane_points.sort(key=lambda p: p.y)
      lanes.append(
        LaneLine(
          points=lane_points,
          lane_id=lane_id,
          confidence=np.mean([p.confidence for p in lane_points]),
        )
      )

    return lanes

  def _track_points(
    self,
    points: list[LanePoint],
    new_frame: np.ndarray,
  ) -> list[LanePoint]:
    """Track points from previous frame to new frame.

    Args:
      points: Points to track
      new_frame: New frame image

    Returns:
      Updated points
    """
    if self._prev_frame is None:
      return points

    # Placeholder - would use optical flow or learned tracker
    # For now, apply small motion with reduced confidence
    updated = []
    for point in points:
      # Simulate slight downward motion (road perspective)
      new_y = point.y + 2
      new_x = point.x

      # Apply temporal smoothing
      alpha = self.config.temporal_smoothing
      smoothed_x = alpha * new_x + (1 - alpha) * point.x
      smoothed_y = alpha * new_y + (1 - alpha) * point.y

      # Check if still in image bounds
      h, w = new_frame.shape[:2]
      if 0 <= smoothed_x < w and 0 <= smoothed_y < h:
        updated.append(
          LanePoint(
            x=smoothed_x,
            y=smoothed_y,
            confidence=point.confidence * 0.98,  # Decay confidence
            visible=True,
            lane_id=point.lane_id,
          )
        )

    return updated

  def _overlaps_existing(
    self,
    new_lane: LaneLine,
    existing_lanes: list[LaneLine],
  ) -> bool:
    """Check if new lane overlaps with existing lanes.

    Args:
      new_lane: New lane to check
      existing_lanes: Existing tracked lanes

    Returns:
      True if overlap detected
    """
    if not new_lane.points:
      return False

    new_xs = [p.x for p in new_lane.points]
    new_x_mean = np.mean(new_xs)

    for existing in existing_lanes:
      if not existing.points:
        continue
      existing_x_mean = np.mean([p.x for p in existing.points])
      if abs(new_x_mean - existing_x_mean) < 50:  # Threshold in pixels
        return True

    return False

  def reset(self) -> None:
    """Reset tracking state."""
    self._lanes = []
    self._prev_frame = None
    self._initialized = False
