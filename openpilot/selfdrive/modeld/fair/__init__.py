"""FAIR-enhanced perception modules.

Provides perception enhancements using FAIR model technology,
optimized for real-time inference on comma devices.
"""

from openpilot.selfdrive.modeld.fair.depth import (
  DepthEstimator,
  DepthConfig,
)
from openpilot.selfdrive.modeld.fair.segmentation import (
  SegmentationModule,
  SegmentationConfig,
)
from openpilot.selfdrive.modeld.fair.lane_tracking import (
  LaneTracker,
  LaneTrackingConfig,
)

__all__ = [
  "DepthEstimator",
  "DepthConfig",
  "SegmentationModule",
  "SegmentationConfig",
  "LaneTracker",
  "LaneTrackingConfig",
]
