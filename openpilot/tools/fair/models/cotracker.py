"""CoTracker model wrapper.

CoTracker is a transformer-based point tracking model that can track
any point through video with high accuracy.

Features:
- Dense point tracking through video
- Occlusion-aware tracking
- Bidirectional tracking
- Joint tracking of multiple points

Reference: https://github.com/facebookresearch/co-tracker
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from openpilot.tools.fair.models.base import ModelConfig, ModelWrapper

# Check CoTracker availability
try:
  import torch

  COTRACKER_AVAILABLE = True
except ImportError:
  COTRACKER_AVAILABLE = False


@dataclass
class CoTrackerConfig(ModelConfig):
  """CoTracker configuration.

  Attributes:
    model_name: Model variant ('cotracker2', 'cotracker3')
    grid_size: Grid size for dense tracking
    window_len: Sliding window length for video
  """

  model_name: str = "cotracker2"
  grid_size: int = 10
  window_len: int = 8


@dataclass
class TrackingResult:
  """Result from point tracking.

  Attributes:
    tracks: Point trajectories [B, T, N, 2] (x, y coordinates)
    visibility: Visibility flags [B, T, N] (True if visible)
    confidence: Tracking confidence [B, T, N]
  """

  tracks: np.ndarray
  visibility: np.ndarray
  confidence: np.ndarray | None = None


class CoTrackerWrapper(ModelWrapper):
  """Wrapper for CoTracker point tracking model.

  Provides point tracking through video sequences.

  Usage:
    config = CoTrackerConfig(model_name="cotracker2")
    model = CoTrackerWrapper(config)

    with model:
      # Track specific points
      tracks = model.track_points(video, points)

      # Dense grid tracking
      tracks = model.track_grid(video)
  """

  def __init__(self, config: CoTrackerConfig | None = None):
    """Initialize CoTracker wrapper.

    Args:
      config: Model configuration
    """
    super().__init__(config or CoTrackerConfig())

  def load(self) -> None:
    """Load CoTracker model from torch hub."""
    if not COTRACKER_AVAILABLE:
      raise ImportError("PyTorch is required for CoTracker. Install with: pip install torch torchvision")

    if self._loaded:
      return

    import torch

    device = self._resolve_device()

    # Load from torch hub
    self._model = torch.hub.load("facebookresearch/co-tracker", self.config.model_name)
    self._model = self._model.to(device)
    self._model.eval()

    self._loaded = True

  def unload(self) -> None:
    """Unload model to free memory."""
    self._model = None
    self._loaded = False

    if COTRACKER_AVAILABLE:
      import torch

      if torch.cuda.is_available():
        torch.cuda.empty_cache()

  def forward(self, inputs: np.ndarray) -> dict[str, Any]:
    """Run forward pass with grid tracking.

    Args:
      inputs: Video as [B, T, H, W, C] numpy array

    Returns:
      Dictionary with 'tracks' and 'visibility'
    """
    result = self.track_grid(inputs)
    return {
      "tracks": result.tracks,
      "visibility": result.visibility,
    }

  def extract_features(self, inputs: np.ndarray) -> np.ndarray:
    """Extract point trajectories as features.

    Args:
      inputs: Video as [B, T, H, W, C] numpy array

    Returns:
      Point tracks as features
    """
    result = self.track_grid(inputs)
    return result.tracks

  def track_points(
    self,
    video: np.ndarray,
    points: np.ndarray,
    backward_tracking: bool = False,
  ) -> TrackingResult:
    """Track specific points through video.

    Args:
      video: Video as [B, T, H, W, C] numpy array (RGB, 0-255)
      points: Query points as [B, N, 3] where each point is [t, x, y]
              t is the frame index where the point appears
      backward_tracking: Also track backwards from query points

    Returns:
      TrackingResult with tracks and visibility
    """
    if not self._loaded:
      raise RuntimeError("Model not loaded")

    import torch

    device = self._resolve_device()

    # Preprocess video
    video_tensor = self._preprocess_video(video).to(device)
    queries = torch.from_numpy(points).float().to(device)

    with torch.no_grad():
      pred_tracks, pred_visibility = self._model(
        video_tensor,
        queries=queries,
        backward_tracking=backward_tracking,
      )

    return TrackingResult(
      tracks=pred_tracks.cpu().numpy(),
      visibility=pred_visibility.cpu().numpy(),
    )

  def track_grid(
    self,
    video: np.ndarray,
    grid_size: int | None = None,
    backward_tracking: bool = False,
  ) -> TrackingResult:
    """Track dense grid of points through video.

    Args:
      video: Video as [B, T, H, W, C] numpy array (RGB, 0-255)
      grid_size: Grid size (default: from config)
      backward_tracking: Also track backwards

    Returns:
      TrackingResult with tracks and visibility
    """
    if not self._loaded:
      raise RuntimeError("Model not loaded")

    import torch

    device = self._resolve_device()
    grid_size = grid_size or self.config.grid_size

    # Preprocess video
    video_tensor = self._preprocess_video(video).to(device)

    with torch.no_grad():
      pred_tracks, pred_visibility = self._model(
        video_tensor,
        grid_size=grid_size,
        backward_tracking=backward_tracking,
      )

    return TrackingResult(
      tracks=pred_tracks.cpu().numpy(),
      visibility=pred_visibility.cpu().numpy(),
    )

  def track_lane_points(
    self,
    video: np.ndarray,
    lane_points: np.ndarray,
  ) -> TrackingResult:
    """Track lane marking points through video.

    Convenience method for lane tracking use case.

    Args:
      video: Video as [B, T, H, W, C] numpy array
      lane_points: Lane points in first frame [N, 2] as (x, y)

    Returns:
      TrackingResult with tracked lane points
    """
    # Convert lane points to query format [B, N, 3] with t=0
    b = video.shape[0]
    n = lane_points.shape[0]

    queries = np.zeros((b, n, 3))
    queries[..., 0] = 0  # Frame 0
    queries[..., 1:] = lane_points

    return self.track_points(video, queries)

  def _preprocess_video(self, video: np.ndarray) -> torch.Tensor:
    """Preprocess video for CoTracker.

    Args:
      video: Video as [B, T, H, W, C] numpy array (RGB, 0-255)

    Returns:
      Preprocessed torch tensor [B, T, C, H, W]
    """
    import torch

    # Convert to float and normalize to [0, 1]
    x = torch.from_numpy(video).float() / 255.0

    # BTHWC -> BTCHW
    x = x.permute(0, 1, 4, 2, 3)

    return x
