"""SAM 2 (Segment Anything Model 2) wrapper.

SAM 2 extends SAM with video capabilities for tracking objects through video.

Features:
- Video object segmentation
- Promptable segmentation (points, boxes, text)
- Memory-based object tracking
- Occlusion handling

Reference: https://github.com/facebookresearch/segment-anything-2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openpilot.tools.fair.models.base import ModelConfig, ModelWrapper

# Check SAM 2 availability
try:
  import torch  # noqa: F401

  SAM2_AVAILABLE = True
except ImportError:
  SAM2_AVAILABLE = False


@dataclass
class SAM2Config(ModelConfig):
  """SAM 2 configuration.

  Attributes:
    model_name: Model variant ('sam2_hiera_tiny', 'sam2_hiera_small',
                               'sam2_hiera_base_plus', 'sam2_hiera_large')
    points_per_side: Points per side for automatic mask generation
    pred_iou_thresh: IoU threshold for filtering predictions
    stability_score_thresh: Stability threshold
  """

  model_name: str = "sam2_hiera_base_plus"
  points_per_side: int = 32
  pred_iou_thresh: float = 0.88
  stability_score_thresh: float = 0.95


@dataclass
class SegmentationResult:
  """Result from segmentation.

  Attributes:
    masks: Binary masks [N, H, W]
    scores: Confidence scores [N]
    boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
  """

  masks: np.ndarray
  scores: np.ndarray
  boxes: np.ndarray | None = None


@dataclass
class VideoTrackingState:
  """State for video object tracking.

  Attributes:
    object_ids: List of tracked object IDs
    masks: Current masks for each object
    scores: Current scores for each object
    memory: Internal memory state
  """

  object_ids: list[int] = field(default_factory=list)
  masks: dict[int, np.ndarray] = field(default_factory=dict)
  scores: dict[int, float] = field(default_factory=dict)
  memory: Any = None


class SAM2Wrapper(ModelWrapper):
  """Wrapper for SAM 2 video segmentation model.

  Provides image and video segmentation with tracking capabilities.

  Usage:
    config = SAM2Config(model_name="sam2_hiera_base_plus")
    model = SAM2Wrapper(config)

    with model:
      # Single image segmentation
      result = model.segment_image(image, points=[[100, 200]])

      # Video tracking
      model.init_video_tracking(first_frame, initial_masks)
      for frame in video_frames:
        result = model.track_video_frame(frame)
  """

  def __init__(self, config: SAM2Config | None = None):
    """Initialize SAM 2 wrapper.

    Args:
      config: Model configuration
    """
    super().__init__(config or SAM2Config())
    self._predictor = None
    self._video_predictor = None
    self._tracking_state: VideoTrackingState | None = None

  def load(self) -> None:
    """Load SAM 2 model."""
    if not SAM2_AVAILABLE:
      raise ImportError("PyTorch is required for SAM 2. Install with: pip install torch torchvision")

    if self._loaded:
      return

    # Note: SAM 2 requires sam2 package
    # pip install git+https://github.com/facebookresearch/segment-anything-2.git
    try:
      from sam2.build_sam import build_sam2
      from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as err:
      raise ImportError("SAM 2 package required. Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git") from err

    device = self._resolve_device()
    model_name = self.config.model_name

    # Map model names to configs
    config_map = {
      "sam2_hiera_tiny": "sam2_hiera_t.yaml",
      "sam2_hiera_small": "sam2_hiera_s.yaml",
      "sam2_hiera_base_plus": "sam2_hiera_b+.yaml",
      "sam2_hiera_large": "sam2_hiera_l.yaml",
    }

    config_file = config_map.get(model_name, "sam2_hiera_b+.yaml")

    self._model = build_sam2(config_file, device=device)
    self._predictor = SAM2ImagePredictor(self._model)
    self._loaded = True

  def unload(self) -> None:
    """Unload model to free memory."""
    self._model = None
    self._predictor = None
    self._video_predictor = None
    self._tracking_state = None
    self._loaded = False

    if SAM2_AVAILABLE:
      import torch

      if torch.cuda.is_available():
        torch.cuda.empty_cache()

  def forward(self, inputs: np.ndarray) -> dict[str, Any]:
    """Run forward pass for automatic mask generation.

    Args:
      inputs: Single image as [H, W, C] numpy array (RGB, 0-255)

    Returns:
      Dictionary with 'masks', 'scores', 'boxes'
    """
    if not self._loaded:
      raise RuntimeError("Model not loaded")

    self._predictor.set_image(inputs)

    # Automatic mask generation
    masks, scores, _ = self._predictor.predict(
      point_coords=None,
      point_labels=None,
      multimask_output=True,
    )

    return {
      "masks": masks,
      "scores": scores,
      "boxes": None,
    }

  def extract_features(self, inputs: np.ndarray) -> np.ndarray:
    """Extract image encoder features.

    Args:
      inputs: Images as [B, H, W, C] numpy array

    Returns:
      Features from image encoder
    """
    if not self._loaded:
      raise RuntimeError("Model not loaded")

    # Process single image for simplicity
    if len(inputs.shape) == 4:
      inputs = inputs[0]

    self._predictor.set_image(inputs)
    features = self._predictor.get_image_embedding()

    return features.cpu().numpy()

  def segment_image(
    self,
    image: np.ndarray,
    points: list[list[int]] | None = None,
    point_labels: list[int] | None = None,
    boxes: list[list[int]] | None = None,
    mask_input: np.ndarray | None = None,
    multimask_output: bool = True,
  ) -> SegmentationResult:
    """Segment image with prompts.

    Args:
      image: Image as [H, W, C] numpy array (RGB, 0-255)
      points: Point prompts as [[x, y], ...] (foreground clicks)
      point_labels: Labels for points (1=foreground, 0=background)
      boxes: Box prompts as [[x1, y1, x2, y2], ...]
      mask_input: Low-res mask input from previous prediction
      multimask_output: Whether to output multiple masks

    Returns:
      SegmentationResult with masks and scores
    """
    if not self._loaded:
      raise RuntimeError("Model not loaded")

    self._predictor.set_image(image)

    # Convert prompts to numpy arrays
    point_coords = np.array(points) if points else None
    point_labels_arr = np.array(point_labels) if point_labels else None
    box = np.array(boxes[0]) if boxes else None

    masks, scores, _ = self._predictor.predict(
      point_coords=point_coords,
      point_labels=point_labels_arr,
      box=box,
      mask_input=mask_input,
      multimask_output=multimask_output,
    )

    return SegmentationResult(
      masks=masks,
      scores=scores,
      boxes=None,
    )

  def init_video_tracking(
    self,
    first_frame: np.ndarray,
    initial_masks: dict[int, np.ndarray],
  ) -> VideoTrackingState:
    """Initialize video object tracking.

    Args:
      first_frame: First frame as [H, W, C] numpy array
      initial_masks: Dictionary mapping object IDs to binary masks

    Returns:
      Initial tracking state
    """
    if not self._loaded:
      raise RuntimeError("Model not loaded")

    self._tracking_state = VideoTrackingState(
      object_ids=list(initial_masks.keys()),
      masks={k: v.copy() for k, v in initial_masks.items()},
      scores=dict.fromkeys(initial_masks.keys(), 1.0),
    )

    return self._tracking_state

  def track_video_frame(self, frame: np.ndarray) -> VideoTrackingState:
    """Track objects in next video frame.

    Args:
      frame: Current frame as [H, W, C] numpy array

    Returns:
      Updated tracking state
    """
    if self._tracking_state is None:
      raise RuntimeError("Video tracking not initialized. Call init_video_tracking() first.")

    # For each tracked object, predict mask in new frame
    for obj_id in self._tracking_state.object_ids:
      prev_mask = self._tracking_state.masks.get(obj_id)
      if prev_mask is None:
        continue

      # Use previous mask as prompt for new frame
      self._predictor.set_image(frame)

      # Get mask centroid as point prompt
      y_coords, x_coords = np.where(prev_mask > 0.5)
      if len(x_coords) > 0:
        cx = int(np.mean(x_coords))
        cy = int(np.mean(y_coords))

        masks, scores, _ = self._predictor.predict(
          point_coords=np.array([[cx, cy]]),
          point_labels=np.array([1]),
          multimask_output=False,
        )

        self._tracking_state.masks[obj_id] = masks[0]
        self._tracking_state.scores[obj_id] = float(scores[0])

    return self._tracking_state

  def reset_tracking(self) -> None:
    """Reset video tracking state."""
    self._tracking_state = None
