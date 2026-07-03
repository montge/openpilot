"""Semantic segmentation using SAM-guided training.

Provides efficient semantic segmentation for driving scenes,
trained with SAM2 as a teacher model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


class DrivingClass(IntEnum):
  """Driving-relevant semantic classes."""

  ROAD = 0
  LANE_MARKING = 1
  VEHICLE = 2
  PEDESTRIAN = 3
  CYCLIST = 4
  BUILDING = 5
  VEGETATION = 6
  SKY = 7
  SIGN = 8
  POLE = 9
  UNKNOWN = 255


@dataclass
class SegmentationConfig:
  """Segmentation configuration.

  Attributes:
    model_path: Path to segmentation model weights
    input_size: Expected input size (H, W)
    output_size: Output mask size (H, W)
    num_classes: Number of segmentation classes
    use_gpu: Use GPU acceleration if available
  """

  model_path: str | None = None
  input_size: tuple[int, int] = (256, 512)
  output_size: tuple[int, int] = (128, 256)
  num_classes: int = 10
  use_gpu: bool = True


class SegmentationModule:
  """Semantic segmentation for driving scenes.

  Uses a SAM-distilled model optimized for driving-relevant classes.

  Usage:
    config = SegmentationConfig(model_path="path/to/model.onnx")
    segmenter = SegmentationModule(config)

    mask = segmenter.segment(image)
    road_mask = segmenter.get_road_mask(image)
  """

  def __init__(self, config: SegmentationConfig | None = None):
    """Initialize segmentation module.

    Args:
      config: Segmentation configuration
    """
    self.config = config or SegmentationConfig()
    self._model = None
    self._loaded = False

  def load(self) -> None:
    """Load segmentation model."""
    if self._loaded:
      return

    # In production, load ONNX/TensorRT model
    self._loaded = True

  def unload(self) -> None:
    """Unload model to free memory."""
    self._model = None
    self._loaded = False

  def segment(self, image: np.ndarray) -> np.ndarray:
    """Segment image into semantic classes.

    Args:
      image: RGB image [H, W, 3] uint8

    Returns:
      Segmentation mask [H, W] with class indices
    """
    if not self._loaded:
      raise RuntimeError("Model not loaded. Call load() first.")

    # Preprocess
    processed = self._preprocess(image)

    # Run inference
    logits = self._infer(processed)

    # Get class predictions
    mask = np.argmax(logits, axis=0)

    return mask.astype(np.uint8)

  def segment_probs(self, image: np.ndarray) -> np.ndarray:
    """Segment image with probability outputs.

    Args:
      image: RGB image [H, W, 3] uint8

    Returns:
      Class probabilities [C, H, W]
    """
    if not self._loaded:
      raise RuntimeError("Model not loaded. Call load() first.")

    processed = self._preprocess(image)
    logits = self._infer(processed)

    # Softmax to probabilities
    exp_logits = np.exp(logits - logits.max(axis=0, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)

    return probs

  def get_road_mask(self, image: np.ndarray) -> np.ndarray:
    """Get binary road mask.

    Args:
      image: RGB image [H, W, 3] uint8

    Returns:
      Binary road mask [H, W] (True where road)
    """
    mask = self.segment(image)
    return mask == DrivingClass.ROAD

  def get_lane_mask(self, image: np.ndarray) -> np.ndarray:
    """Get binary lane marking mask.

    Args:
      image: RGB image [H, W, 3] uint8

    Returns:
      Binary lane marking mask [H, W]
    """
    mask = self.segment(image)
    return mask == DrivingClass.LANE_MARKING

  def get_vehicle_mask(self, image: np.ndarray) -> np.ndarray:
    """Get binary vehicle mask.

    Args:
      image: RGB image [H, W, 3] uint8

    Returns:
      Binary vehicle mask [H, W]
    """
    mask = self.segment(image)
    return mask == DrivingClass.VEHICLE

  def get_drivable_area(self, image: np.ndarray) -> np.ndarray:
    """Get drivable area mask (road + lane markings).

    Args:
      image: RGB image [H, W, 3] uint8

    Returns:
      Binary drivable area mask [H, W]
    """
    mask = self.segment(image)
    return (mask == DrivingClass.ROAD) | (mask == DrivingClass.LANE_MARKING)

  def _preprocess(self, image: np.ndarray) -> np.ndarray:
    """Preprocess image for inference."""
    import cv2

    h, w = self.config.input_size

    # Resize
    if image.shape[:2] != (h, w):
      image = cv2.resize(image, (w, h))

    # Normalize
    image = image.astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))

    return image

  def _infer(self, processed: np.ndarray) -> np.ndarray:
    """Run model inference.

    Args:
      processed: Preprocessed image [C, H, W]

    Returns:
      Logits [num_classes, H, W]
    """
    # Placeholder - returns dummy logits
    h, w = self.config.output_size
    c = self.config.num_classes

    # Create dummy logits favoring road in bottom half
    logits = np.zeros((c, h, w), dtype=np.float32)
    logits[DrivingClass.SKY, : h // 3, :] = 2.0  # Sky at top
    logits[DrivingClass.ROAD, h // 2 :, :] = 2.0  # Road at bottom
    logits[DrivingClass.VEGETATION, h // 3 : h // 2, :] = 1.0  # Vegetation in middle

    return logits

  def __enter__(self) -> SegmentationModule:
    """Context manager entry."""
    self.load()
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    """Context manager exit."""
    self.unload()
