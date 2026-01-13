"""DETR (DEtection TRansformer) model wrapper.

DETR is an end-to-end object detection model using transformers.

Features:
- Set-based object detection (no NMS required)
- Panoptic segmentation support
- Strong performance on COCO benchmark
- Multiple variants (DETR, Deformable DETR, DINO)

Reference: https://github.com/facebookresearch/detr
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from openpilot.tools.fair.models.base import ModelConfig, ModelWrapper

# Check DETR availability
try:
  import torch

  DETR_AVAILABLE = True
except ImportError:
  DETR_AVAILABLE = False


@dataclass
class DETRConfig(ModelConfig):
  """DETR configuration.

  Attributes:
    model_name: Model variant ('detr_resnet50', 'detr_resnet101',
                               'detr_resnet50_dc5', 'detr_resnet101_dc5')
    confidence_threshold: Minimum confidence for detections
    num_classes: Number of detection classes (91 for COCO)
  """

  model_name: str = "detr_resnet50"
  confidence_threshold: float = 0.7
  num_classes: int = 91


@dataclass
class Detection:
  """Single object detection.

  Attributes:
    box: Bounding box [x1, y1, x2, y2]
    score: Detection confidence
    label: Class label (integer)
    label_name: Class name (if available)
  """

  box: np.ndarray
  score: float
  label: int
  label_name: str | None = None


@dataclass
class DetectionResult:
  """Result from object detection.

  Attributes:
    boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
    scores: Confidence scores [N]
    labels: Class labels [N]
    detections: List of Detection objects
  """

  boxes: np.ndarray
  scores: np.ndarray
  labels: np.ndarray
  detections: list[Detection]


# COCO class names for reference
COCO_CLASSES = [
  "N/A",
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "N/A",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "N/A",
  "backpack",
  "umbrella",
  "N/A",
  "N/A",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "N/A",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "N/A",
  "dining table",
  "N/A",
  "N/A",
  "toilet",
  "N/A",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "N/A",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
]


class DETRWrapper(ModelWrapper):
  """Wrapper for DETR object detection model.

  Provides object detection with optional vehicle-focused filtering.

  Usage:
    config = DETRConfig(model_name="detr_resnet50")
    model = DETRWrapper(config)

    with model:
      # Detect all objects
      result = model.detect(image)

      # Detect vehicles only
      vehicles = model.detect_vehicles(image)
  """

  # Vehicle-related class indices in COCO
  VEHICLE_CLASSES = {2, 3, 4, 6, 7, 8}  # bicycle, car, motorcycle, bus, train, truck

  def __init__(self, config: DETRConfig | None = None):
    """Initialize DETR wrapper.

    Args:
      config: Model configuration
    """
    super().__init__(config or DETRConfig())

  def load(self) -> None:
    """Load DETR model from torch hub."""
    if not DETR_AVAILABLE:
      raise ImportError("PyTorch is required for DETR. Install with: pip install torch torchvision")

    if self._loaded:
      return

    import torch

    device = self._resolve_device()

    # Load from torch hub
    self._model = torch.hub.load("facebookresearch/detr", self.config.model_name, pretrained=True)
    self._model = self._model.to(device)
    self._model.eval()

    self._loaded = True

  def unload(self) -> None:
    """Unload model to free memory."""
    self._model = None
    self._loaded = False

    if DETR_AVAILABLE:
      import torch

      if torch.cuda.is_available():
        torch.cuda.empty_cache()

  def forward(self, inputs: np.ndarray) -> dict[str, Any]:
    """Run forward pass.

    Args:
      inputs: Images as [B, H, W, C] numpy array (RGB, 0-255)

    Returns:
      Dictionary with 'pred_logits' and 'pred_boxes'
    """
    if not self._loaded:
      raise RuntimeError("Model not loaded")

    import torch

    device = self._resolve_device()

    # Preprocess
    x = self._preprocess(inputs).to(device)

    with torch.no_grad():
      outputs = self._model(x)

    return {
      "pred_logits": outputs["pred_logits"].cpu().numpy(),
      "pred_boxes": outputs["pred_boxes"].cpu().numpy(),
    }

  def extract_features(self, inputs: np.ndarray) -> np.ndarray:
    """Extract backbone features.

    Args:
      inputs: Images as [B, H, W, C] numpy array

    Returns:
      Backbone feature maps
    """
    if not self._loaded:
      raise RuntimeError("Model not loaded")

    import torch

    device = self._resolve_device()

    x = self._preprocess(inputs).to(device)

    with torch.no_grad():
      # Get backbone features
      features = self._model.backbone(x)
      # Return last feature map
      last_feature = list(features.values())[-1]

    return last_feature.tensors.cpu().numpy()

  def detect(
    self,
    image: np.ndarray,
    confidence_threshold: float | None = None,
  ) -> DetectionResult:
    """Detect objects in image.

    Args:
      image: Image as [H, W, C] numpy array (RGB, 0-255)
      confidence_threshold: Override default confidence threshold

    Returns:
      DetectionResult with detected objects
    """
    if not self._loaded:
      raise RuntimeError("Model not loaded")

    import torch

    device = self._resolve_device()
    threshold = confidence_threshold or self.config.confidence_threshold

    # Add batch dimension if needed
    if len(image.shape) == 3:
      image = image[np.newaxis, ...]

    x = self._preprocess(image).to(device)
    h, w = image.shape[1:3]

    with torch.no_grad():
      outputs = self._model(x)

    # Post-process predictions
    return self._post_process(outputs, (h, w), threshold)

  def detect_vehicles(
    self,
    image: np.ndarray,
    confidence_threshold: float | None = None,
  ) -> DetectionResult:
    """Detect vehicles in image.

    Filters detections to only include vehicle classes.

    Args:
      image: Image as [H, W, C] numpy array (RGB, 0-255)
      confidence_threshold: Override default confidence threshold

    Returns:
      DetectionResult with vehicle detections only
    """
    result = self.detect(image, confidence_threshold)

    # Filter to vehicle classes
    vehicle_mask = np.isin(result.labels, list(self.VEHICLE_CLASSES))

    if not np.any(vehicle_mask):
      return DetectionResult(
        boxes=np.array([]).reshape(0, 4),
        scores=np.array([]),
        labels=np.array([], dtype=np.int64),
        detections=[],
      )

    return DetectionResult(
      boxes=result.boxes[vehicle_mask],
      scores=result.scores[vehicle_mask],
      labels=result.labels[vehicle_mask],
      detections=[d for d in result.detections if d.label in self.VEHICLE_CLASSES],
    )

  def detect_batch(
    self,
    images: np.ndarray,
    confidence_threshold: float | None = None,
  ) -> list[DetectionResult]:
    """Detect objects in batch of images.

    Args:
      images: Images as [B, H, W, C] numpy array
      confidence_threshold: Override default confidence threshold

    Returns:
      List of DetectionResult, one per image
    """
    if not self._loaded:
      raise RuntimeError("Model not loaded")

    import torch

    device = self._resolve_device()
    threshold = confidence_threshold or self.config.confidence_threshold

    x = self._preprocess(images).to(device)
    h, w = images.shape[1:3]

    with torch.no_grad():
      outputs = self._model(x)

    # Post-process each image
    results = []
    batch_size = images.shape[0]

    for i in range(batch_size):
      batch_outputs = {
        "pred_logits": outputs["pred_logits"][i : i + 1],
        "pred_boxes": outputs["pred_boxes"][i : i + 1],
      }
      results.append(self._post_process(batch_outputs, (h, w), threshold))

    return results

  def _post_process(
    self,
    outputs: dict[str, Any],
    image_size: tuple[int, int],
    threshold: float,
  ) -> DetectionResult:
    """Post-process DETR outputs.

    Args:
      outputs: Raw model outputs
      image_size: (height, width) of input image
      threshold: Confidence threshold

    Returns:
      DetectionResult with filtered detections
    """
    import torch
    import torch.nn.functional as F

    h, w = image_size

    # Get predictions
    pred_logits = outputs["pred_logits"]  # [B, num_queries, num_classes+1]
    pred_boxes = outputs["pred_boxes"]  # [B, num_queries, 4] in cxcywh format

    # Convert logits to probabilities
    probs = F.softmax(pred_logits, dim=-1)[0, :, :-1]  # Remove no-object class
    scores, labels = probs.max(dim=-1)

    # Filter by confidence
    mask = scores > threshold
    scores = scores[mask]
    labels = labels[mask]
    boxes = pred_boxes[0, mask]

    # Convert boxes from cxcywh to xyxy
    boxes = self._box_cxcywh_to_xyxy(boxes)

    # Scale to image size
    boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)

    # Convert to numpy
    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Create Detection objects
    detections = []
    for i in range(len(boxes_np)):
      label_idx = int(labels_np[i])
      label_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else None
      detections.append(
        Detection(
          box=boxes_np[i],
          score=float(scores_np[i]),
          label=label_idx,
          label_name=label_name,
        )
      )

    return DetectionResult(
      boxes=boxes_np,
      scores=scores_np,
      labels=labels_np,
      detections=detections,
    )

  def _preprocess(self, inputs: np.ndarray) -> torch.Tensor:
    """Preprocess images for DETR.

    Args:
      inputs: Images as [B, H, W, C] numpy array (RGB, 0-255)

    Returns:
      Preprocessed torch tensor [B, C, H, W]
    """
    import torch

    # Convert to float and normalize to [0, 1]
    x = torch.from_numpy(inputs).float() / 255.0

    # BHWC -> BCHW
    x = x.permute(0, 3, 1, 2)

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std

    return x

  @staticmethod
  def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from center format to corner format.

    Args:
      boxes: Boxes in [cx, cy, w, h] format

    Returns:
      Boxes in [x1, y1, x2, y2] format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return boxes.__class__(list(zip(x1, y1, x2, y2, strict=False)))
