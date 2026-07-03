"""Dataset utilities for FAIR model training.

Provides dataset classes for loading openpilot route data
and preparing it for perception model training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Iterator

import numpy as np

# Check PyTorch availability
try:
  import torch
  from torch.utils.data import Dataset, IterableDataset

  TORCH_AVAILABLE = True
except ImportError:
  TORCH_AVAILABLE = False
  Dataset = object
  IterableDataset = object


@dataclass
class FrameData:
  """Single frame data for training.

  Attributes:
    image: RGB image [H, W, 3]
    timestamp: Frame timestamp in nanoseconds
    depth: Optional depth map [H, W]
    segmentation: Optional segmentation mask [H, W]
    detections: Optional detection boxes [N, 5] (x1, y1, x2, y2, class)
    pose: Optional ego pose [4, 4]
    calibration: Optional camera calibration
  """

  image: np.ndarray
  timestamp: int
  depth: np.ndarray | None = None
  segmentation: np.ndarray | None = None
  detections: np.ndarray | None = None
  pose: np.ndarray | None = None
  calibration: dict[str, Any] | None = None


class RouteDataset(Dataset):
  """Dataset for loading frames from openpilot routes.

  Loads camera frames and associated data from route logs.
  """

  def __init__(
    self,
    route_paths: list[Path | str],
    camera: str = "road",
    image_size: tuple[int, int] = (256, 512),
    augment: bool = True,
    load_depth: bool = False,
    load_segmentation: bool = False,
  ):
    """Initialize route dataset.

    Args:
      route_paths: List of paths to route directories
      camera: Camera to load ('road', 'driver', 'wide')
      image_size: Target image size (H, W)
      augment: Apply data augmentation
      load_depth: Load depth pseudo-labels if available
      load_segmentation: Load segmentation pseudo-labels if available
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required for dataset")

    self.route_paths = [Path(p) for p in route_paths]
    self.camera = camera
    self.image_size = image_size
    self.augment = augment
    self.load_depth = load_depth
    self.load_segmentation = load_segmentation

    # Index all frames
    self.frame_index = self._build_index()

  def _build_index(self) -> list[tuple[Path, int]]:
    """Build index of all frames across routes."""
    index = []

    for route_path in self.route_paths:
      # Try to find frame count from metadata
      hevc_path = route_path / f"{self.camera}camera.hevc"
      if hevc_path.exists():
        # Estimate frame count (would need proper video loading)
        # For now, use a fixed estimate or metadata
        frame_count = self._estimate_frame_count(route_path)
        for i in range(frame_count):
          index.append((route_path, i))

    return index

  def _estimate_frame_count(self, route_path: Path) -> int:
    """Estimate number of frames in a route.

    In practice, this would read from route metadata or
    probe the video file.
    """
    # Default estimate: 20 FPS * 60 seconds per segment
    return 1200

  def __len__(self) -> int:
    """Return number of frames."""
    return len(self.frame_index)

  def __getitem__(self, idx: int) -> dict[str, Any]:
    """Load frame at index.

    Args:
      idx: Frame index

    Returns:
      Dictionary with image and optional labels
    """
    route_path, frame_idx = self.frame_index[idx]

    # Load frame (placeholder - would use actual video loading)
    frame_data = self._load_frame(route_path, frame_idx)

    # Preprocess
    image = self._preprocess_image(frame_data.image)

    # Build output dict
    output = {
      "image": torch.from_numpy(image).float(),
      "timestamp": frame_data.timestamp,
    }

    if self.load_depth and frame_data.depth is not None:
      depth = self._preprocess_depth(frame_data.depth)
      output["depth_target"] = torch.from_numpy(depth).float()

    if self.load_segmentation and frame_data.segmentation is not None:
      seg = self._preprocess_segmentation(frame_data.segmentation)
      output["segmentation_target"] = torch.from_numpy(seg).long()

    if frame_data.detections is not None:
      output["detection_target"] = torch.from_numpy(frame_data.detections).float()

    return output

  def _load_frame(self, route_path: Path, frame_idx: int) -> FrameData:
    """Load a single frame from route.

    In practice, this would:
    1. Decode the HEVC video at the given frame
    2. Load corresponding log data
    3. Load pseudo-labels if available
    """
    # Placeholder implementation
    h, w = self.image_size
    image = np.zeros((h, w, 3), dtype=np.uint8)

    depth = None
    if self.load_depth:
      depth_path = route_path / "depth" / f"{frame_idx:06d}.npy"
      if depth_path.exists():
        depth = np.load(depth_path)

    segmentation = None
    if self.load_segmentation:
      seg_path = route_path / "segmentation" / f"{frame_idx:06d}.npy"
      if seg_path.exists():
        segmentation = np.load(seg_path)

    return FrameData(
      image=image,
      timestamp=frame_idx * 50_000_000,  # 20 FPS
      depth=depth,
      segmentation=segmentation,
    )

  def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
    """Preprocess image for training.

    Args:
      image: Input image [H, W, 3] uint8

    Returns:
      Preprocessed image [3, H, W] float32
    """
    import cv2

    # Resize if needed
    if image.shape[:2] != self.image_size:
      image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Apply augmentation
    if self.augment:
      image = self._augment_image(image)

    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))

    return image

  def _preprocess_depth(self, depth: np.ndarray) -> np.ndarray:
    """Preprocess depth map."""
    import cv2

    if depth.shape[:2] != self.image_size:
      depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]))

    # Normalize depth to [0, 1] assuming max depth of 100m
    depth = np.clip(depth / 100.0, 0, 1)

    return depth[np.newaxis, ...]  # Add channel dim

  def _preprocess_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
    """Preprocess segmentation mask."""
    import cv2

    if segmentation.shape[:2] != self.image_size:
      segmentation = cv2.resize(
        segmentation,
        (self.image_size[1], self.image_size[0]),
        interpolation=cv2.INTER_NEAREST,
      )

    return segmentation

  def _augment_image(self, image: np.ndarray) -> np.ndarray:
    """Apply data augmentation.

    Args:
      image: Input image [H, W, 3] float32

    Returns:
      Augmented image
    """
    # Random horizontal flip
    if np.random.random() < 0.5:
      image = np.fliplr(image).copy()

    # Random brightness/contrast
    if np.random.random() < 0.5:
      brightness = np.random.uniform(0.8, 1.2)
      contrast = np.random.uniform(0.8, 1.2)
      image = np.clip(contrast * (image - 0.5) + 0.5 + (brightness - 1), 0, 1)

    # Random color jitter
    if np.random.random() < 0.3:
      for c in range(3):
        image[..., c] = np.clip(image[..., c] * np.random.uniform(0.9, 1.1), 0, 1)

    return image


class StreamingRouteDataset(IterableDataset):
  """Streaming dataset for large-scale route training.

  Loads frames on-the-fly without building a full index,
  suitable for training on many routes.
  """

  def __init__(
    self,
    route_paths: list[Path | str],
    camera: str = "road",
    image_size: tuple[int, int] = (256, 512),
    shuffle: bool = True,
    samples_per_route: int = 100,
  ):
    """Initialize streaming dataset.

    Args:
      route_paths: List of route directories
      camera: Camera to load
      image_size: Target image size
      shuffle: Shuffle routes and frames
      samples_per_route: Number of frames to sample per route
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required for dataset")

    self.route_paths = [Path(p) for p in route_paths]
    self.camera = camera
    self.image_size = image_size
    self.shuffle = shuffle
    self.samples_per_route = samples_per_route

  def __iter__(self) -> Iterator[dict[str, Any]]:
    """Iterate over frames."""
    routes = list(self.route_paths)

    if self.shuffle:
      np.random.shuffle(routes)

    for route_path in routes:
      # Sample frames from this route
      frame_count = self._estimate_frame_count(route_path)
      if self.shuffle:
        indices = np.random.choice(
          frame_count,
          min(self.samples_per_route, frame_count),
          replace=False,
        )
      else:
        step = max(1, frame_count // self.samples_per_route)
        indices = np.arange(0, frame_count, step)[: self.samples_per_route]

      for frame_idx in indices:
        yield self._load_frame(route_path, int(frame_idx))

  def _estimate_frame_count(self, route_path: Path) -> int:
    """Estimate frames in route."""
    return 1200  # Default estimate

  def _load_frame(self, route_path: Path, frame_idx: int) -> dict[str, Any]:
    """Load and preprocess a frame."""
    # Placeholder
    h, w = self.image_size
    image = np.zeros((3, h, w), dtype=np.float32)

    return {
      "image": torch.from_numpy(image),
      "timestamp": frame_idx * 50_000_000,
    }


def create_pseudo_labels(
  route_path: Path,
  model: Any,
  task: str,
  output_dir: Path | None = None,
) -> None:
  """Generate pseudo-labels for a route using a teacher model.

  Args:
    route_path: Path to route directory
    model: Teacher model for label generation
    task: Task type ('depth', 'segmentation', 'detection')
    output_dir: Output directory (default: route_path/task)
  """
  if output_dir is None:
    output_dir = route_path / task

  output_dir.mkdir(parents=True, exist_ok=True)

  # This would iterate through frames and save predictions
  # Placeholder for actual implementation
