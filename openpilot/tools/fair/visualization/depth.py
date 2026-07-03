"""Depth visualization utilities.

Provides tools for visualizing depth maps and creating
depth overlays on images.
"""

from __future__ import annotations

import numpy as np

# Check optional dependencies
try:
  import cv2

  CV2_AVAILABLE = True
except ImportError:
  CV2_AVAILABLE = False


def depth_to_colormap(
  depth: np.ndarray,
  min_depth: float | None = None,
  max_depth: float | None = None,
  colormap: int | None = None,
  invalid_mask: np.ndarray | None = None,
) -> np.ndarray:
  """Convert depth map to colorized visualization.

  Args:
    depth: Depth map [H, W] in meters
    min_depth: Minimum depth for normalization (uses data min if None)
    max_depth: Maximum depth for normalization (uses data max if None)
    colormap: OpenCV colormap (default: COLORMAP_MAGMA)
    invalid_mask: Mask for invalid pixels [H, W]

  Returns:
    Colorized depth [H, W, 3] BGR
  """
  if not CV2_AVAILABLE:
    raise ImportError("OpenCV required for depth visualization")

  if colormap is None:
    colormap = cv2.COLORMAP_MAGMA

  # Handle min/max
  if min_depth is None:
    min_depth = np.nanmin(depth)
  if max_depth is None:
    max_depth = np.nanmax(depth)

  # Normalize depth
  depth_norm = (depth - min_depth) / (max_depth - min_depth + 1e-8)
  depth_norm = np.clip(depth_norm, 0, 1)

  # Invert so closer = brighter
  depth_norm = 1.0 - depth_norm

  # Convert to uint8
  depth_uint8 = (depth_norm * 255).astype(np.uint8)

  # Apply colormap
  colored = cv2.applyColorMap(depth_uint8, colormap)

  # Handle invalid pixels
  if invalid_mask is not None:
    colored[invalid_mask] = [0, 0, 0]  # Black for invalid

  return colored


def visualize_depth(
  depth: np.ndarray,
  min_depth: float = 0.1,
  max_depth: float = 100.0,
  colormap: str = "magma",
) -> np.ndarray:
  """Visualize depth map with named colormap.

  Args:
    depth: Depth map [H, W]
    min_depth: Minimum depth
    max_depth: Maximum depth
    colormap: Colormap name ('magma', 'viridis', 'turbo', 'jet')

  Returns:
    RGB visualization [H, W, 3]
  """
  if not CV2_AVAILABLE:
    raise ImportError("OpenCV required")

  colormap_dict = {
    "magma": cv2.COLORMAP_MAGMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
  }

  cv_colormap = colormap_dict.get(colormap, cv2.COLORMAP_MAGMA)
  colored_bgr = depth_to_colormap(depth, min_depth, max_depth, cv_colormap)

  # Convert BGR to RGB
  return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def create_depth_overlay(
  image: np.ndarray,
  depth: np.ndarray,
  alpha: float = 0.5,
  min_depth: float = 0.1,
  max_depth: float = 100.0,
  colormap: str = "magma",
) -> np.ndarray:
  """Create overlay of depth visualization on original image.

  Args:
    image: Original image [H, W, 3] RGB
    depth: Depth map [H, W]
    alpha: Blending factor (0=image, 1=depth)
    min_depth: Minimum depth for colormap
    max_depth: Maximum depth for colormap
    colormap: Colormap name

  Returns:
    Blended image [H, W, 3] RGB
  """
  if not CV2_AVAILABLE:
    raise ImportError("OpenCV required")

  # Resize depth to image size if needed
  if depth.shape != image.shape[:2]:
    depth = cv2.resize(depth, (image.shape[1], image.shape[0]))

  # Get depth visualization
  depth_vis = visualize_depth(depth, min_depth, max_depth, colormap)

  # Blend
  blended = (1 - alpha) * image + alpha * depth_vis
  return blended.astype(np.uint8)


def create_depth_comparison(
  image: np.ndarray,
  depth_pred: np.ndarray,
  depth_gt: np.ndarray | None = None,
  min_depth: float = 0.1,
  max_depth: float = 100.0,
) -> np.ndarray:
  """Create side-by-side depth comparison visualization.

  Args:
    image: Original image [H, W, 3]
    depth_pred: Predicted depth [H, W]
    depth_gt: Ground truth depth [H, W] (optional)
    min_depth: Minimum depth
    max_depth: Maximum depth

  Returns:
    Comparison image [H, W*2 or W*3, 3]
  """
  if not CV2_AVAILABLE:
    raise ImportError("OpenCV required")

  h, w = image.shape[:2]

  # Resize depth maps to image size
  if depth_pred.shape != (h, w):
    depth_pred = cv2.resize(depth_pred, (w, h))

  # Get depth visualizations
  pred_vis = visualize_depth(depth_pred, min_depth, max_depth)

  if depth_gt is not None:
    if depth_gt.shape != (h, w):
      depth_gt = cv2.resize(depth_gt, (w, h))
    gt_vis = visualize_depth(depth_gt, min_depth, max_depth)

    # Concatenate: image | pred | gt
    comparison = np.hstack([image, pred_vis, gt_vis])
  else:
    # Concatenate: image | pred
    comparison = np.hstack([image, pred_vis])

  return comparison


def compute_depth_error_map(
  depth_pred: np.ndarray,
  depth_gt: np.ndarray,
  metric: str = "abs_rel",
) -> np.ndarray:
  """Compute per-pixel depth error map.

  Args:
    depth_pred: Predicted depth [H, W]
    depth_gt: Ground truth depth [H, W]
    metric: Error metric ('abs_rel', 'sq_rel', 'rmse', 'log')

  Returns:
    Error map [H, W]
  """
  # Handle size mismatch
  if depth_pred.shape != depth_gt.shape:
    if CV2_AVAILABLE:
      depth_pred = cv2.resize(depth_pred, (depth_gt.shape[1], depth_gt.shape[0]))
    else:
      raise ValueError("Depth shapes must match without OpenCV")

  # Compute error
  if metric == "abs_rel":
    error = np.abs(depth_pred - depth_gt) / (depth_gt + 1e-8)
  elif metric == "sq_rel":
    error = (depth_pred - depth_gt) ** 2 / (depth_gt + 1e-8)
  elif metric == "rmse":
    error = np.sqrt((depth_pred - depth_gt) ** 2)
  elif metric == "log":
    error = np.abs(np.log(depth_pred + 1e-8) - np.log(depth_gt + 1e-8))
  else:
    raise ValueError(f"Unknown metric: {metric}")

  return error


def visualize_depth_error(
  depth_pred: np.ndarray,
  depth_gt: np.ndarray,
  metric: str = "abs_rel",
  max_error: float | None = None,
) -> np.ndarray:
  """Visualize depth prediction error.

  Args:
    depth_pred: Predicted depth
    depth_gt: Ground truth depth
    metric: Error metric
    max_error: Maximum error for colormap scaling

  Returns:
    Error visualization [H, W, 3] RGB
  """
  if not CV2_AVAILABLE:
    raise ImportError("OpenCV required")

  error = compute_depth_error_map(depth_pred, depth_gt, metric)

  # Clip and normalize
  if max_error is None:
    max_error = np.percentile(error, 95)

  error_norm = np.clip(error / max_error, 0, 1)
  error_uint8 = (error_norm * 255).astype(np.uint8)

  # Use hot colormap (red = high error)
  colored = cv2.applyColorMap(error_uint8, cv2.COLORMAP_HOT)
  return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
