"""Feature visualization utilities for vision transformers.

Provides tools for visualizing DINOv2 and other transformer features
including PCA visualization, attention maps, and feature grids.
"""

from __future__ import annotations

import math

import numpy as np

# Check optional dependencies
try:
  import cv2

  CV2_AVAILABLE = True
except ImportError:
  CV2_AVAILABLE = False


def visualize_features(
  features: np.ndarray,
  image_size: tuple[int, int] | None = None,
  colormap: int | None = None,
) -> np.ndarray:
  """Visualize feature map as heatmap.

  Args:
    features: Feature map [H, W] or [H, W, C]
    image_size: Target size for visualization (H, W)
    colormap: OpenCV colormap constant (default: COLORMAP_VIRIDIS)

  Returns:
    Colorized visualization [H, W, 3]
  """
  if not CV2_AVAILABLE:
    raise ImportError("OpenCV required for visualization")

  if colormap is None:
    colormap = cv2.COLORMAP_VIRIDIS

  # Handle multi-channel features
  if features.ndim == 3:
    # Take mean across channels
    features = np.mean(features, axis=-1)

  # Normalize to [0, 255]
  features = features - features.min()
  if features.max() > 0:
    features = features / features.max()
  features = (features * 255).astype(np.uint8)

  # Apply colormap
  colored = cv2.applyColorMap(features, colormap)

  # Resize if requested
  if image_size is not None:
    colored = cv2.resize(colored, (image_size[1], image_size[0]))

  return colored


def visualize_pca_features(
  features: np.ndarray,
  n_components: int = 3,
  image_size: tuple[int, int] | None = None,
) -> np.ndarray:
  """Visualize features using PCA for RGB colorization.

  Projects high-dimensional features to 3 components and maps
  them to RGB for visualization.

  Args:
    features: Patch features [N, D] or [B, N, D]
    n_components: Number of PCA components (3 for RGB)
    image_size: Target visualization size (H, W)

  Returns:
    RGB visualization [H, W, 3] uint8
  """
  # Handle batch dimension
  if features.ndim == 3:
    features = features[0]  # Take first image

  n_patches, d = features.shape

  # Compute spatial dimensions
  h = w = int(math.sqrt(n_patches))
  if h * w != n_patches:
    raise ValueError(f"Cannot reshape {n_patches} patches to square grid")

  # Center features
  features_centered = features - features.mean(axis=0)

  # Simple PCA via SVD
  u, s, vh = np.linalg.svd(features_centered, full_matrices=False)
  components = vh[:n_components]  # [n_components, D]

  # Project features
  projected = features_centered @ components.T  # [N, n_components]

  # Normalize each component to [0, 1]
  for i in range(n_components):
    pmin, pmax = projected[:, i].min(), projected[:, i].max()
    if pmax > pmin:
      projected[:, i] = (projected[:, i] - pmin) / (pmax - pmin)
    else:
      projected[:, i] = 0.5

  # Reshape to spatial
  rgb = projected.reshape(h, w, n_components)

  # Convert to uint8
  rgb = (rgb * 255).astype(np.uint8)

  # Resize if requested
  if image_size is not None:
    if not CV2_AVAILABLE:
      raise ImportError("OpenCV required for resizing")
    rgb = cv2.resize(rgb, (image_size[1], image_size[0]))

  return rgb


def visualize_attention(
  attention: np.ndarray,
  image_size: tuple[int, int] | None = None,
  head_idx: int | None = None,
  colormap: int | None = None,
) -> np.ndarray:
  """Visualize attention maps from transformer.

  Args:
    attention: Attention weights [num_heads, N, N] or [N, N]
    image_size: Target visualization size (H, W)
    head_idx: Specific head to visualize (averages if None)
    colormap: OpenCV colormap constant

  Returns:
    Attention visualization [H, W, 3]
  """
  if not CV2_AVAILABLE:
    raise ImportError("OpenCV required for visualization")

  if colormap is None:
    colormap = cv2.COLORMAP_HOT

  # Handle multi-head attention
  if attention.ndim == 3:
    if head_idx is not None:
      attention = attention[head_idx]
    else:
      attention = attention.mean(axis=0)

  # Attention is [N, N] - we want CLS token attention to patches
  # Assuming first token is CLS
  cls_attention = attention[0, 1:]  # Exclude self-attention

  # Reshape to spatial
  n_patches = len(cls_attention)
  h = w = int(math.sqrt(n_patches))
  attention_map = cls_attention.reshape(h, w)

  return visualize_features(attention_map, image_size, colormap)


def create_feature_grid(
  features: np.ndarray,
  grid_size: int = 8,
  normalize: bool = True,
) -> np.ndarray:
  """Create grid visualization of individual feature channels.

  Args:
    features: Features [N, D] or [H, W, D]
    grid_size: Number of features per row/column
    normalize: Normalize each feature independently

  Returns:
    Grid visualization [grid_H, grid_W]
  """
  # Reshape to spatial if needed
  if features.ndim == 2:
    n_patches, d = features.shape
    h = w = int(math.sqrt(n_patches))
    features = features.reshape(h, w, d)

  h, w, d = features.shape

  # Select features to show
  n_show = min(grid_size * grid_size, d)
  indices = np.linspace(0, d - 1, n_show, dtype=int)

  # Create grid
  rows = []
  for i in range(grid_size):
    row_features = []
    for j in range(grid_size):
      idx = i * grid_size + j
      if idx < len(indices):
        feat = features[:, :, indices[idx]]
        if normalize:
          feat = feat - feat.min()
          if feat.max() > 0:
            feat = feat / feat.max()
        row_features.append(feat)
      else:
        row_features.append(np.zeros((h, w)))
    rows.append(np.hstack(row_features))

  grid = np.vstack(rows)
  return grid


def overlay_features_on_image(
  image: np.ndarray,
  features: np.ndarray,
  alpha: float = 0.5,
  colormap: int | None = None,
) -> np.ndarray:
  """Overlay feature visualization on original image.

  Args:
    image: Original image [H, W, 3] RGB
    features: Feature map to overlay
    alpha: Blending factor (0=image only, 1=features only)
    colormap: OpenCV colormap for features

  Returns:
    Blended visualization [H, W, 3]
  """
  if not CV2_AVAILABLE:
    raise ImportError("OpenCV required for overlay")

  # Get feature visualization at image size
  feat_vis = visualize_features(
    features,
    image_size=(image.shape[0], image.shape[1]),
    colormap=colormap,
  )

  # Convert image to BGR for OpenCV blending
  image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  # Blend
  blended = cv2.addWeighted(image_bgr, 1 - alpha, feat_vis, alpha, 0)

  # Back to RGB
  return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def compute_feature_similarity(
  features: np.ndarray,
  query_point: tuple[int, int],
) -> np.ndarray:
  """Compute similarity of all patches to a query point.

  Args:
    features: Patch features [N, D]
    query_point: (y, x) grid coordinates of query patch

  Returns:
    Similarity map [H, W]
  """
  n_patches, d = features.shape
  h = w = int(math.sqrt(n_patches))

  # Get query feature
  query_idx = query_point[0] * w + query_point[1]
  query_feat = features[query_idx]

  # Normalize features
  features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
  query_norm = query_feat / (np.linalg.norm(query_feat) + 1e-8)

  # Compute cosine similarity
  similarity = features_norm @ query_norm

  return similarity.reshape(h, w)
