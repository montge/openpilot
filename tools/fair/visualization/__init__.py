"""Visualization utilities for FAIR models.

Provides tools for visualizing model features, attention maps,
and predictions.
"""

from openpilot.tools.fair.visualization.features import (
  visualize_features,
  visualize_attention,
  visualize_pca_features,
  create_feature_grid,
)
from openpilot.tools.fair.visualization.depth import (
  visualize_depth,
  depth_to_colormap,
  create_depth_overlay,
)

__all__ = [
  "visualize_features",
  "visualize_attention",
  "visualize_pca_features",
  "create_feature_grid",
  "visualize_depth",
  "depth_to_colormap",
  "create_depth_overlay",
]
