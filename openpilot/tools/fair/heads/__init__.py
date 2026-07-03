"""Prediction heads for FAIR models.

Provides task-specific heads that can be attached to FAIR model backbones.
"""

from openpilot.tools.fair.heads.depth import (
  DepthHead,
  DPTDepthHead,
  LinearDepthHead,
)

__all__ = [
  "DepthHead",
  "DPTDepthHead",
  "LinearDepthHead",
]
