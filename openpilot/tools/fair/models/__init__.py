"""FAIR model wrappers.

Provides unified interfaces for Meta FAIR research models.
Models are optional dependencies - wrappers gracefully handle missing packages.
"""

from openpilot.tools.fair.models.base import ModelWrapper
from openpilot.tools.fair.models.dinov2 import DINOV2_AVAILABLE, DINOv2Wrapper
from openpilot.tools.fair.models.sam2 import SAM2_AVAILABLE, SAM2Wrapper
from openpilot.tools.fair.models.cotracker import COTRACKER_AVAILABLE, CoTrackerWrapper
from openpilot.tools.fair.models.detr import DETR_AVAILABLE, DETRWrapper
from openpilot.tools.fair.models.unsam_flow import UNSAMFLOW_AVAILABLE, UnSAMFlowWrapper

__all__ = [
  "ModelWrapper",
  "DINOV2_AVAILABLE",
  "DINOv2Wrapper",
  "SAM2_AVAILABLE",
  "SAM2Wrapper",
  "COTRACKER_AVAILABLE",
  "CoTrackerWrapper",
  "DETR_AVAILABLE",
  "DETRWrapper",
  "UNSAMFLOW_AVAILABLE",
  "UnSAMFlowWrapper",
]
