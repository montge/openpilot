"""Student model architectures for knowledge distillation.

Provides lightweight student models that can be distilled from
FAIR teacher models for efficient real-time inference.
"""

from openpilot.tools.fair.students.vision import (
  TinyViT,
  MobileViT,
  EfficientStudent,
)
from openpilot.tools.fair.students.detection import (
  TinyDETR,
  MobileDetector,
)

__all__ = [
  "TinyViT",
  "MobileViT",
  "EfficientStudent",
  "TinyDETR",
  "MobileDetector",
]
