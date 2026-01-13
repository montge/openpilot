"""Stone Soup algorithm comparison tools.

This module provides adapters for comparing openpilot's tracking algorithms
with implementations from Stone Soup, an open-source sensor fusion library.

Stone Soup provides reference implementations of:
- Kalman Filters (KF, EKF, UKF, CKF)
- Particle Filters
- Multi-target tracking (JPDA, MHT)
- Track-to-track fusion
"""

from openpilot.tools.stonesoup.adapters import (
  STONESOUP_AVAILABLE,
)

__all__ = [
  "STONESOUP_AVAILABLE",
]
