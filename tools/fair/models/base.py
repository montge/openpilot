"""Base model wrapper interface.

Provides common interface for all FAIR model wrappers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ModelConfig:
  """Base configuration for model wrappers.

  Attributes:
    model_name: Name/variant of the model
    device: Device to run on ('cuda', 'cpu', 'auto')
    precision: Model precision ('fp32', 'fp16', 'int8')
    cache_dir: Directory for caching model weights
  """

  model_name: str = "default"
  device: str = "auto"
  precision: str = "fp32"
  cache_dir: str | None = None


class ModelWrapper(ABC):
  """Abstract base class for FAIR model wrappers.

  Provides common interface for model loading, inference, and feature extraction.
  """

  def __init__(self, config: ModelConfig | None = None):
    """Initialize model wrapper.

    Args:
      config: Model configuration
    """
    self.config = config or ModelConfig()
    self._model = None
    self._loaded = False

  @property
  def loaded(self) -> bool:
    """Check if model is loaded."""
    return self._loaded

  @abstractmethod
  def load(self) -> None:
    """Load model weights and initialize for inference."""

  @abstractmethod
  def unload(self) -> None:
    """Unload model to free memory."""

  @abstractmethod
  def forward(self, inputs: np.ndarray) -> dict[str, Any]:
    """Run forward pass.

    Args:
      inputs: Input tensor (typically images as [B, H, W, C] or [B, C, H, W])

    Returns:
      Dictionary of model outputs
    """

  @abstractmethod
  def extract_features(self, inputs: np.ndarray) -> np.ndarray:
    """Extract feature representations.

    Args:
      inputs: Input tensor

    Returns:
      Feature tensor
    """

  def __enter__(self) -> ModelWrapper:
    """Context manager entry - loads model."""
    self.load()
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    """Context manager exit - unloads model."""
    self.unload()

  def _resolve_device(self) -> str:
    """Resolve 'auto' device to actual device."""
    if self.config.device != "auto":
      return self.config.device

    try:
      import torch

      if torch.cuda.is_available():
        return "cuda"
      elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    except ImportError:
      pass

    return "cpu"
