"""DINOv2 model wrapper.

DINOv2 (Self-DIstillation with NO labels v2) is a self-supervised vision transformer
that produces high-quality visual features without labeled data.

Features:
- Strong visual features for downstream tasks
- Depth estimation via linear probing
- Semantic segmentation capabilities
- Multiple model sizes (vits14, vitb14, vitl14, vitg14)

Reference: https://github.com/facebookresearch/dinov2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from openpilot.tools.fair.models.base import ModelConfig, ModelWrapper

# Check DINOv2 availability
try:
  import torch
  import torch.nn.functional as F

  DINOV2_AVAILABLE = True
except ImportError:
  DINOV2_AVAILABLE = False


@dataclass
class DINOv2Config(ModelConfig):
  """DINOv2 configuration.

  Attributes:
    model_name: Model variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
    with_registers: Whether to use register tokens
    image_size: Expected input image size
  """

  model_name: str = "dinov2_vitb14"
  with_registers: bool = False
  image_size: int = 518


class DINOv2Wrapper(ModelWrapper):
  """Wrapper for DINOv2 vision transformer.

  Provides feature extraction and optional depth estimation head.

  Usage:
    config = DINOv2Config(model_name="dinov2_vitb14")
    model = DINOv2Wrapper(config)

    with model:
      features = model.extract_features(images)
      depth = model.estimate_depth(images)
  """

  def __init__(self, config: DINOv2Config | None = None):
    """Initialize DINOv2 wrapper.

    Args:
      config: Model configuration
    """
    super().__init__(config or DINOv2Config())
    self._depth_head = None

  def load(self) -> None:
    """Load DINOv2 model from torch hub."""
    if not DINOV2_AVAILABLE:
      raise ImportError("PyTorch is required for DINOv2. Install with: pip install torch torchvision")

    if self._loaded:
      return

    import torch

    device = self._resolve_device()
    model_name = self.config.model_name

    # Add register suffix if needed
    if self.config.with_registers and not model_name.endswith("_reg"):
      model_name = f"{model_name}_reg"

    # Load from torch hub
    self._model = torch.hub.load("facebookresearch/dinov2", model_name)
    self._model = self._model.to(device)
    self._model.eval()

    self._loaded = True

  def unload(self) -> None:
    """Unload model to free memory."""
    if self._model is not None:
      del self._model
      self._model = None

    if self._depth_head is not None:
      del self._depth_head
      self._depth_head = None

    self._loaded = False

    # Clear CUDA cache if available
    if DINOV2_AVAILABLE:
      import torch

      if torch.cuda.is_available():
        torch.cuda.empty_cache()

  def forward(self, inputs: np.ndarray) -> dict[str, Any]:
    """Run forward pass.

    Args:
      inputs: Images as [B, H, W, C] numpy array (RGB, 0-255)

    Returns:
      Dictionary with 'features' and 'cls_token'
    """
    if not self._loaded:
      raise RuntimeError("Model not loaded. Call load() first or use context manager.")

    import torch

    device = self._resolve_device()

    # Convert to torch tensor and normalize
    x = self._preprocess(inputs)
    x = x.to(device)

    with torch.no_grad():
      output = self._model.forward_features(x)

    return {
      "features": output["x_norm_patchtokens"].cpu().numpy(),
      "cls_token": output["x_norm_clstoken"].cpu().numpy(),
    }

  def extract_features(self, inputs: np.ndarray) -> np.ndarray:
    """Extract patch features.

    Args:
      inputs: Images as [B, H, W, C] numpy array

    Returns:
      Features as [B, N, D] where N is number of patches
    """
    output = self.forward(inputs)
    return output["features"]

  def get_cls_token(self, inputs: np.ndarray) -> np.ndarray:
    """Get CLS token representation.

    Args:
      inputs: Images as [B, H, W, C] numpy array

    Returns:
      CLS tokens as [B, D]
    """
    output = self.forward(inputs)
    return output["cls_token"]

  def estimate_depth(self, inputs: np.ndarray) -> np.ndarray:
    """Estimate depth using linear probe.

    Note: Requires depth head to be loaded with load_depth_head().

    Args:
      inputs: Images as [B, H, W, C] numpy array

    Returns:
      Depth maps as [B, H, W]
    """
    if self._depth_head is None:
      raise RuntimeError("Depth head not loaded. Call load_depth_head() first.")

    import torch

    device = self._resolve_device()

    x = self._preprocess(inputs)
    x = x.to(device)

    with torch.no_grad():
      features = self._model.forward_features(x)["x_norm_patchtokens"]
      depth = self._depth_head(features)

    # Reshape to spatial dimensions
    b, n, _ = features.shape
    h = w = int(n**0.5)
    depth = depth.view(b, h, w)

    # Upsample to original size
    depth = F.interpolate(
      depth.unsqueeze(1),
      size=(inputs.shape[1], inputs.shape[2]),
      mode="bilinear",
      align_corners=False,
    ).squeeze(1)

    return depth.cpu().numpy()

  def load_depth_head(self, head_path: str | None = None) -> None:
    """Load depth estimation linear probe.

    Args:
      head_path: Path to saved depth head weights (uses default if None)
    """
    if not DINOV2_AVAILABLE:
      raise ImportError("PyTorch required")

    import torch
    import torch.nn as nn

    device = self._resolve_device()

    # Get feature dimension from model
    if not self._loaded:
      self.load()

    embed_dim = self._model.embed_dim

    # Simple linear depth head
    self._depth_head = nn.Sequential(
      nn.Linear(embed_dim, embed_dim // 2),
      nn.ReLU(),
      nn.Linear(embed_dim // 2, 1),
    ).to(device)

    if head_path is not None:
      self._depth_head.load_state_dict(torch.load(head_path, map_location=device))

    self._depth_head.eval()

  def _preprocess(self, inputs: np.ndarray) -> torch.Tensor:
    """Preprocess images for DINOv2.

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

  @property
  def embed_dim(self) -> int:
    """Get embedding dimension."""
    if not self._loaded:
      raise RuntimeError("Model not loaded")
    return self._model.embed_dim

  @property
  def patch_size(self) -> int:
    """Get patch size."""
    if not self._loaded:
      raise RuntimeError("Model not loaded")
    return self._model.patch_size
