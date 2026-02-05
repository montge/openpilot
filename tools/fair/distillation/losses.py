"""Distillation loss functions.

Provides various loss functions for knowledge distillation:
- Response-based: KL divergence between logits
- Feature-based: MSE between intermediate features
- Attention-based: Matching attention patterns
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


# Check PyTorch availability
try:
  import torch
  import torch.nn as nn
  import torch.nn.functional as F

  TORCH_AVAILABLE = True
except ImportError:
  TORCH_AVAILABLE = False


class DistillationLoss(ABC):
  """Abstract base class for distillation losses."""

  @abstractmethod
  def compute(
    self,
    student_output: Any,
    teacher_output: Any,
    targets: Any | None = None,
  ) -> float:
    """Compute distillation loss.

    Args:
      student_output: Output from student model
      teacher_output: Output from teacher model
      targets: Ground truth labels (optional)

    Returns:
      Loss value
    """

  def __call__(
    self,
    student_output: Any,
    teacher_output: Any,
    targets: Any | None = None,
  ) -> float:
    """Compute loss (alias for compute)."""
    return self.compute(student_output, teacher_output, targets)


@dataclass
class ResponseDistillationConfig:
  """Configuration for response distillation.

  Attributes:
    temperature: Softmax temperature for soft targets
    alpha: Weight for distillation loss vs task loss
  """

  temperature: float = 4.0
  alpha: float = 0.7


class ResponseDistillationLoss(DistillationLoss):
  """Response-based knowledge distillation.

  Uses KL divergence between softened teacher and student predictions.
  Based on Hinton et al. "Distilling the Knowledge in a Neural Network".

  Usage:
    loss_fn = ResponseDistillationLoss(temperature=4.0)
    loss = loss_fn(student_logits, teacher_logits, targets)
  """

  def __init__(self, config: ResponseDistillationConfig | None = None):
    """Initialize response distillation loss.

    Args:
      config: Loss configuration
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required for distillation losses")

    self.config = config or ResponseDistillationConfig()
    self._ce_loss = nn.CrossEntropyLoss()
    self._kl_loss = nn.KLDivLoss(reduction="batchmean")

  def compute(
    self,
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    targets: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Compute response distillation loss.

    Args:
      student_output: Student logits [B, C]
      teacher_output: Teacher logits [B, C]
      targets: Ground truth labels [B] (optional)

    Returns:
      Combined distillation loss
    """
    T = self.config.temperature
    alpha = self.config.alpha

    # Soft targets from teacher
    soft_targets = F.softmax(teacher_output / T, dim=-1)
    soft_student = F.log_softmax(student_output / T, dim=-1)

    # KL divergence loss (scaled by T^2 as per original paper)
    distill_loss = self._kl_loss(soft_student, soft_targets) * (T * T)

    if targets is not None:
      # Combined loss with hard targets
      hard_loss = self._ce_loss(student_output, targets)
      return alpha * distill_loss + (1 - alpha) * hard_loss

    return distill_loss


@dataclass
class FeatureDistillationConfig:
  """Configuration for feature distillation.

  Attributes:
    normalize: Whether to normalize features before comparison
    loss_type: Type of loss ('mse', 'cosine', 'l1')
  """

  normalize: bool = True
  loss_type: str = "mse"


class FeatureDistillationLoss(DistillationLoss):
  """Feature-based knowledge distillation.

  Matches intermediate feature representations between teacher and student.

  Usage:
    loss_fn = FeatureDistillationLoss()
    loss = loss_fn(student_features, teacher_features)
  """

  def __init__(self, config: FeatureDistillationConfig | None = None):
    """Initialize feature distillation loss.

    Args:
      config: Loss configuration
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required for distillation losses")

    self.config = config or FeatureDistillationConfig()

  def compute(
    self,
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    targets: Any | None = None,
  ) -> torch.Tensor:
    """Compute feature distillation loss.

    Args:
      student_output: Student features [B, D] or [B, H, W, D]
      teacher_output: Teacher features [B, D] or [B, H, W, D]
      targets: Unused

    Returns:
      Feature matching loss
    """
    student_feat = student_output
    teacher_feat = teacher_output

    # Normalize if configured
    if self.config.normalize:
      student_feat = F.normalize(student_feat, dim=-1)
      teacher_feat = F.normalize(teacher_feat, dim=-1)

    # Compute loss based on type
    if self.config.loss_type == "mse":
      return F.mse_loss(student_feat, teacher_feat)
    elif self.config.loss_type == "cosine":
      return 1 - F.cosine_similarity(student_feat, teacher_feat, dim=-1).mean()
    elif self.config.loss_type == "l1":
      return F.l1_loss(student_feat, teacher_feat)
    else:
      raise ValueError(f"Unknown loss type: {self.config.loss_type}")


@dataclass
class AttentionDistillationConfig:
  """Configuration for attention distillation.

  Attributes:
    normalize: Whether to normalize attention maps
    loss_type: Type of loss ('mse', 'kl')
  """

  normalize: bool = True
  loss_type: str = "mse"


class AttentionDistillationLoss(DistillationLoss):
  """Attention-based knowledge distillation.

  Matches attention patterns between teacher and student models.
  Useful for transformer-based models like DINOv2.

  Usage:
    loss_fn = AttentionDistillationLoss()
    loss = loss_fn(student_attention, teacher_attention)
  """

  def __init__(self, config: AttentionDistillationConfig | None = None):
    """Initialize attention distillation loss.

    Args:
      config: Loss configuration
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required for distillation losses")

    self.config = config or AttentionDistillationConfig()

  def compute(
    self,
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    targets: Any | None = None,
  ) -> torch.Tensor:
    """Compute attention distillation loss.

    Args:
      student_output: Student attention maps [B, H, N, N]
      teacher_output: Teacher attention maps [B, H, N, N]
      targets: Unused

    Returns:
      Attention matching loss
    """
    student_attn = student_output
    teacher_attn = teacher_output

    # Normalize attention maps
    if self.config.normalize:
      student_attn = F.softmax(student_attn, dim=-1)
      teacher_attn = F.softmax(teacher_attn, dim=-1)

    # Compute loss
    if self.config.loss_type == "mse":
      return F.mse_loss(student_attn, teacher_attn)
    elif self.config.loss_type == "kl":
      # Use log of student for KL divergence
      student_log = torch.log(student_attn + 1e-8)
      return F.kl_div(student_log, teacher_attn, reduction="batchmean")
    else:
      raise ValueError(f"Unknown loss type: {self.config.loss_type}")


@dataclass
class TaskDistillationConfig:
  """Configuration for combined task + distillation loss.

  Attributes:
    task_weight: Weight for the task-specific loss
    distill_weight: Weight for the distillation loss
    temperature: Softmax temperature for soft targets
    task_loss_type: Type of task loss ('ce', 'mse', 'l1')
  """

  task_weight: float = 0.3
  distill_weight: float = 0.7
  temperature: float = 4.0
  task_loss_type: str = "ce"


class TaskDistillationLoss(DistillationLoss):
  """Combined task + distillation loss.

  Jointly optimizes for task performance (using ground truth) and
  knowledge transfer from teacher. Useful for DoRA + FAIR distillation
  where both task labels and teacher outputs are available.

  Usage:
    loss_fn = TaskDistillationLoss(config)
    loss = loss_fn(student_logits, teacher_logits, targets)
  """

  def __init__(self, config: TaskDistillationConfig | None = None):
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required for distillation losses")

    self.config = config or TaskDistillationConfig()

    if self.config.task_loss_type == "ce":
      self._task_loss = nn.CrossEntropyLoss()
    elif self.config.task_loss_type == "mse":
      self._task_loss = nn.MSELoss()
    elif self.config.task_loss_type == "l1":
      self._task_loss = nn.L1Loss()
    else:
      raise ValueError(f"Unknown task loss type: {self.config.task_loss_type}")

    self._kl_loss = nn.KLDivLoss(reduction="batchmean")

  def compute(
    self,
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    targets: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Compute combined task + distillation loss.

    Args:
      student_output: Student predictions
      teacher_output: Teacher predictions
      targets: Ground truth labels (required)

    Returns:
      Weighted sum of task and distillation losses
    """
    T = self.config.temperature

    # Distillation loss (soft targets)
    soft_student = F.log_softmax(student_output / T, dim=-1)
    soft_teacher = F.softmax(teacher_output / T, dim=-1)
    distill_loss = self._kl_loss(soft_student, soft_teacher) * (T * T)

    # Task loss (hard targets)
    if targets is not None:
      task_loss = self._task_loss(student_output, targets)
      return self.config.distill_weight * distill_loss + self.config.task_weight * task_loss

    return distill_loss


class CombinedDistillationLoss(DistillationLoss):
  """Combined distillation loss with multiple components.

  Allows combining response, feature, and attention losses with weights.

  Usage:
    loss_fn = CombinedDistillationLoss([
      (ResponseDistillationLoss(), 1.0),
      (FeatureDistillationLoss(), 0.5),
    ])
    loss = loss_fn(student_outputs, teacher_outputs)
  """

  def __init__(self, loss_components: list[tuple[DistillationLoss, float]]):
    """Initialize combined loss.

    Args:
      loss_components: List of (loss_fn, weight) tuples
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required for distillation losses")

    self.loss_components = loss_components

  def compute(
    self,
    student_output: dict[str, torch.Tensor],
    teacher_output: dict[str, torch.Tensor],
    targets: Any | None = None,
  ) -> torch.Tensor:
    """Compute combined distillation loss.

    Args:
      student_output: Dict of student outputs by name
      teacher_output: Dict of teacher outputs by name
      targets: Ground truth (optional)

    Returns:
      Weighted sum of all loss components
    """
    total_loss = 0.0

    for (loss_fn, weight), (key, student_val) in zip(
      self.loss_components,
      student_output.items(),
      strict=False,
    ):
      teacher_val = teacher_output.get(key)
      if teacher_val is not None:
        component_loss = loss_fn.compute(student_val, teacher_val, targets)
        total_loss = total_loss + weight * component_loss

    return total_loss
