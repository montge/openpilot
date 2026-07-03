"""Multi-task learning for perception models.

Trains models jointly on depth estimation, segmentation, and detection
using FAIR teacher models for supervision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Check PyTorch availability
try:
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  from torch.utils.data import DataLoader

  TORCH_AVAILABLE = True
  _BaseModule = nn.Module
except ImportError:
  TORCH_AVAILABLE = False
  _BaseModule = object


@dataclass
class MultiTaskConfig:
  """Multi-task training configuration.

  Attributes:
    tasks: List of task names ('depth', 'segmentation', 'detection')
    task_weights: Weight for each task loss
    epochs: Number of training epochs
    learning_rate: Initial learning rate
    batch_size: Training batch size
    uncertainty_weighting: Use learned uncertainty weights
  """

  tasks: list[str] = field(default_factory=lambda: ["depth", "segmentation"])
  task_weights: dict[str, float] = field(default_factory=lambda: {"depth": 1.0, "segmentation": 1.0})
  epochs: int = 100
  learning_rate: float = 1e-4
  batch_size: int = 16
  uncertainty_weighting: bool = True


class MultiTaskHead(_BaseModule):
  """Multi-task prediction head.

  Takes backbone features and produces task-specific outputs.
  """

  def __init__(
    self,
    in_features: int,
    tasks: list[str],
    num_classes: int = 19,  # Default for Cityscapes
  ):
    """Initialize multi-task head.

    Args:
      in_features: Input feature dimension
      tasks: List of task names
      num_classes: Number of segmentation classes
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required for multi-task training")

    super().__init__()

    self.tasks = tasks
    self.heads = nn.ModuleDict()

    for task in tasks:
      if task == "depth":
        self.heads[task] = self._make_depth_head(in_features)
      elif task == "segmentation":
        self.heads[task] = self._make_segmentation_head(in_features, num_classes)
      elif task == "detection":
        self.heads[task] = self._make_detection_head(in_features)
      else:
        raise ValueError(f"Unknown task: {task}")

  def _make_depth_head(self, in_features: int) -> nn.Module:
    """Create depth estimation head."""
    return nn.Sequential(
      nn.Conv2d(in_features, 256, 3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 128, 3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 1, 1),
      nn.Sigmoid(),  # Output normalized depth [0, 1]
    )

  def _make_segmentation_head(self, in_features: int, num_classes: int) -> nn.Module:
    """Create semantic segmentation head."""
    return nn.Sequential(
      nn.Conv2d(in_features, 256, 3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 128, 3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, num_classes, 1),
    )

  def _make_detection_head(self, in_features: int) -> nn.Module:
    """Create object detection head."""
    return nn.Sequential(
      nn.Conv2d(in_features, 256, 3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 128, 3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      # Output: 4 box coords + 1 objectness per anchor
      nn.Conv2d(128, 5 * 3, 1),  # 3 anchors per location
    )

  def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
    """Forward pass for all tasks.

    Args:
      features: Backbone features [B, C, H, W]

    Returns:
      Dictionary of task outputs
    """
    outputs = {}
    for task, head in self.heads.items():
      outputs[task] = head(features)
    return outputs


class UncertaintyWeights(_BaseModule):
  """Learnable uncertainty weights for multi-task learning.

  Based on "Multi-Task Learning Using Uncertainty to Weigh Losses"
  (Kendall et al., 2018).
  """

  def __init__(self, tasks: list[str]):
    """Initialize uncertainty weights.

    Args:
      tasks: List of task names
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required")

    super().__init__()

    # Log variance for each task (initialized to 0, meaning variance=1)
    self.log_vars = nn.ParameterDict({task: nn.Parameter(torch.zeros(1)) for task in tasks})

  def forward(
    self,
    losses: dict[str, torch.Tensor],
  ) -> tuple[torch.Tensor, dict[str, float]]:
    """Weight losses by learned uncertainty.

    Args:
      losses: Dictionary of task losses

    Returns:
      Tuple of (total weighted loss, weight dictionary)
    """
    total_loss = 0.0
    weights = {}

    for task, loss in losses.items():
      log_var = self.log_vars[task]
      # Loss = (1/2σ²) * L + log(σ) = (1/2)exp(-log_var) * L + (1/2)log_var
      precision = torch.exp(-log_var)
      weighted = 0.5 * precision * loss + 0.5 * log_var
      total_loss = total_loss + weighted
      weights[task] = precision.item()

    return total_loss, weights


class MultiTaskTrainer:
  """Trainer for multi-task perception models.

  Handles joint training on multiple perception tasks with
  optional uncertainty weighting and teacher supervision.
  """

  def __init__(
    self,
    backbone: nn.Module,
    head: MultiTaskHead,
    config: MultiTaskConfig,
    teachers: dict[str, nn.Module] | None = None,
  ):
    """Initialize multi-task trainer.

    Args:
      backbone: Shared feature backbone
      head: Multi-task prediction head
      config: Training configuration
      teachers: Optional teacher models per task for distillation
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required for multi-task training")

    self.backbone = backbone
    self.head = head
    self.config = config
    self.teachers = teachers or {}

    # Move to device
    self.device = self._resolve_device()
    self.backbone = self.backbone.to(self.device)
    self.head = self.head.to(self.device)
    for teacher in self.teachers.values():
      teacher.to(self.device)
      teacher.eval()

    # Uncertainty weights
    if config.uncertainty_weighting:
      self.uncertainty = UncertaintyWeights(config.tasks).to(self.device)
    else:
      self.uncertainty = None

    # Optimizer
    params = list(self.backbone.parameters()) + list(self.head.parameters())
    if self.uncertainty is not None:
      params.extend(self.uncertainty.parameters())
    self.optimizer = torch.optim.AdamW(params, lr=config.learning_rate)

    # Scheduler
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)

    # Loss functions
    self.loss_fns = {
      "depth": nn.L1Loss(),
      "segmentation": nn.CrossEntropyLoss(ignore_index=255),
      "detection": nn.SmoothL1Loss(),
    }

    # Training history
    self.history: dict[str, list[float]] = {}

  def train(
    self,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
  ) -> dict[str, list[float]]:
    """Run multi-task training.

    Args:
      train_loader: Training data loader
      val_loader: Validation data loader (optional)

    Returns:
      Training history
    """
    for epoch in range(self.config.epochs):
      # Training epoch
      train_losses = self._train_epoch(train_loader)
      self._log_losses("train", train_losses, epoch)

      # Validation
      if val_loader is not None:
        val_losses = self._validate(val_loader)
        self._log_losses("val", val_losses, epoch)

      # Step scheduler
      self.scheduler.step()

    return self.history

  def _train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
    """Run one training epoch."""
    self.backbone.train()
    self.head.train()

    epoch_losses: dict[str, list[float]] = {task: [] for task in self.config.tasks}
    epoch_losses["total"] = []

    for batch in train_loader:
      loss, task_losses = self._train_step(batch)

      epoch_losses["total"].append(loss)
      for task, task_loss in task_losses.items():
        epoch_losses[task].append(task_loss)

    return {k: np.mean(v) for k, v in epoch_losses.items()}

  def _train_step(self, batch: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Run one training step."""
    self.optimizer.zero_grad()

    # Get inputs
    images = batch["image"].to(self.device)

    # Forward pass
    features = self.backbone(images)
    predictions = self.head(features)

    # Compute task losses
    task_losses = {}
    for task in self.config.tasks:
      if task in self.teachers:
        # Use teacher for supervision
        with torch.no_grad():
          teacher_out = self.teachers[task](images)
        loss = self._compute_distillation_loss(predictions[task], teacher_out, task)
      elif f"{task}_target" in batch:
        # Use ground truth
        target = batch[f"{task}_target"].to(self.device)
        loss = self.loss_fns[task](predictions[task], target)
      else:
        continue

      task_losses[task] = loss

    # Combine losses
    if self.uncertainty is not None:
      total_loss, weights = self.uncertainty(task_losses)
    else:
      total_loss = sum(self.config.task_weights.get(task, 1.0) * loss for task, loss in task_losses.items())

    # Backward pass
    total_loss.backward()
    self.optimizer.step()

    return (
      total_loss.item(),
      {task: loss.item() for task, loss in task_losses.items()},
    )

  def _validate(self, val_loader: DataLoader) -> dict[str, float]:
    """Run validation."""
    self.backbone.eval()
    self.head.eval()

    epoch_losses: dict[str, list[float]] = {task: [] for task in self.config.tasks}
    epoch_losses["total"] = []

    with torch.no_grad():
      for batch in val_loader:
        images = batch["image"].to(self.device)
        features = self.backbone(images)
        predictions = self.head(features)

        task_losses = {}
        for task in self.config.tasks:
          if f"{task}_target" in batch:
            target = batch[f"{task}_target"].to(self.device)
            loss = self.loss_fns[task](predictions[task], target)
            task_losses[task] = loss.item()
            epoch_losses[task].append(loss.item())

        if task_losses:
          total = sum(self.config.task_weights.get(task, 1.0) * loss for task, loss in task_losses.items())
          epoch_losses["total"].append(total)

    return {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}

  def _compute_distillation_loss(
    self,
    student_out: torch.Tensor,
    teacher_out: torch.Tensor,
    task: str,
  ) -> torch.Tensor:
    """Compute distillation loss for a task."""
    if task == "depth":
      return F.l1_loss(student_out, teacher_out)
    elif task == "segmentation":
      # Soft cross-entropy with temperature
      T = 4.0
      student_soft = F.log_softmax(student_out / T, dim=1)
      teacher_soft = F.softmax(teacher_out / T, dim=1)
      return F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)
    else:
      return F.mse_loss(student_out, teacher_out)

  def _log_losses(
    self,
    phase: str,
    losses: dict[str, float],
    epoch: int,
  ) -> None:
    """Log losses to history."""
    for key, value in losses.items():
      history_key = f"{phase}_{key}"
      if history_key not in self.history:
        self.history[history_key] = []
      self.history[history_key].append(value)

  def save(self, path: Path | str) -> None:
    """Save model checkpoint."""
    checkpoint = {
      "backbone": self.backbone.state_dict(),
      "head": self.head.state_dict(),
      "optimizer": self.optimizer.state_dict(),
      "scheduler": self.scheduler.state_dict(),
      "history": self.history,
      "config": self.config,
    }
    if self.uncertainty is not None:
      checkpoint["uncertainty"] = self.uncertainty.state_dict()
    torch.save(checkpoint, path)

  def load(self, path: Path | str) -> None:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=self.device)
    self.backbone.load_state_dict(checkpoint["backbone"])
    self.head.load_state_dict(checkpoint["head"])
    self.optimizer.load_state_dict(checkpoint["optimizer"])
    self.scheduler.load_state_dict(checkpoint["scheduler"])
    self.history = checkpoint["history"]
    if self.uncertainty is not None and "uncertainty" in checkpoint:
      self.uncertainty.load_state_dict(checkpoint["uncertainty"])

  def _resolve_device(self) -> str:
    """Resolve device for training."""
    if torch.cuda.is_available():
      return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
      return "mps"
    return "cpu"
