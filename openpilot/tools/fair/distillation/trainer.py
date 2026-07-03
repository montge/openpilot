"""Distillation trainer.

Provides training loop and utilities for knowledge distillation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable


from openpilot.tools.fair.distillation.losses import (
  DistillationLoss,
  ResponseDistillationLoss,
)
from openpilot.tools.fair.models.base import ModelWrapper

# Check PyTorch availability
try:
  import torch
  import torch.nn as nn
  from torch.utils.data import DataLoader

  TORCH_AVAILABLE = True
except ImportError:
  TORCH_AVAILABLE = False


@dataclass
class DistillationConfig:
  """Configuration for distillation training.

  Attributes:
    epochs: Number of training epochs
    learning_rate: Initial learning rate
    batch_size: Training batch size
    weight_decay: L2 regularization weight
    warmup_epochs: Number of warmup epochs
    save_every: Save checkpoint every N epochs
    log_every: Log metrics every N steps
    checkpoint_dir: Directory for saving checkpoints
    use_amp: Use automatic mixed precision
  """

  epochs: int = 100
  learning_rate: float = 1e-4
  batch_size: int = 32
  weight_decay: float = 0.01
  warmup_epochs: int = 5
  save_every: int = 10
  log_every: int = 100
  checkpoint_dir: str = "checkpoints"
  use_amp: bool = True


@dataclass
class TrainingState:
  """Current training state.

  Attributes:
    epoch: Current epoch
    step: Global step count
    best_loss: Best validation loss seen
    history: Training history (losses, metrics)
  """

  epoch: int = 0
  step: int = 0
  best_loss: float = float("inf")
  history: dict[str, list[float]] = field(default_factory=dict)


class DistillationTrainer:
  """Trainer for knowledge distillation.

  Handles training loop, checkpointing, and logging for
  distilling knowledge from teacher to student models.

  Usage:
    trainer = DistillationTrainer(
      teacher=teacher_model,
      student=student_model,
      config=DistillationConfig(epochs=100),
    )

    trainer.train(train_loader, val_loader)
  """

  def __init__(
    self,
    teacher: ModelWrapper | nn.Module,
    student: nn.Module,
    config: DistillationConfig | None = None,
    loss_fn: DistillationLoss | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
  ):
    """Initialize distillation trainer.

    Args:
      teacher: Teacher model (frozen)
      student: Student model (trainable)
      config: Training configuration
      loss_fn: Distillation loss function
      optimizer: Custom optimizer (default: AdamW)
      scheduler: Custom scheduler (default: CosineAnnealingLR)
    """
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required for distillation training")

    self.config = config or DistillationConfig()
    self.loss_fn = loss_fn or ResponseDistillationLoss()
    self.state = TrainingState()

    # Setup models
    self.teacher = teacher
    self.student = student

    # Freeze teacher
    if isinstance(teacher, nn.Module):
      for param in teacher.parameters():
        param.requires_grad = False
      teacher.eval()

    # Setup device
    self.device = self._resolve_device()
    if isinstance(self.student, nn.Module):
      self.student = self.student.to(self.device)
    if isinstance(self.teacher, nn.Module):
      self.teacher = self.teacher.to(self.device)

    # Setup optimizer
    self.optimizer = optimizer or torch.optim.AdamW(
      self.student.parameters(),
      lr=self.config.learning_rate,
      weight_decay=self.config.weight_decay,
    )

    # Setup scheduler
    self.scheduler = scheduler or torch.optim.lr_scheduler.CosineAnnealingLR(
      self.optimizer,
      T_max=self.config.epochs,
    )

    # Setup AMP scaler
    self.scaler = torch.amp.GradScaler("cuda") if self.config.use_amp else None

    # Callbacks
    self._callbacks: list[Callable] = []

  def train(
    self,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
  ) -> TrainingState:
    """Run distillation training.

    Args:
      train_loader: Training data loader
      val_loader: Validation data loader (optional)

    Returns:
      Final training state
    """
    checkpoint_dir = Path(self.config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(self.config.epochs):
      self.state.epoch = epoch

      # Training epoch
      train_loss = self._train_epoch(train_loader)
      self._add_to_history("train_loss", train_loss)

      # Validation
      if val_loader is not None:
        val_loss = self._validate(val_loader)
        self._add_to_history("val_loss", val_loss)

        # Save best model
        if val_loss < self.state.best_loss:
          self.state.best_loss = val_loss
          self.save_checkpoint(checkpoint_dir / "best.pt")

      # Step scheduler
      self.scheduler.step()

      # Periodic checkpoint
      if (epoch + 1) % self.config.save_every == 0:
        self.save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1}.pt")

      # Callbacks
      for callback in self._callbacks:
        callback(self.state)

    return self.state

  def _train_epoch(self, train_loader: DataLoader) -> float:
    """Run one training epoch.

    Args:
      train_loader: Training data loader

    Returns:
      Average training loss
    """
    self.student.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
      loss = self._train_step(batch)
      total_loss += loss
      num_batches += 1
      self.state.step += 1

      if (batch_idx + 1) % self.config.log_every == 0:
        avg_loss = total_loss / num_batches
        self._log(f"Epoch {self.state.epoch}, Step {self.state.step}: loss={avg_loss:.4f}")

    return total_loss / max(num_batches, 1)

  def _train_step(self, batch: Any) -> float:
    """Run one training step.

    Args:
      batch: Batch of training data

    Returns:
      Loss value
    """
    # Unpack batch
    if isinstance(batch, (list, tuple)):
      inputs = batch[0].to(self.device)
      targets = batch[1].to(self.device) if len(batch) > 1 else None
    else:
      inputs = batch.to(self.device)
      targets = None

    self.optimizer.zero_grad()

    # Forward pass with optional AMP
    if self.config.use_amp and self.scaler is not None:
      with torch.amp.autocast("cuda"):
        loss = self._compute_loss(inputs, targets)

      self.scaler.scale(loss).backward()
      self.scaler.step(self.optimizer)
      self.scaler.update()
    else:
      loss = self._compute_loss(inputs, targets)
      loss.backward()
      self.optimizer.step()

    return loss.item()

  def _compute_loss(
    self,
    inputs: torch.Tensor,
    targets: torch.Tensor | None,
  ) -> torch.Tensor:
    """Compute distillation loss.

    Args:
      inputs: Input batch
      targets: Target labels (optional)

    Returns:
      Loss tensor
    """
    # Get teacher outputs (no grad)
    with torch.no_grad():
      if isinstance(self.teacher, ModelWrapper):
        teacher_out = self.teacher.forward(inputs.cpu().numpy())
        teacher_out = {k: torch.from_numpy(v).to(self.device) for k, v in teacher_out.items()}
      else:
        teacher_out = self.teacher(inputs)

    # Get student outputs
    student_out = self.student(inputs)

    # Compute loss
    return self.loss_fn.compute(student_out, teacher_out, targets)

  def _validate(self, val_loader: DataLoader) -> float:
    """Run validation.

    Args:
      val_loader: Validation data loader

    Returns:
      Average validation loss
    """
    self.student.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
      for batch in val_loader:
        if isinstance(batch, (list, tuple)):
          inputs = batch[0].to(self.device)
          targets = batch[1].to(self.device) if len(batch) > 1 else None
        else:
          inputs = batch.to(self.device)
          targets = None

        loss = self._compute_loss(inputs, targets)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)

  def save_checkpoint(self, path: Path | str) -> None:
    """Save training checkpoint.

    Args:
      path: Path to save checkpoint
    """
    checkpoint = {
      "epoch": self.state.epoch,
      "step": self.state.step,
      "best_loss": self.state.best_loss,
      "history": self.state.history,
      "student_state_dict": self.student.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "scheduler_state_dict": self.scheduler.state_dict(),
      "config": self.config,
    }

    if self.scaler is not None:
      checkpoint["scaler_state_dict"] = self.scaler.state_dict()

    torch.save(checkpoint, path)

  def load_checkpoint(self, path: Path | str) -> None:
    """Load training checkpoint.

    Args:
      path: Path to checkpoint
    """
    checkpoint = torch.load(path, map_location=self.device)

    self.state.epoch = checkpoint["epoch"]
    self.state.step = checkpoint["step"]
    self.state.best_loss = checkpoint["best_loss"]
    self.state.history = checkpoint["history"]

    self.student.load_state_dict(checkpoint["student_state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if self.scaler is not None and "scaler_state_dict" in checkpoint:
      self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

  def add_callback(self, callback: Callable[[TrainingState], None]) -> None:
    """Add training callback.

    Args:
      callback: Function called after each epoch
    """
    self._callbacks.append(callback)

  def _add_to_history(self, key: str, value: float) -> None:
    """Add value to training history.

    Args:
      key: Metric name
      value: Metric value
    """
    if key not in self.state.history:
      self.state.history[key] = []
    self.state.history[key].append(value)

  def _log(self, message: str) -> None:
    """Log message.

    Args:
      message: Message to log
    """
    print(message)

  def _resolve_device(self) -> str:
    """Resolve device for training."""
    if torch.cuda.is_available():
      return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
      return "mps"
    return "cpu"
