"""Tests for distillation framework."""

import pytest

# Check PyTorch availability
try:
  import torch
  import torch.nn as nn

  TORCH_AVAILABLE = True
except ImportError:
  TORCH_AVAILABLE = False

from openpilot.tools.fair.distillation.losses import (
  ResponseDistillationConfig,
  FeatureDistillationConfig,
  AttentionDistillationConfig,
)
from openpilot.tools.fair.distillation.trainer import (
  DistillationConfig,
  TrainingState,
)


class TestDistillationConfigs:
  """Tests for distillation configurations."""

  def test_response_distillation_config(self):
    """Test response distillation config defaults."""
    config = ResponseDistillationConfig()
    assert config.temperature == 4.0
    assert config.alpha == 0.7

  def test_response_distillation_config_custom(self):
    """Test custom response distillation config."""
    config = ResponseDistillationConfig(temperature=2.0, alpha=0.5)
    assert config.temperature == 2.0
    assert config.alpha == 0.5

  def test_feature_distillation_config(self):
    """Test feature distillation config defaults."""
    config = FeatureDistillationConfig()
    assert config.normalize is True
    assert config.loss_type == "mse"

  def test_feature_distillation_config_options(self):
    """Test feature distillation loss types."""
    for loss_type in ["mse", "cosine", "l1"]:
      config = FeatureDistillationConfig(loss_type=loss_type)
      assert config.loss_type == loss_type

  def test_attention_distillation_config(self):
    """Test attention distillation config defaults."""
    config = AttentionDistillationConfig()
    assert config.normalize is True
    assert config.loss_type == "mse"


class TestTrainingConfig:
  """Tests for training configuration."""

  def test_default_config(self):
    """Test default training config."""
    config = DistillationConfig()
    assert config.epochs == 100
    assert config.learning_rate == 1e-4
    assert config.batch_size == 32
    assert config.weight_decay == 0.01
    assert config.warmup_epochs == 5
    assert config.save_every == 10
    assert config.log_every == 100
    assert config.use_amp is True

  def test_custom_config(self):
    """Test custom training config."""
    config = DistillationConfig(
      epochs=50,
      learning_rate=1e-3,
      batch_size=64,
      use_amp=False,
    )
    assert config.epochs == 50
    assert config.learning_rate == 1e-3
    assert config.batch_size == 64
    assert config.use_amp is False


class TestTrainingState:
  """Tests for training state."""

  def test_default_state(self):
    """Test default training state."""
    state = TrainingState()
    assert state.epoch == 0
    assert state.step == 0
    assert state.best_loss == float("inf")
    assert state.history == {}

  def test_state_with_history(self):
    """Test training state with history."""
    state = TrainingState(
      epoch=10,
      step=1000,
      best_loss=0.5,
      history={"train_loss": [1.0, 0.8, 0.6], "val_loss": [1.1, 0.9, 0.7]},
    )
    assert state.epoch == 10
    assert state.step == 1000
    assert state.best_loss == 0.5
    assert len(state.history["train_loss"]) == 3


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestLossFunctions:
  """Tests for loss functions (requires PyTorch)."""

  def test_response_distillation_loss(self):
    """Test response distillation loss computation."""
    from openpilot.tools.fair.distillation.losses import ResponseDistillationLoss

    loss_fn = ResponseDistillationLoss()

    student_logits = torch.randn(8, 10)
    teacher_logits = torch.randn(8, 10)

    loss = loss_fn.compute(student_logits, teacher_logits)
    assert loss.item() >= 0

  def test_response_distillation_with_targets(self):
    """Test response distillation with hard targets."""
    from openpilot.tools.fair.distillation.losses import ResponseDistillationLoss

    loss_fn = ResponseDistillationLoss()

    student_logits = torch.randn(8, 10)
    teacher_logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))

    loss = loss_fn.compute(student_logits, teacher_logits, targets)
    assert loss.item() >= 0

  def test_feature_distillation_loss_mse(self):
    """Test feature distillation with MSE loss."""
    from openpilot.tools.fair.distillation.losses import FeatureDistillationLoss

    loss_fn = FeatureDistillationLoss()

    student_features = torch.randn(8, 256)
    teacher_features = torch.randn(8, 256)

    loss = loss_fn.compute(student_features, teacher_features)
    assert loss.item() >= 0

  def test_feature_distillation_loss_cosine(self):
    """Test feature distillation with cosine loss."""
    from openpilot.tools.fair.distillation.losses import (
      FeatureDistillationLoss,
      FeatureDistillationConfig,
    )

    config = FeatureDistillationConfig(loss_type="cosine")
    loss_fn = FeatureDistillationLoss(config)

    student_features = torch.randn(8, 256)
    teacher_features = torch.randn(8, 256)

    loss = loss_fn.compute(student_features, teacher_features)
    assert loss.item() >= 0
    assert loss.item() <= 2  # Cosine loss bounded

  def test_feature_distillation_loss_l1(self):
    """Test feature distillation with L1 loss."""
    from openpilot.tools.fair.distillation.losses import (
      FeatureDistillationLoss,
      FeatureDistillationConfig,
    )

    config = FeatureDistillationConfig(loss_type="l1")
    loss_fn = FeatureDistillationLoss(config)

    student_features = torch.randn(8, 256)
    teacher_features = torch.randn(8, 256)

    loss = loss_fn.compute(student_features, teacher_features)
    assert loss.item() >= 0

  def test_attention_distillation_loss(self):
    """Test attention distillation loss."""
    from openpilot.tools.fair.distillation.losses import AttentionDistillationLoss

    loss_fn = AttentionDistillationLoss()

    # [B, H, N, N] attention maps
    student_attn = torch.randn(2, 4, 16, 16)
    teacher_attn = torch.randn(2, 4, 16, 16)

    loss = loss_fn.compute(student_attn, teacher_attn)
    assert loss.item() >= 0

  def test_combined_distillation_loss(self):
    """Test combined distillation loss."""
    from openpilot.tools.fair.distillation.losses import (
      CombinedDistillationLoss,
      ResponseDistillationLoss,
      FeatureDistillationLoss,
    )

    loss_fn = CombinedDistillationLoss(
      [
        (ResponseDistillationLoss(), 1.0),
        (FeatureDistillationLoss(), 0.5),
      ]
    )

    student_outputs = {
      "logits": torch.randn(8, 10),
      "features": torch.randn(8, 256),
    }
    teacher_outputs = {
      "logits": torch.randn(8, 10),
      "features": torch.randn(8, 256),
    }

    loss = loss_fn.compute(student_outputs, teacher_outputs)
    assert loss >= 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDistillationTrainer:
  """Tests for distillation trainer (requires PyTorch)."""

  def test_trainer_initialization(self):
    """Test trainer initialization."""
    from openpilot.tools.fair.distillation.trainer import DistillationTrainer

    # Simple teacher and student
    teacher = nn.Linear(10, 5)
    student = nn.Linear(10, 5)

    trainer = DistillationTrainer(teacher, student)

    assert trainer.teacher is teacher
    assert trainer.student is student
    assert trainer.state.epoch == 0

  def test_trainer_with_config(self):
    """Test trainer with custom config."""
    from openpilot.tools.fair.distillation.trainer import DistillationTrainer

    teacher = nn.Linear(10, 5)
    student = nn.Linear(10, 5)

    config = DistillationConfig(epochs=10, learning_rate=1e-3)
    trainer = DistillationTrainer(teacher, student, config=config)

    assert trainer.config.epochs == 10
    assert trainer.config.learning_rate == 1e-3

  def test_trainer_adds_callback(self):
    """Test adding callbacks to trainer."""
    from openpilot.tools.fair.distillation.trainer import DistillationTrainer

    teacher = nn.Linear(10, 5)
    student = nn.Linear(10, 5)

    trainer = DistillationTrainer(teacher, student)

    callback_called = []

    def my_callback(state):
      callback_called.append(state.epoch)

    trainer.add_callback(my_callback)
    assert len(trainer._callbacks) == 1

  def test_trainer_history_tracking(self):
    """Test training history tracking."""
    from openpilot.tools.fair.distillation.trainer import DistillationTrainer

    teacher = nn.Linear(10, 5)
    student = nn.Linear(10, 5)

    trainer = DistillationTrainer(teacher, student)

    trainer._add_to_history("train_loss", 1.0)
    trainer._add_to_history("train_loss", 0.8)
    trainer._add_to_history("val_loss", 1.1)

    assert "train_loss" in trainer.state.history
    assert len(trainer.state.history["train_loss"]) == 2
    assert trainer.state.history["train_loss"][1] == 0.8
