"""Loss functions for openpilot model training.

Based on the approach from mbalesni/openpilot-pipeline, the Laplacian NLL
with winner-takes-all selection significantly outperforms KL-divergence.
"""

from __future__ import annotations

import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
import torch.nn.functional as F  # type: ignore[import-not-found]


class LaplacianNLLLoss(nn.Module):
  """Laplacian Negative Log-Likelihood loss with winner-takes-all.

  For multi-hypothesis predictions, only backpropagates through the
  best-matching hypothesis. This encourages diversity in predictions
  while training the most relevant hypothesis.

  The Laplace distribution has heavier tails than Gaussian, making it
  more robust to outliers in driving trajectories.
  """

  def __init__(self, eps: float = 1e-6):
    super().__init__()
    self.eps = eps

  def forward(
    self,
    pred_mean: torch.Tensor,
    pred_std: torch.Tensor,
    target: torch.Tensor,
  ) -> torch.Tensor:
    """Compute Laplacian NLL loss.

    Args:
      pred_mean: (batch, num_hypotheses, horizon, 2) predicted positions
      pred_std: (batch, num_hypotheses, horizon, 2) predicted uncertainties
      target: (batch, horizon, 2) ground truth positions

    Returns:
      Scalar loss value
    """
    # Ensure positive std
    pred_std = pred_std.clamp(min=self.eps)

    # Expand target for broadcasting: (batch, 1, horizon, 2)
    target = target.unsqueeze(1)

    # Laplacian NLL: |x - mu| / b + log(2b)
    diff = torch.abs(pred_mean - target)
    nll = diff / pred_std + torch.log(2 * pred_std)

    # Sum over horizon and coordinates
    nll_per_hyp = nll.sum(dim=(-1, -2))  # (batch, num_hypotheses)

    # Winner-takes-all: select best hypothesis per sample
    best_idx = nll_per_hyp.argmin(dim=1)  # (batch,)
    best_nll = nll_per_hyp.gather(1, best_idx.unsqueeze(1))  # (batch, 1)

    return best_nll.mean()


class GaussianNLLLoss(nn.Module):
  """Gaussian NLL loss with winner-takes-all."""

  def __init__(self, eps: float = 1e-6):
    super().__init__()
    self.eps = eps

  def forward(
    self,
    pred_mean: torch.Tensor,
    pred_std: torch.Tensor,
    target: torch.Tensor,
  ) -> torch.Tensor:
    pred_std = pred_std.clamp(min=self.eps)
    target = target.unsqueeze(1)

    # Gaussian NLL: 0.5 * ((x - mu) / sigma)^2 + log(sigma) + 0.5 * log(2*pi)
    diff = pred_mean - target
    nll = 0.5 * (diff / pred_std) ** 2 + torch.log(pred_std)
    nll_per_hyp = nll.sum(dim=(-1, -2))

    best_idx = nll_per_hyp.argmin(dim=1)
    best_nll = nll_per_hyp.gather(1, best_idx.unsqueeze(1))

    return best_nll.mean()


class PathDistillationLoss(nn.Module):
  """Combined loss for distilling path predictions.

  Combines:
  - Laplacian NLL for path positions (winner-takes-all)
  - Cross-entropy for hypothesis selection probabilities
  - L1 loss for lane lines and road edges
  """

  def __init__(
    self,
    path_weight: float = 1.0,
    prob_weight: float = 0.1,
    lane_weight: float = 0.5,
    edge_weight: float = 0.5,
  ):
    super().__init__()
    self.path_weight = path_weight
    self.prob_weight = prob_weight
    self.lane_weight = lane_weight
    self.edge_weight = edge_weight

    self.path_loss = LaplacianNLLLoss()

  def forward(
    self,
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
  ) -> dict[str, torch.Tensor]:
    """Compute combined distillation loss.

    Args:
      pred: Dictionary with keys:
        - 'path_mean': (batch, num_hyp, horizon, 2)
        - 'path_std': (batch, num_hyp, horizon, 2)
        - 'path_prob': (batch, num_hyp) hypothesis probabilities
        - 'lane_lines': (batch, num_lanes, horizon, 2) optional
        - 'road_edges': (batch, 2, horizon, 2) optional
      target: Dictionary with same structure from teacher

    Returns:
      Dictionary with individual losses and total
    """
    losses = {}

    # Path prediction loss (main objective)
    if "path_mean" in pred and "path_mean" in target:
      path_loss = self.path_loss(
        pred["path_mean"],
        pred["path_std"],
        target["path_mean"][:, 0],  # Use teacher's best hypothesis as target
      )
      losses["path"] = path_loss * self.path_weight

    # Hypothesis probability matching
    if "path_prob" in pred and "path_prob" in target:
      prob_loss = F.kl_div(
        F.log_softmax(pred["path_prob"], dim=-1),
        F.softmax(target["path_prob"], dim=-1),
        reduction="batchmean",
      )
      losses["prob"] = prob_loss * self.prob_weight

    # Lane line loss
    if "lane_lines" in pred and "lane_lines" in target:
      lane_loss = F.l1_loss(pred["lane_lines"], target["lane_lines"])
      losses["lane"] = lane_loss * self.lane_weight

    # Road edge loss
    if "road_edges" in pred and "road_edges" in target:
      edge_loss = F.l1_loss(pred["road_edges"], target["road_edges"])
      losses["edge"] = edge_loss * self.edge_weight

    # Total loss
    losses["total"] = sum(losses.values())

    return losses


class FeatureDistillationLoss(nn.Module):
  """Feature-level distillation loss.

  Matches intermediate feature representations between student and teacher.
  Useful for transferring knowledge from larger FAIR models.
  """

  def __init__(self, temperature: float = 4.0):
    super().__init__()
    self.temperature = temperature

  def forward(
    self,
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
  ) -> torch.Tensor:
    """Compute feature distillation loss.

    Args:
      student_features: (batch, channels, h, w) or (batch, seq, dim)
      teacher_features: Same shape as student

    Returns:
      Scalar loss
    """
    # Flatten spatial dimensions if present
    if student_features.dim() == 4:
      b, c, h, w = student_features.shape
      student_features = student_features.view(b, c, -1)
      teacher_features = teacher_features.view(b, c, -1)

    # Softmax over feature dimension with temperature
    student_soft = F.softmax(student_features / self.temperature, dim=-1)
    teacher_soft = F.softmax(teacher_features / self.temperature, dim=-1)

    # KL divergence
    loss = F.kl_div(
      student_soft.log(),
      teacher_soft,
      reduction="batchmean",
    ) * (self.temperature**2)

    return loss


class CombinedTrainingLoss(nn.Module):
  """Full training loss combining path distillation and feature matching."""

  def __init__(
    self,
    path_weight: float = 1.0,
    feature_weight: float = 0.1,
    feature_temperature: float = 4.0,
  ):
    super().__init__()
    self.path_loss = PathDistillationLoss(path_weight=path_weight)
    self.feature_loss = FeatureDistillationLoss(temperature=feature_temperature)
    self.feature_weight = feature_weight

  def forward(
    self,
    student_pred: dict[str, torch.Tensor],
    teacher_pred: dict[str, torch.Tensor],
    student_features: torch.Tensor | None = None,
    teacher_features: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    # Path distillation losses
    losses = self.path_loss(student_pred, teacher_pred)

    # Optional feature distillation
    if student_features is not None and teacher_features is not None:
      feat_loss = self.feature_loss(student_features, teacher_features)
      losses["feature"] = feat_loss * self.feature_weight
      losses["total"] = losses["total"] + losses["feature"]

    return losses
