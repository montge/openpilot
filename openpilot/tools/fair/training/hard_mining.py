"""Hard example mining for training data selection.

Identifies and prioritizes difficult training examples to improve
model performance on challenging scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

import numpy as np

try:
  import torch
  from torch.utils.data import Dataset, Sampler

  TORCH_AVAILABLE = True
except ImportError:
  TORCH_AVAILABLE = False


@dataclass
class HardMiningConfig:
  """Hard example mining configuration.

  Attributes:
    strategy: Mining strategy ('loss', 'gradient', 'uncertainty')
    top_k_percent: Percentage of hardest examples to keep
    min_loss_threshold: Minimum loss to consider a hard example
    update_interval: How often to re-mine (in epochs)
    memory_size: Maximum number of difficulty scores to store
  """

  strategy: str = "loss"
  top_k_percent: float = 30.0
  min_loss_threshold: float = 0.0
  update_interval: int = 1
  memory_size: int = 100000


class DifficultyTracker:
  """Tracks per-sample difficulty scores across training.

  Maintains a running estimate of how difficult each sample is
  based on loss values.
  """

  def __init__(self, config: HardMiningConfig | None = None):
    self.config = config or HardMiningConfig()
    self._scores: dict[int, float] = {}
    self._counts: dict[int, int] = {}

  def update(self, indices: list[int], losses: list[float]) -> None:
    """Update difficulty scores with new loss values.

    Uses exponential moving average to smooth scores.

    Args:
      indices: Sample indices
      losses: Per-sample loss values
    """
    alpha = 0.3  # EMA smoothing factor
    for idx, loss in zip(indices, losses, strict=True):
      if idx in self._scores:
        self._scores[idx] = alpha * loss + (1 - alpha) * self._scores[idx]
        self._counts[idx] += 1
      else:
        self._scores[idx] = loss
        self._counts[idx] = 1

      # Evict oldest if over memory limit
      if len(self._scores) > self.config.memory_size:
        min_count_idx = min(self._counts, key=self._counts.get)  # type: ignore[arg-type]
        del self._scores[min_count_idx]
        del self._counts[min_count_idx]

  def get_hard_indices(self, top_k: int | None = None) -> list[int]:
    """Get indices of hardest examples.

    Args:
      top_k: Number of hard examples (default: top_k_percent of total)

    Returns:
      Sorted list of hard example indices (hardest first)
    """
    if not self._scores:
      return []

    if top_k is None:
      top_k = max(1, int(len(self._scores) * self.config.top_k_percent / 100))

    # Filter by minimum threshold
    filtered = {idx: score for idx, score in self._scores.items() if score >= self.config.min_loss_threshold}

    # Sort by difficulty (highest loss first)
    sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

    return [idx for idx, _ in sorted_items[:top_k]]

  def get_statistics(self) -> dict[str, float]:
    """Get difficulty distribution statistics."""
    if not self._scores:
      return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    values = list(self._scores.values())
    return {
      "count": len(values),
      "mean": float(np.mean(values)),
      "std": float(np.std(values)),
      "min": float(np.min(values)),
      "max": float(np.max(values)),
    }


class HardExampleSampler(Sampler):
  """Sampler that prioritizes hard examples.

  Oversamples difficult examples and undersamples easy ones.

  Usage:
    tracker = DifficultyTracker()
    sampler = HardExampleSampler(dataset, tracker)
    dataloader = DataLoader(dataset, sampler=sampler)
  """

  def __init__(
    self,
    dataset: Dataset,
    tracker: DifficultyTracker,
    config: HardMiningConfig | None = None,
  ):
    if not TORCH_AVAILABLE:
      raise ImportError("PyTorch required")

    self.dataset = dataset
    self.tracker = tracker
    self.config = config or HardMiningConfig()
    self._epoch = 0

  def __iter__(self) -> Iterator[int]:
    """Generate sample indices prioritizing hard examples."""
    n = len(self.dataset)  # type: ignore[arg-type]
    hard_indices = self.tracker.get_hard_indices()

    if not hard_indices:
      # No difficulty info yet - use random sampling
      yield from torch.randperm(n).tolist()
      return

    # Split into hard and easy
    hard_set = set(hard_indices)
    easy_indices = [i for i in range(n) if i not in hard_set]

    # Oversample hard examples (2x), undersample easy
    hard_count = min(len(hard_indices), n // 2)
    easy_count = n - hard_count

    sampled_hard = np.random.choice(hard_indices, hard_count, replace=len(hard_indices) < hard_count)
    sampled_easy = np.random.choice(easy_indices, easy_count, replace=len(easy_indices) < easy_count) if easy_indices else np.array([], dtype=int)

    combined = np.concatenate([sampled_hard, sampled_easy])
    np.random.shuffle(combined)

    yield from combined.tolist()

  def __len__(self) -> int:
    return len(self.dataset)  # type: ignore[arg-type]

  def set_epoch(self, epoch: int) -> None:
    """Set current epoch for deterministic shuffling."""
    self._epoch = epoch
