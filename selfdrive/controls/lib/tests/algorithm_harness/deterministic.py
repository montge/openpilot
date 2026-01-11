"""
Deterministic execution utilities for algorithm testing.

This module provides utilities to ensure reproducible test execution:
- DeterministicContext: Context manager for deterministic mode
- FakeTimeSource: Monotonic fake time for timing-sensitive code
- Seed management for all random sources

Usage:
  with DeterministicContext(seed=42) as ctx:
    # All random operations are deterministic within this block
    result = run_algorithm(...)

  # Or use the fake time source directly:
  time_source = FakeTimeSource(start_time_ns=0, dt_ns=10_000_000)
  for _ in range(100):
    timestamp = time_source.monotonic_ns()  # Advances by dt_ns each call
"""

import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable
from unittest.mock import patch
import numpy as np


@dataclass
class FakeTimeSource:
  """
  A deterministic time source for testing.

  Provides predictable timestamps that advance by a fixed amount
  on each call, enabling reproducible timing behavior.

  Attributes:
    start_time_ns: Initial timestamp in nanoseconds
    dt_ns: Time step in nanoseconds per call
    current_time_ns: Current timestamp (read-only)
  """

  start_time_ns: int = 0
  dt_ns: int = 10_000_000  # 10ms = 100Hz default
  _current_time_ns: int = field(init=False, repr=False)
  _call_count: int = field(init=False, repr=False)

  def __post_init__(self):
    self._current_time_ns = self.start_time_ns
    self._call_count = 0

  @property
  def current_time_ns(self) -> int:
    """Current timestamp in nanoseconds."""
    return self._current_time_ns

  @property
  def current_time_s(self) -> float:
    """Current timestamp in seconds."""
    return self._current_time_ns / 1e9

  @property
  def call_count(self) -> int:
    """Number of times time was requested."""
    return self._call_count

  def monotonic_ns(self) -> int:
    """
    Get current time and advance by dt_ns.

    Returns:
      Current timestamp in nanoseconds
    """
    current = self._current_time_ns
    self._current_time_ns += self.dt_ns
    self._call_count += 1
    return current

  def monotonic(self) -> float:
    """
    Get current time in seconds and advance by dt.

    Returns:
      Current timestamp in seconds
    """
    return self.monotonic_ns() / 1e9

  def time(self) -> float:
    """
    Alias for monotonic() to replace time.time().

    Returns:
      Current timestamp in seconds
    """
    return self.monotonic()

  def time_ns(self) -> int:
    """
    Alias for monotonic_ns() to replace time.time_ns().

    Returns:
      Current timestamp in nanoseconds
    """
    return self.monotonic_ns()

  def peek_ns(self) -> int:
    """
    Get current time without advancing.

    Returns:
      Current timestamp in nanoseconds
    """
    return self._current_time_ns

  def peek(self) -> float:
    """
    Get current time in seconds without advancing.

    Returns:
      Current timestamp in seconds
    """
    return self._current_time_ns / 1e9

  def advance(self, ns: int) -> None:
    """
    Advance time by specified nanoseconds.

    Args:
      ns: Nanoseconds to advance
    """
    self._current_time_ns += ns

  def advance_seconds(self, seconds: float) -> None:
    """
    Advance time by specified seconds.

    Args:
      seconds: Seconds to advance
    """
    self._current_time_ns += int(seconds * 1e9)

  def reset(self) -> None:
    """Reset time to start and clear call count."""
    self._current_time_ns = self.start_time_ns
    self._call_count = 0

  def set_time_ns(self, time_ns: int) -> None:
    """
    Set current time to specified value.

    Args:
      time_ns: New timestamp in nanoseconds
    """
    self._current_time_ns = time_ns


@dataclass
class DeterministicContext:
  """
  Context manager for deterministic execution.

  Seeds all random number generators and optionally provides
  a fake time source for reproducible timing.

  Usage:
    with DeterministicContext(seed=42) as ctx:
      # Random operations are deterministic
      values = [random.random() for _ in range(10)]
      np_values = np.random.random(10)

      # Access fake time if enabled
      if ctx.fake_time:
        current_time = ctx.time_source.monotonic()
  """

  seed: int = 42
  fake_time: bool = False
  start_time_ns: int = 0
  dt_ns: int = 10_000_000

  time_source: FakeTimeSource = field(init=False, repr=False)
  _original_random_state: Any = field(init=False, repr=False)
  _original_np_state: Any = field(init=False, repr=False)
  _patches: list = field(init=False, repr=False)

  def __post_init__(self):
    self.time_source = FakeTimeSource(
      start_time_ns=self.start_time_ns,
      dt_ns=self.dt_ns,
    )
    self._patches = []

  def __enter__(self) -> 'DeterministicContext':
    """Enter deterministic context."""
    # Save original random states
    self._original_random_state = random.getstate()
    self._original_np_state = np.random.get_state()

    # Set deterministic seeds
    random.seed(self.seed)
    np.random.seed(self.seed)

    # Optionally patch time functions
    if self.fake_time:
      self._patches = [
        patch('time.monotonic', self.time_source.monotonic),
        patch('time.monotonic_ns', self.time_source.monotonic_ns),
        patch('time.time', self.time_source.time),
        patch('time.time_ns', self.time_source.time_ns),
      ]
      for p in self._patches:
        p.start()

    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Exit deterministic context and restore original state."""
    # Stop time patches
    for p in self._patches:
      p.stop()
    self._patches = []

    # Restore original random states
    random.setstate(self._original_random_state)
    np.random.set_state(self._original_np_state)

    return False  # Don't suppress exceptions

  def reset(self) -> None:
    """Reset random seeds and time source to initial state."""
    random.seed(self.seed)
    np.random.seed(self.seed)
    self.time_source.reset()


@contextmanager
def deterministic_mode(
  seed: int = 42,
  fake_time: bool = False,
  start_time_ns: int = 0,
  dt_ns: int = 10_000_000,
):
  """
  Context manager for deterministic execution (functional style).

  Args:
    seed: Random seed for all generators
    fake_time: Whether to use fake time source
    start_time_ns: Start time for fake time source
    dt_ns: Time step for fake time source

  Yields:
    DeterministicContext instance
  """
  ctx = DeterministicContext(
    seed=seed,
    fake_time=fake_time,
    start_time_ns=start_time_ns,
    dt_ns=dt_ns,
  )
  with ctx:
    yield ctx


def seed_all(seed: int) -> None:
  """
  Seed all random number generators.

  Args:
    seed: Seed value for all generators
  """
  random.seed(seed)
  np.random.seed(seed)


def get_deterministic_hash(data: bytes, seed: int = 0) -> int:
  """
  Compute a deterministic hash of data.

  Uses a simple hash function that produces the same result
  regardless of Python hash randomization.

  Args:
    data: Bytes to hash
    seed: Optional seed for hash

  Returns:
    Deterministic hash value
  """
  # Simple FNV-1a hash for determinism
  FNV_PRIME = 0x01000193
  FNV_OFFSET = 0x811C9DC5

  hash_value = FNV_OFFSET ^ seed
  for byte in data:
    hash_value ^= byte
    hash_value = (hash_value * FNV_PRIME) & 0xFFFFFFFF

  return hash_value


class DeterministicRandom:
  """
  A standalone deterministic random number generator.

  Useful when you need isolated random state that doesn't affect
  the global random module.
  """

  def __init__(self, seed: int = 42):
    self._rng = random.Random(seed)
    self._np_rng = np.random.RandomState(seed)
    self.seed = seed

  def random(self) -> float:
    """Get random float in [0, 1)."""
    return self._rng.random()

  def randint(self, a: int, b: int) -> int:
    """Get random integer in [a, b]."""
    return self._rng.randint(a, b)

  def uniform(self, a: float, b: float) -> float:
    """Get random float in [a, b]."""
    return self._rng.uniform(a, b)

  def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Get random Gaussian/normal value."""
    return self._rng.gauss(mu, sigma)

  def choice(self, seq):
    """Choose random element from sequence."""
    return self._rng.choice(seq)

  def shuffle(self, seq):
    """Shuffle sequence in place."""
    self._rng.shuffle(seq)

  def sample(self, population, k: int):
    """Choose k unique random elements."""
    return self._rng.sample(population, k)

  def np_random(self, size=None) -> np.ndarray:
    """Get numpy array of random floats in [0, 1)."""
    return self._np_rng.random(size)

  def np_randn(self, *shape) -> np.ndarray:
    """Get numpy array of standard normal values."""
    return self._np_rng.randn(*shape)

  def np_randint(self, low: int, high: int = None, size=None) -> np.ndarray:
    """Get numpy array of random integers."""
    return self._np_rng.randint(low, high, size)

  def reset(self) -> None:
    """Reset to initial seed."""
    self._rng = random.Random(self.seed)
    self._np_rng = np.random.RandomState(self.seed)


class ReplayableSequence:
  """
  A sequence generator that can be replayed deterministically.

  Records generated values and can replay them exactly.
  Useful for debugging algorithm behavior.
  """

  def __init__(self, generator: Callable[[], Any], seed: int = 42):
    self._generator = generator
    self._seed = seed
    self._recorded: list[Any] = []
    self._replay_index: int = 0
    self._recording: bool = True
    self._rng = DeterministicRandom(seed)

  def next(self) -> Any:
    """Get next value (generate or replay)."""
    if self._recording:
      value = self._generator()
      self._recorded.append(value)
      return value
    else:
      if self._replay_index >= len(self._recorded):
        raise IndexError("Replay sequence exhausted")
      value = self._recorded[self._replay_index]
      self._replay_index += 1
      return value

  def start_recording(self) -> None:
    """Start recording mode."""
    self._recording = True
    self._recorded = []
    self._replay_index = 0

  def start_replay(self) -> None:
    """Start replay mode from beginning."""
    self._recording = False
    self._replay_index = 0

  @property
  def recorded_count(self) -> int:
    """Number of recorded values."""
    return len(self._recorded)

  @property
  def replay_remaining(self) -> int:
    """Number of values remaining in replay."""
    return len(self._recorded) - self._replay_index
