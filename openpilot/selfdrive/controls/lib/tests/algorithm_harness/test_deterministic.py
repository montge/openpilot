"""
Tests for deterministic execution utilities.

Tests cover:
- FakeTimeSource timing behavior
- DeterministicContext random seeding
- Time patching functionality
- DeterministicRandom isolation
- ReplayableSequence recording/replay
"""

import random
import time
import pytest
import numpy as np

from openpilot.selfdrive.controls.lib.tests.algorithm_harness.deterministic import (
  FakeTimeSource,
  DeterministicContext,
  deterministic_mode,
  seed_all,
  get_deterministic_hash,
  DeterministicRandom,
  ReplayableSequence,
)


class TestFakeTimeSource:
  """Tests for FakeTimeSource."""

  def test_initial_time(self):
    """Test initial time is start time."""
    ts = FakeTimeSource(start_time_ns=1000)
    assert ts.current_time_ns == 1000
    assert ts.peek_ns() == 1000

  def test_default_start_time(self):
    """Test default start time is 0."""
    ts = FakeTimeSource()
    assert ts.current_time_ns == 0

  def test_monotonic_ns_advances(self):
    """Test monotonic_ns advances by dt_ns."""
    ts = FakeTimeSource(start_time_ns=0, dt_ns=10_000_000)

    t1 = ts.monotonic_ns()
    t2 = ts.monotonic_ns()
    t3 = ts.monotonic_ns()

    assert t1 == 0
    assert t2 == 10_000_000
    assert t3 == 20_000_000

  def test_monotonic_seconds(self):
    """Test monotonic returns seconds."""
    ts = FakeTimeSource(start_time_ns=0, dt_ns=100_000_000)  # 100ms

    t1 = ts.monotonic()
    t2 = ts.monotonic()

    assert t1 == 0.0
    assert abs(t2 - 0.1) < 1e-9

  def test_peek_does_not_advance(self):
    """Test peek_ns doesn't advance time."""
    ts = FakeTimeSource(start_time_ns=1000)

    peek1 = ts.peek_ns()
    peek2 = ts.peek_ns()
    peek3 = ts.peek_ns()

    assert peek1 == peek2 == peek3 == 1000

  def test_advance_manual(self):
    """Test manual time advancement."""
    ts = FakeTimeSource(start_time_ns=0)

    ts.advance(5_000_000)
    assert ts.peek_ns() == 5_000_000

    ts.advance(3_000_000)
    assert ts.peek_ns() == 8_000_000

  def test_advance_seconds(self):
    """Test advancement by seconds."""
    ts = FakeTimeSource(start_time_ns=0)

    ts.advance_seconds(1.5)
    assert ts.peek_ns() == 1_500_000_000

  def test_reset(self):
    """Test reset returns to start time."""
    ts = FakeTimeSource(start_time_ns=100)

    ts.monotonic_ns()
    ts.monotonic_ns()
    ts.monotonic_ns()

    assert ts.call_count == 3
    assert ts.current_time_ns != 100

    ts.reset()

    assert ts.call_count == 0
    assert ts.current_time_ns == 100

  def test_set_time(self):
    """Test setting time directly."""
    ts = FakeTimeSource(start_time_ns=0)

    ts.set_time_ns(1_000_000_000)
    assert ts.peek_ns() == 1_000_000_000

  def test_time_alias(self):
    """Test time() is alias for monotonic()."""
    ts = FakeTimeSource(start_time_ns=0, dt_ns=100_000_000)

    t1 = ts.time()
    t2 = ts.time()

    assert t1 == 0.0
    assert abs(t2 - 0.1) < 1e-9

  def test_call_count(self):
    """Test call count tracking."""
    ts = FakeTimeSource()

    assert ts.call_count == 0

    for i in range(10):
      ts.monotonic_ns()
      assert ts.call_count == i + 1

  def test_current_time_s(self):
    """Test current_time_s property."""
    ts = FakeTimeSource(start_time_ns=1_500_000_000)
    assert ts.current_time_s == 1.5


class TestDeterministicContext:
  """Tests for DeterministicContext."""

  def test_seeds_random(self):
    """Test context seeds random module."""
    # Get values with seed 42
    with DeterministicContext(seed=42):
      values1 = [random.random() for _ in range(5)]

    # Get values again with same seed
    with DeterministicContext(seed=42):
      values2 = [random.random() for _ in range(5)]

    assert values1 == values2

  def test_seeds_numpy(self):
    """Test context seeds numpy random."""
    with DeterministicContext(seed=42):
      arr1 = np.random.random(5)

    with DeterministicContext(seed=42):
      arr2 = np.random.random(5)

    np.testing.assert_array_equal(arr1, arr2)

  def test_different_seeds_different_values(self):
    """Test different seeds produce different values."""
    with DeterministicContext(seed=42):
      values1 = [random.random() for _ in range(5)]

    with DeterministicContext(seed=123):
      values2 = [random.random() for _ in range(5)]

    assert values1 != values2

  def test_restores_random_state(self):
    """Test context restores original random state on exit."""
    # Set known state
    random.seed(999)
    before = random.random()

    # Reset and do different operations in context
    random.seed(999)
    with DeterministicContext(seed=42):
      _ = [random.random() for _ in range(100)]

    # State should be restored
    after = random.random()
    assert before == after

  def test_restores_numpy_state(self):
    """Test context restores original numpy state on exit."""
    np.random.seed(999)
    before = np.random.random()

    np.random.seed(999)
    with DeterministicContext(seed=42):
      _ = np.random.random(100)

    after = np.random.random()
    assert before == after

  def test_fake_time_patches(self):
    """Test fake time patches time module."""
    with DeterministicContext(fake_time=True, dt_ns=100_000_000):
      t1 = time.monotonic()
      t2 = time.monotonic()
      t3 = time.monotonic()

      assert t1 == 0.0
      assert abs(t2 - 0.1) < 1e-9
      assert abs(t3 - 0.2) < 1e-9

  def test_time_source_accessible(self):
    """Test time source is accessible from context."""
    with DeterministicContext(fake_time=True) as ctx:
      assert ctx.time_source is not None
      ctx.time_source.advance_seconds(1.0)
      t = time.monotonic()
      assert abs(t - 1.0) < 1e-6

  def test_reset_within_context(self):
    """Test reset within context."""
    with DeterministicContext(seed=42) as ctx:
      values1 = [random.random() for _ in range(5)]

      ctx.reset()

      values2 = [random.random() for _ in range(5)]

      assert values1 == values2

  def test_no_fake_time_by_default(self):
    """Test fake time is not enabled by default."""
    before = time.monotonic()
    with DeterministicContext(seed=42):
      after = time.monotonic()

    # Real time should have passed
    assert after >= before


class TestDeterministicModeFunction:
  """Tests for deterministic_mode context manager function."""

  def test_basic_usage(self):
    """Test basic functional style usage."""
    with deterministic_mode(seed=42):
      values1 = [random.random() for _ in range(5)]

    with deterministic_mode(seed=42):
      values2 = [random.random() for _ in range(5)]

    assert values1 == values2

  def test_with_fake_time(self):
    """Test functional style with fake time."""
    with deterministic_mode(fake_time=True, dt_ns=50_000_000):
      t1 = time.monotonic()
      t2 = time.monotonic()

      assert t1 == 0.0
      assert abs(t2 - 0.05) < 1e-9


class TestSeedAll:
  """Tests for seed_all utility."""

  def test_seeds_both_generators(self):
    """Test seed_all seeds both random and numpy."""
    seed_all(42)
    r1 = random.random()
    n1 = np.random.random()

    seed_all(42)
    r2 = random.random()
    n2 = np.random.random()

    assert r1 == r2
    assert n1 == n2


class TestGetDeterministicHash:
  """Tests for get_deterministic_hash."""

  def test_same_input_same_hash(self):
    """Test same input produces same hash."""
    data = b"hello world"
    h1 = get_deterministic_hash(data)
    h2 = get_deterministic_hash(data)
    assert h1 == h2

  def test_different_input_different_hash(self):
    """Test different input produces different hash."""
    h1 = get_deterministic_hash(b"hello")
    h2 = get_deterministic_hash(b"world")
    assert h1 != h2

  def test_seed_affects_hash(self):
    """Test seed affects hash result."""
    data = b"hello"
    h1 = get_deterministic_hash(data, seed=0)
    h2 = get_deterministic_hash(data, seed=42)
    assert h1 != h2

  def test_hash_is_consistent(self):
    """Test hash is consistent across calls."""
    data = b"test data for hashing"
    expected = get_deterministic_hash(data)

    for _ in range(100):
      assert get_deterministic_hash(data) == expected


class TestDeterministicRandom:
  """Tests for DeterministicRandom class."""

  def test_random_reproducible(self):
    """Test random values are reproducible."""
    rng1 = DeterministicRandom(seed=42)
    rng2 = DeterministicRandom(seed=42)

    for _ in range(10):
      assert rng1.random() == rng2.random()

  def test_randint(self):
    """Test randint reproducibility."""
    rng1 = DeterministicRandom(seed=42)
    rng2 = DeterministicRandom(seed=42)

    for _ in range(10):
      assert rng1.randint(0, 100) == rng2.randint(0, 100)

  def test_uniform(self):
    """Test uniform distribution."""
    rng = DeterministicRandom(seed=42)
    values = [rng.uniform(0, 10) for _ in range(100)]

    assert all(0 <= v <= 10 for v in values)

  def test_gauss(self):
    """Test Gaussian distribution."""
    rng = DeterministicRandom(seed=42)
    values = [rng.gauss(0, 1) for _ in range(1000)]

    # Mean should be close to 0
    assert abs(sum(values) / len(values)) < 0.2

  def test_choice(self):
    """Test choice reproducibility."""
    rng1 = DeterministicRandom(seed=42)
    rng2 = DeterministicRandom(seed=42)
    options = ['a', 'b', 'c', 'd', 'e']

    for _ in range(10):
      assert rng1.choice(options) == rng2.choice(options)

  def test_shuffle(self):
    """Test shuffle reproducibility."""
    rng1 = DeterministicRandom(seed=42)
    rng2 = DeterministicRandom(seed=42)

    list1 = [1, 2, 3, 4, 5]
    list2 = [1, 2, 3, 4, 5]

    rng1.shuffle(list1)
    rng2.shuffle(list2)

    assert list1 == list2

  def test_sample(self):
    """Test sample reproducibility."""
    rng1 = DeterministicRandom(seed=42)
    rng2 = DeterministicRandom(seed=42)
    population = list(range(100))

    s1 = rng1.sample(population, 10)
    s2 = rng2.sample(population, 10)

    assert s1 == s2

  def test_np_random(self):
    """Test numpy random array generation."""
    rng1 = DeterministicRandom(seed=42)
    rng2 = DeterministicRandom(seed=42)

    arr1 = rng1.np_random(10)
    arr2 = rng2.np_random(10)

    np.testing.assert_array_equal(arr1, arr2)

  def test_np_randn(self):
    """Test numpy normal distribution."""
    rng1 = DeterministicRandom(seed=42)
    rng2 = DeterministicRandom(seed=42)

    arr1 = rng1.np_randn(5, 5)
    arr2 = rng2.np_randn(5, 5)

    np.testing.assert_array_equal(arr1, arr2)

  def test_np_randint(self):
    """Test numpy random integers."""
    rng1 = DeterministicRandom(seed=42)
    rng2 = DeterministicRandom(seed=42)

    arr1 = rng1.np_randint(0, 100, size=10)
    arr2 = rng2.np_randint(0, 100, size=10)

    np.testing.assert_array_equal(arr1, arr2)

  def test_reset(self):
    """Test reset returns to initial state."""
    rng = DeterministicRandom(seed=42)

    values1 = [rng.random() for _ in range(10)]

    rng.reset()

    values2 = [rng.random() for _ in range(10)]

    assert values1 == values2

  def test_isolated_from_global(self):
    """Test DeterministicRandom is isolated from global random."""
    rng = DeterministicRandom(seed=42)

    # Change global random state
    random.seed(999)

    # DeterministicRandom should be unaffected
    v1 = rng.random()

    rng.reset()
    random.seed(123)  # Different global seed

    v2 = rng.random()

    assert v1 == v2


class TestReplayableSequence:
  """Tests for ReplayableSequence."""

  def test_recording(self):
    """Test recording mode captures values."""
    counter = [0]

    def gen():
      counter[0] += 1
      return counter[0]

    seq = ReplayableSequence(gen)

    # Record 5 values
    values = [seq.next() for _ in range(5)]

    assert values == [1, 2, 3, 4, 5]
    assert seq.recorded_count == 5

  def test_replay(self):
    """Test replay mode returns recorded values."""
    counter = [0]

    def gen():
      counter[0] += 1
      return counter[0]

    seq = ReplayableSequence(gen)

    # Record values
    original = [seq.next() for _ in range(5)]

    # Start replay
    seq.start_replay()
    counter[0] = 100  # Change counter (should be ignored in replay)

    replayed = [seq.next() for _ in range(5)]

    assert original == replayed

  def test_replay_exhausted_raises(self):
    """Test replay raises when exhausted."""
    seq = ReplayableSequence(lambda: 1)

    seq.next()
    seq.next()
    seq.next()

    seq.start_replay()

    seq.next()
    seq.next()
    seq.next()

    with pytest.raises(IndexError, match="exhausted"):
      seq.next()

  def test_replay_remaining(self):
    """Test replay_remaining count."""
    seq = ReplayableSequence(lambda: 1)

    for _ in range(5):
      seq.next()

    seq.start_replay()

    assert seq.replay_remaining == 5

    seq.next()
    assert seq.replay_remaining == 4

    seq.next()
    seq.next()
    assert seq.replay_remaining == 2

  def test_restart_recording(self):
    """Test restarting recording clears history."""
    counter = [0]

    def gen():
      counter[0] += 1
      return counter[0]

    seq = ReplayableSequence(gen)

    # First recording
    seq.next()
    seq.next()
    seq.next()
    assert seq.recorded_count == 3

    # Restart recording
    seq.start_recording()
    assert seq.recorded_count == 0

    seq.next()
    seq.next()
    assert seq.recorded_count == 2


@pytest.mark.algorithm_benchmark
class TestDeterministicPerformance:
  """Performance tests for deterministic utilities."""

  def test_fake_time_overhead(self):
    """Test FakeTimeSource has low overhead."""
    ts = FakeTimeSource()

    # Call many times
    for _ in range(10000):
      ts.monotonic_ns()

    assert ts.call_count == 10000

  def test_deterministic_context_overhead(self):
    """Test DeterministicContext has acceptable overhead."""
    # Many context entries/exits
    for _ in range(100):
      with DeterministicContext(seed=42):
        _ = [random.random() for _ in range(10)]

  def test_deterministic_random_performance(self):
    """Test DeterministicRandom performance."""
    rng = DeterministicRandom(seed=42)

    # Generate many values
    for _ in range(10000):
      rng.random()
      rng.randint(0, 100)

    # Should complete quickly (implicit test - no timeout)
