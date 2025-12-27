"""Tests for stat_live.py - running statistics utilities."""
import unittest
import numpy as np

from openpilot.common.stat_live import RunningStat, RunningStatFilter


class TestRunningStatInit(unittest.TestCase):
  """Test RunningStat initialization."""

  def test_init_no_priors(self):
    """Test initialization without priors."""
    stat = RunningStat()
    self.assertEqual(stat.M, 0.0)
    self.assertEqual(stat.S, 0.0)
    self.assertEqual(stat.n, 0)
    self.assertEqual(stat.max_trackable, -1)

  def test_init_with_priors(self):
    """Test initialization with priors."""
    priors = (10.0, 5.0, 100)
    stat = RunningStat(priors=priors)
    self.assertEqual(stat.M, 10.0)
    self.assertEqual(stat.S, 5.0)
    self.assertEqual(stat.n, 100)

  def test_init_with_max_trackable(self):
    """Test initialization with max_trackable."""
    stat = RunningStat(max_trackable=50)
    self.assertEqual(stat.max_trackable, 50)


class TestRunningStatReset(unittest.TestCase):
  """Test RunningStat reset method."""

  def test_reset_clears_values(self):
    """Test reset sets all values to zero."""
    stat = RunningStat(priors=(10.0, 5.0, 100))
    stat.reset()
    self.assertEqual(stat.M, 0.0)
    self.assertEqual(stat.S, 0.0)
    self.assertEqual(stat.n, 0)
    self.assertEqual(stat.M_last, 0.0)
    self.assertEqual(stat.S_last, 0.0)


class TestRunningStatPushData(unittest.TestCase):
  """Test RunningStat push_data method."""

  def test_push_single_value(self):
    """Test pushing a single value."""
    stat = RunningStat()
    stat.push_data(10.0)
    self.assertEqual(stat.n, 1)
    self.assertEqual(stat.mean(), 10.0)

  def test_push_multiple_values(self):
    """Test pushing multiple values calculates mean correctly."""
    stat = RunningStat()
    values = [10.0, 20.0, 30.0]
    for v in values:
      stat.push_data(v)
    self.assertEqual(stat.n, 3)
    self.assertAlmostEqual(stat.mean(), 20.0)

  def test_push_respects_max_trackable(self):
    """Test n stops incrementing at max_trackable."""
    stat = RunningStat(max_trackable=3)
    for i in range(10):
      stat.push_data(float(i))
    self.assertEqual(stat.n, 3)


class TestRunningStatMean(unittest.TestCase):
  """Test RunningStat mean method."""

  def test_mean_empty(self):
    """Test mean of empty stat is 0."""
    stat = RunningStat()
    self.assertEqual(stat.mean(), 0.0)

  def test_mean_single_value(self):
    """Test mean of single value."""
    stat = RunningStat()
    stat.push_data(42.0)
    self.assertEqual(stat.mean(), 42.0)

  def test_mean_multiple_values(self):
    """Test mean of multiple values."""
    stat = RunningStat()
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
      stat.push_data(v)
    self.assertAlmostEqual(stat.mean(), 3.0)


class TestRunningStatVariance(unittest.TestCase):
  """Test RunningStat variance method."""

  def test_variance_empty(self):
    """Test variance of empty stat is 0."""
    stat = RunningStat()
    self.assertEqual(stat.variance(), 0.0)

  def test_variance_single_value(self):
    """Test variance of single value is 0."""
    stat = RunningStat()
    stat.push_data(10.0)
    self.assertEqual(stat.variance(), 0.0)

  def test_variance_identical_values(self):
    """Test variance of identical values is 0."""
    stat = RunningStat()
    for _ in range(5):
      stat.push_data(10.0)
    self.assertAlmostEqual(stat.variance(), 0.0)

  def test_variance_different_values(self):
    """Test variance of different values."""
    stat = RunningStat()
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for v in values:
      stat.push_data(v)
    expected_var = np.var(values, ddof=1)
    self.assertAlmostEqual(stat.variance(), expected_var)


class TestRunningStatStd(unittest.TestCase):
  """Test RunningStat std method."""

  def test_std_empty(self):
    """Test std of empty stat is 0."""
    stat = RunningStat()
    self.assertEqual(stat.std(), 0.0)

  def test_std_is_sqrt_variance(self):
    """Test std is square root of variance."""
    stat = RunningStat()
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
      stat.push_data(v)
    self.assertAlmostEqual(stat.std(), np.sqrt(stat.variance()))


class TestRunningStatParamsToSave(unittest.TestCase):
  """Test RunningStat params_to_save method."""

  def test_params_to_save(self):
    """Test params_to_save returns correct list."""
    stat = RunningStat()
    for v in [10.0, 20.0, 30.0]:
      stat.push_data(v)

    params = stat.params_to_save()

    self.assertEqual(len(params), 3)
    self.assertEqual(params[0], stat.M)
    self.assertEqual(params[1], stat.S)
    self.assertEqual(params[2], stat.n)


class TestRunningStatFilterInit(unittest.TestCase):
  """Test RunningStatFilter initialization."""

  def test_init_no_priors(self):
    """Test initialization without priors."""
    filt = RunningStatFilter()
    self.assertIsNotNone(filt.raw_stat)
    self.assertIsNotNone(filt.filtered_stat)

  def test_init_with_raw_priors(self):
    """Test initialization with raw priors."""
    raw_priors = (5.0, 2.0, 10)
    filt = RunningStatFilter(raw_priors=raw_priors)
    self.assertEqual(filt.raw_stat.M, 5.0)

  def test_init_with_filtered_priors(self):
    """Test initialization with filtered priors."""
    filtered_priors = (3.0, 1.0, 5)
    filt = RunningStatFilter(filtered_priors=filtered_priors)
    self.assertEqual(filt.filtered_stat.M, 3.0)

  def test_init_with_max_trackable(self):
    """Test initialization with max_trackable applies to filtered_stat."""
    filt = RunningStatFilter(max_trackable=100)
    self.assertEqual(filt.filtered_stat.max_trackable, 100)
    self.assertEqual(filt.raw_stat.max_trackable, -1)


class TestRunningStatFilterReset(unittest.TestCase):
  """Test RunningStatFilter reset method."""

  def test_reset_clears_both_stats(self):
    """Test reset clears both raw and filtered stats."""
    filt = RunningStatFilter()
    filt.push_and_update(10.0)
    filt.push_and_update(20.0)

    filt.reset()

    self.assertEqual(filt.raw_stat.n, 0)
    self.assertEqual(filt.filtered_stat.n, 0)


class TestRunningStatFilterPushAndUpdate(unittest.TestCase):
  """Test RunningStatFilter push_and_update method."""

  def test_push_and_update_updates_raw(self):
    """Test push_and_update always updates raw_stat."""
    filt = RunningStatFilter()
    filt.push_and_update(10.0)
    self.assertEqual(filt.raw_stat.n, 1)

  def test_push_and_update_stable_data(self):
    """Test push_and_update with stable data updates filtered_stat."""
    filt = RunningStatFilter()
    # Push stable data (low variance)
    for v in [10.0, 10.0, 10.0, 10.0]:
      filt.push_and_update(v)
    # Filtered stat should have data
    self.assertGreater(filt.filtered_stat.n, 0)


if __name__ == '__main__':
  unittest.main()
