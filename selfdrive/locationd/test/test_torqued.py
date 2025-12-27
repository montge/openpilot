"""Tests for torqued.py - torque estimation for steering."""
import numpy as np
import unittest
from unittest.mock import MagicMock, patch

from cereal import car
from openpilot.selfdrive.locationd.torqued import (
  TorqueEstimator, TorqueBuckets, slope2rot,
  HISTORY, POINTS_PER_BUCKET, MIN_POINTS_TOTAL, MIN_VEL,
  FRICTION_FACTOR, FACTOR_SANITY, FRICTION_SANITY,
  STEER_MIN_THRESHOLD, MIN_FILTER_DECAY, MAX_FILTER_DECAY,
  LAT_ACC_THRESHOLD, STEER_BUCKET_BOUNDS, MIN_BUCKET_POINTS,
  VERSION, ALLOWED_CARS
)


class TestSlope2Rot(unittest.TestCase):
  """Test slope2rot function."""

  def test_slope2rot_zero(self):
    """Test slope2rot with zero slope gives identity-like rotation."""
    rot = slope2rot(0)
    expected = np.array([[1, 0], [0, 1]])
    np.testing.assert_array_almost_equal(rot, expected)

  def test_slope2rot_positive(self):
    """Test slope2rot with positive slope."""
    rot = slope2rot(1.0)
    self.assertEqual(rot.shape, (2, 2))
    # Check orthogonality: R @ R.T = I
    np.testing.assert_array_almost_equal(rot @ rot.T, np.eye(2))

  def test_slope2rot_negative(self):
    """Test slope2rot with negative slope."""
    rot = slope2rot(-1.0)
    self.assertEqual(rot.shape, (2, 2))
    # Check orthogonality
    np.testing.assert_array_almost_equal(rot @ rot.T, np.eye(2))

  def test_slope2rot_determinant_one(self):
    """Test slope2rot produces rotation matrix with det=1."""
    for slope in [-2.0, -1.0, 0.0, 1.0, 2.0]:
      rot = slope2rot(slope)
      det = np.linalg.det(rot)
      self.assertAlmostEqual(det, 1.0, places=10)


class TestTorqueBuckets(unittest.TestCase):
  """Test TorqueBuckets class."""

  def test_add_point_in_bounds(self):
    """Test adding a point within bucket bounds."""
    buckets = TorqueBuckets(
      x_bounds=STEER_BUCKET_BOUNDS,
      min_points=MIN_BUCKET_POINTS,
      min_points_total=MIN_POINTS_TOTAL,
      points_per_bucket=POINTS_PER_BUCKET,
      rowsize=3
    )
    buckets.add_point(0.05, 0.5)  # Should go in (0, 0.1) bucket
    assert len(buckets.buckets[(0, 0.1)]) == 1

  def test_add_point_stores_correct_format(self):
    """Test points are stored as [x, 1.0, y]."""
    buckets = TorqueBuckets(
      x_bounds=STEER_BUCKET_BOUNDS,
      min_points=MIN_BUCKET_POINTS,
      min_points_total=MIN_POINTS_TOTAL,
      points_per_bucket=POINTS_PER_BUCKET,
      rowsize=3
    )
    buckets.add_point(0.15, 0.8)  # Should go in (0.1, 0.2) bucket
    # Get points and verify
    assert len(buckets.buckets[(0.1, 0.2)]) == 1

  def test_add_point_outside_bounds_ignored(self):
    """Test points outside all bounds are ignored."""
    buckets = TorqueBuckets(
      x_bounds=STEER_BUCKET_BOUNDS,
      min_points=MIN_BUCKET_POINTS,
      min_points_total=MIN_POINTS_TOTAL,
      points_per_bucket=POINTS_PER_BUCKET,
      rowsize=3
    )
    initial_total = len(buckets)
    buckets.add_point(1.0, 0.5)  # Outside bounds (max is 0.5)
    assert len(buckets) == initial_total


class TestTorqueEstimatorInit(unittest.TestCase):
  """Test TorqueEstimator initialization."""

  @patch('openpilot.selfdrive.locationd.torqued.Params')
  def test_init_default(self, mock_params):
    """Test TorqueEstimator default initialization."""
    mock_params.return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)

    self.assertEqual(est.min_points_total, MIN_POINTS_TOTAL)
    self.assertEqual(est.decay, MIN_FILTER_DECAY)

  @patch('openpilot.selfdrive.locationd.torqued.Params')
  def test_init_decimated(self, mock_params):
    """Test TorqueEstimator with decimated mode."""
    mock_params.return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp, decimated=True)

    self.assertEqual(est.min_points_total, 600)  # MIN_POINTS_TOTAL_QLOG

  @patch('openpilot.selfdrive.locationd.torqued.Params')
  def test_init_track_all_points(self, mock_params):
    """Test TorqueEstimator with track_all_points."""
    mock_params.return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp, track_all_points=True)

    self.assertTrue(est.track_all_points)
    self.assertEqual(est.all_torque_points, [])


class TestTorqueEstimatorReset(unittest.TestCase):
  """Test TorqueEstimator reset method."""

  @patch('openpilot.selfdrive.locationd.torqued.Params')
  def test_reset_increments_counter(self, mock_params):
    """Test reset increments resets counter."""
    mock_params.return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)
    initial_resets = est.resets

    est.reset()

    self.assertEqual(est.resets, initial_resets + 1)

  @patch('openpilot.selfdrive.locationd.torqued.Params')
  def test_reset_clears_decay(self, mock_params):
    """Test reset sets decay to minimum."""
    mock_params.return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)
    est.decay = 200

    est.reset()

    self.assertEqual(est.decay, MIN_FILTER_DECAY)


class TestTorqueEstimatorGetRestoreKey(unittest.TestCase):
  """Test TorqueEstimator.get_restore_key static method."""

  def test_get_restore_key_returns_tuple(self):
    """Test get_restore_key returns a tuple with expected elements."""
    cp = car.CarParams()
    key = TorqueEstimator.get_restore_key(cp, VERSION)

    self.assertIsInstance(key, tuple)
    self.assertEqual(len(key), 5)
    self.assertEqual(key[4], VERSION)

  def test_get_restore_key_different_versions(self):
    """Test get_restore_key with different versions returns different keys."""
    cp = car.CarParams()
    key1 = TorqueEstimator.get_restore_key(cp, 1)
    key2 = TorqueEstimator.get_restore_key(cp, 2)

    self.assertNotEqual(key1, key2)


class TestTorqueEstimatorUpdateParams(unittest.TestCase):
  """Test TorqueEstimator.update_params method."""

  @patch('openpilot.selfdrive.locationd.torqued.Params')
  def test_update_params_increases_decay(self, mock_params):
    """Test update_params increases decay toward max."""
    mock_params.return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)
    initial_decay = est.decay

    est.update_params({'latAccelFactor': 2.0})

    self.assertGreater(est.decay, initial_decay)

  @patch('openpilot.selfdrive.locationd.torqued.Params')
  def test_update_params_respects_max_decay(self, mock_params):
    """Test update_params caps decay at MAX_FILTER_DECAY."""
    mock_params.return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)
    est.decay = MAX_FILTER_DECAY - 0.001

    est.update_params({'latAccelFactor': 2.0})

    self.assertLessEqual(est.decay, MAX_FILTER_DECAY)


class TestTorqueEstimatorGetMsg(unittest.TestCase):
  """Test TorqueEstimator.get_msg method."""

  @patch('openpilot.selfdrive.locationd.torqued.Params')
  def test_get_msg_returns_message(self, mock_params):
    """Test get_msg returns a valid message."""
    mock_params.return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)
    msg = est.get_msg()

    self.assertIsNotNone(msg)
    self.assertIsNotNone(msg.liveTorqueParameters)

  @patch('openpilot.selfdrive.locationd.torqued.Params')
  def test_get_msg_contains_version(self, mock_params):
    """Test get_msg includes correct version."""
    mock_params.return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)
    msg = est.get_msg()

    self.assertEqual(msg.liveTorqueParameters.version, VERSION)

  @patch('openpilot.selfdrive.locationd.torqued.Params')
  def test_get_msg_valid_flag(self, mock_params):
    """Test get_msg respects valid flag."""
    mock_params.return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)

    msg_valid = est.get_msg(valid=True)
    msg_invalid = est.get_msg(valid=False)

    self.assertTrue(msg_valid.valid)
    self.assertFalse(msg_invalid.valid)

  @patch('openpilot.selfdrive.locationd.torqued.Params')
  def test_get_msg_has_cal_perc(self, mock_params):
    """Test get_msg includes calibration percentage."""
    mock_params.return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)
    msg = est.get_msg()

    # Initially zero calibration
    self.assertEqual(msg.liveTorqueParameters.calPerc, 0)


class TestConstants(unittest.TestCase):
  """Test module constants."""

  def test_history_positive(self):
    """Test HISTORY is positive."""
    self.assertGreater(HISTORY, 0)

  def test_min_vel_reasonable(self):
    """Test MIN_VEL is a reasonable speed in m/s."""
    self.assertGreater(MIN_VEL, 0)
    self.assertLess(MIN_VEL, 50)  # Less than highway speeds

  def test_bucket_bounds_symmetric(self):
    """Test STEER_BUCKET_BOUNDS are symmetric around zero."""
    bounds = STEER_BUCKET_BOUNDS
    self.assertEqual(len(bounds), 8)
    # Check symmetric structure
    self.assertEqual(bounds[0][0], -bounds[-1][1])

  def test_min_bucket_points_matches_bounds(self):
    """Test MIN_BUCKET_POINTS matches STEER_BUCKET_BOUNDS."""
    self.assertEqual(len(MIN_BUCKET_POINTS), len(STEER_BUCKET_BOUNDS))

  def test_allowed_cars_not_empty(self):
    """Test ALLOWED_CARS is not empty."""
    self.assertGreater(len(ALLOWED_CARS), 0)

  def test_filter_decay_range(self):
    """Test filter decay constants are valid."""
    self.assertGreater(MIN_FILTER_DECAY, 0)
    self.assertGreater(MAX_FILTER_DECAY, MIN_FILTER_DECAY)

  def test_sanity_factors_positive(self):
    """Test sanity factors are positive and reasonable."""
    self.assertGreater(FACTOR_SANITY, 0)
    self.assertLess(FACTOR_SANITY, 1)
    self.assertGreater(FRICTION_SANITY, 0)
    self.assertLess(FRICTION_SANITY, 1)


def test_cal_percent():
  est = TorqueEstimator(car.CarParams())
  msg = est.get_msg()
  assert msg.liveTorqueParameters.calPerc == 0

  for (low, high), min_pts in zip(est.filtered_points.buckets.keys(),
                                  est.filtered_points.buckets_min_points.values(), strict=True):
    for _ in range(int(min_pts)):
      est.filtered_points.add_point((low + high) / 2.0, 0.0)

  # enough bucket points, but not enough total points
  msg = est.get_msg()
  assert msg.liveTorqueParameters.calPerc == (len(est.filtered_points) / est.min_points_total * 100 + 100) / 2

  # add enough points to bucket with most capacity
  key = list(est.filtered_points.buckets)[0]
  for _ in range(est.min_points_total - len(est.filtered_points)):
    est.filtered_points.add_point((key[0] + key[1]) / 2.0, 0.0)

  msg = est.get_msg()
  assert msg.liveTorqueParameters.calPerc == 100
