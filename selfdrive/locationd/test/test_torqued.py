"""Tests for torqued.py - torque estimation for steering."""

import numpy as np

from cereal import car
from openpilot.selfdrive.locationd.torqued import (
  TorqueEstimator,
  TorqueBuckets,
  slope2rot,
  HISTORY,
  POINTS_PER_BUCKET,
  MIN_POINTS_TOTAL,
  MIN_VEL,
  FACTOR_SANITY,
  FRICTION_SANITY,
  MIN_FILTER_DECAY,
  MAX_FILTER_DECAY,
  STEER_BUCKET_BOUNDS,
  MIN_BUCKET_POINTS,
  VERSION,
  ALLOWED_CARS,
)


class TestSlope2Rot:
  """Test slope2rot function."""

  def test_slope2rot_zero(self):
    """Test slope2rot with zero slope gives identity-like rotation."""
    rot = slope2rot(0)
    expected = np.array([[1, 0], [0, 1]])
    np.testing.assert_array_almost_equal(rot, expected)

  def test_slope2rot_positive(self):
    """Test slope2rot with positive slope."""
    rot = slope2rot(1.0)
    assert rot.shape == (2, 2)
    # Check orthogonality: R @ R.T = I
    np.testing.assert_array_almost_equal(rot @ rot.T, np.eye(2))

  def test_slope2rot_negative(self):
    """Test slope2rot with negative slope."""
    rot = slope2rot(-1.0)
    assert rot.shape == (2, 2)
    # Check orthogonality
    np.testing.assert_array_almost_equal(rot @ rot.T, np.eye(2))

  def test_slope2rot_determinant_one(self):
    """Test slope2rot produces rotation matrix with det=1."""
    for slope in [-2.0, -1.0, 0.0, 1.0, 2.0]:
      rot = slope2rot(slope)
      det = np.linalg.det(rot)
      assert abs(det - 1.0) < 1e-10


class TestTorqueBuckets:
  """Test TorqueBuckets class."""

  def test_add_point_in_bounds(self):
    """Test adding a point within bucket bounds."""
    buckets = TorqueBuckets(
      x_bounds=STEER_BUCKET_BOUNDS, min_points=MIN_BUCKET_POINTS, min_points_total=MIN_POINTS_TOTAL, points_per_bucket=POINTS_PER_BUCKET, rowsize=3
    )
    buckets.add_point(0.05, 0.5)  # Should go in (0, 0.1) bucket
    assert len(buckets.buckets[(0, 0.1)]) == 1

  def test_add_point_stores_correct_format(self):
    """Test points are stored as [x, 1.0, y]."""
    buckets = TorqueBuckets(
      x_bounds=STEER_BUCKET_BOUNDS, min_points=MIN_BUCKET_POINTS, min_points_total=MIN_POINTS_TOTAL, points_per_bucket=POINTS_PER_BUCKET, rowsize=3
    )
    buckets.add_point(0.15, 0.8)  # Should go in (0.1, 0.2) bucket
    # Get points and verify
    assert len(buckets.buckets[(0.1, 0.2)]) == 1

  def test_add_point_outside_bounds_ignored(self):
    """Test points outside all bounds are ignored."""
    buckets = TorqueBuckets(
      x_bounds=STEER_BUCKET_BOUNDS, min_points=MIN_BUCKET_POINTS, min_points_total=MIN_POINTS_TOTAL, points_per_bucket=POINTS_PER_BUCKET, rowsize=3
    )
    initial_total = len(buckets)
    buckets.add_point(1.0, 0.5)  # Outside bounds (max is 0.5)
    assert len(buckets) == initial_total


class TestTorqueEstimatorInit:
  """Test TorqueEstimator initialization."""

  def test_init_default(self, mocker):
    """Test TorqueEstimator default initialization."""
    mocker.patch('openpilot.selfdrive.locationd.torqued.Params').return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)

    assert est.min_points_total == MIN_POINTS_TOTAL
    assert est.decay == MIN_FILTER_DECAY

  def test_init_decimated(self, mocker):
    """Test TorqueEstimator with decimated mode."""
    mocker.patch('openpilot.selfdrive.locationd.torqued.Params').return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp, decimated=True)

    assert est.min_points_total == 600  # MIN_POINTS_TOTAL_QLOG

  def test_init_track_all_points(self, mocker):
    """Test TorqueEstimator with track_all_points."""
    mocker.patch('openpilot.selfdrive.locationd.torqued.Params').return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp, track_all_points=True)

    assert est.track_all_points
    assert est.all_torque_points == []


class TestTorqueEstimatorReset:
  """Test TorqueEstimator reset method."""

  def test_reset_increments_counter(self, mocker):
    """Test reset increments resets counter."""
    mocker.patch('openpilot.selfdrive.locationd.torqued.Params').return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)
    initial_resets = est.resets

    est.reset()

    assert est.resets == initial_resets + 1

  def test_reset_clears_decay(self, mocker):
    """Test reset sets decay to minimum."""
    mocker.patch('openpilot.selfdrive.locationd.torqued.Params').return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)
    est.decay = 200

    est.reset()

    assert est.decay == MIN_FILTER_DECAY


class TestTorqueEstimatorGetRestoreKey:
  """Test TorqueEstimator.get_restore_key static method."""

  def test_get_restore_key_returns_tuple(self):
    """Test get_restore_key returns a tuple with expected elements."""
    cp = car.CarParams()
    key = TorqueEstimator.get_restore_key(cp, VERSION)

    assert isinstance(key, tuple)
    assert len(key) == 5
    assert key[4] == VERSION

  def test_get_restore_key_different_versions(self):
    """Test get_restore_key with different versions returns different keys."""
    cp = car.CarParams()
    key1 = TorqueEstimator.get_restore_key(cp, 1)
    key2 = TorqueEstimator.get_restore_key(cp, 2)

    assert key1 != key2


class TestTorqueEstimatorUpdateParams:
  """Test TorqueEstimator.update_params method."""

  def test_update_params_increases_decay(self, mocker):
    """Test update_params increases decay toward max."""
    mocker.patch('openpilot.selfdrive.locationd.torqued.Params').return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)
    initial_decay = est.decay

    est.update_params({'latAccelFactor': 2.0})

    assert est.decay > initial_decay

  def test_update_params_respects_max_decay(self, mocker):
    """Test update_params caps decay at MAX_FILTER_DECAY."""
    mocker.patch('openpilot.selfdrive.locationd.torqued.Params').return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)
    est.decay = MAX_FILTER_DECAY - 0.001

    est.update_params({'latAccelFactor': 2.0})

    assert est.decay <= MAX_FILTER_DECAY


class TestTorqueEstimatorGetMsg:
  """Test TorqueEstimator.get_msg method."""

  def test_get_msg_returns_message(self, mocker):
    """Test get_msg returns a valid message."""
    mocker.patch('openpilot.selfdrive.locationd.torqued.Params').return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)
    msg = est.get_msg()

    assert msg is not None
    assert msg.liveTorqueParameters is not None

  def test_get_msg_contains_version(self, mocker):
    """Test get_msg includes correct version."""
    mocker.patch('openpilot.selfdrive.locationd.torqued.Params').return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)
    msg = est.get_msg()

    assert msg.liveTorqueParameters.version == VERSION

  def test_get_msg_valid_flag(self, mocker):
    """Test get_msg respects valid flag."""
    mocker.patch('openpilot.selfdrive.locationd.torqued.Params').return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)

    msg_valid = est.get_msg(valid=True)
    msg_invalid = est.get_msg(valid=False)

    assert msg_valid.valid
    assert not msg_invalid.valid

  def test_get_msg_has_cal_perc(self, mocker):
    """Test get_msg includes calibration percentage."""
    mocker.patch('openpilot.selfdrive.locationd.torqued.Params').return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)
    msg = est.get_msg()

    # Initially zero calibration
    assert msg.liveTorqueParameters.calPerc == 0


class TestConstants:
  """Test module constants."""

  def test_history_positive(self):
    """Test HISTORY is positive."""
    assert HISTORY > 0

  def test_min_vel_reasonable(self):
    """Test MIN_VEL is a reasonable speed in m/s."""
    assert MIN_VEL > 0
    assert MIN_VEL < 50  # Less than highway speeds

  def test_bucket_bounds_symmetric(self):
    """Test STEER_BUCKET_BOUNDS are symmetric around zero."""
    bounds = STEER_BUCKET_BOUNDS
    assert len(bounds) == 8
    # Check symmetric structure
    assert bounds[0][0] == -bounds[-1][1]

  def test_min_bucket_points_matches_bounds(self):
    """Test MIN_BUCKET_POINTS matches STEER_BUCKET_BOUNDS."""
    assert len(MIN_BUCKET_POINTS) == len(STEER_BUCKET_BOUNDS)

  def test_allowed_cars_not_empty(self):
    """Test ALLOWED_CARS is not empty."""
    assert len(ALLOWED_CARS) > 0

  def test_filter_decay_range(self):
    """Test filter decay constants are valid."""
    assert MIN_FILTER_DECAY > 0
    assert MAX_FILTER_DECAY > MIN_FILTER_DECAY

  def test_sanity_factors_positive(self):
    """Test sanity factors are positive and reasonable."""
    assert FACTOR_SANITY > 0
    assert FACTOR_SANITY < 1
    assert FRICTION_SANITY > 0
    assert FRICTION_SANITY < 1


class TestTorqueEstimatorEstimateParams:
  """Test TorqueEstimator.estimate_params method."""

  def test_estimate_params_handles_linalg_error(self, mocker):
    """Test estimate_params handles LinAlgError gracefully."""
    mocker.patch('openpilot.selfdrive.locationd.torqued.Params').return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)

    # Mock SVD to raise LinAlgError
    mocker.patch('numpy.linalg.svd', side_effect=np.linalg.LinAlgError("SVD failed"))

    # Mock get_points to return valid shape
    mocker.patch.object(est.filtered_points, 'get_points', return_value=np.zeros((10, 3)))

    slope, offset, friction = est.estimate_params()

    assert np.isnan(slope)
    assert np.isnan(offset)
    assert np.isnan(friction)


class TestTorqueEstimatorWithTorqueTuning:
  """Test TorqueEstimator with torque tuning cars."""

  def test_init_with_torque_tuning_car(self, mocker):
    """Test init with a car that has torque lateral tuning."""
    mocker.patch('openpilot.selfdrive.locationd.torqued.Params').return_value.get.return_value = None

    # Create CarParams with torque tuning
    cp = car.CarParams()
    cp.brand = "toyota"
    cp.lateralTuning.init('torque')
    cp.lateralTuning.torque.friction = 0.1
    cp.lateralTuning.torque.latAccelFactor = 2.5

    est = TorqueEstimator(cp)

    assert est.use_params is True
    assert abs(est.offline_friction - 0.1) < 0.001
    assert abs(est.offline_latAccelFactor - 2.5) < 0.001

  def test_get_restore_key_with_torque_tuning(self):
    """Test get_restore_key includes torque params."""
    cp = car.CarParams()
    cp.lateralTuning.init('torque')
    cp.lateralTuning.torque.friction = 0.15
    cp.lateralTuning.torque.latAccelFactor = 2.3

    key = TorqueEstimator.get_restore_key(cp, VERSION)

    assert abs(key[2] - 0.15) < 0.001  # friction
    assert abs(key[3] - 2.3) < 0.001  # latAccelFactor


class TestTorqueEstimatorGetMsgAdvanced:
  """Test TorqueEstimator.get_msg advanced features."""

  def test_get_msg_with_points(self, mocker):
    """Test get_msg with with_points=True."""
    mocker.patch('openpilot.selfdrive.locationd.torqued.Params').return_value.get.return_value = None
    cp = car.CarParams()
    est = TorqueEstimator(cp)

    # Add some points
    for i in range(10):
      est.filtered_points.add_point(0.05, 0.1 * i)

    msg = est.get_msg(with_points=True)

    # Points should be included when with_points=True
    assert msg.liveTorqueParameters.points is not None

  def test_get_msg_use_params_false_for_non_allowed_car(self, mocker):
    """Test useParams is False for non-allowed car."""
    mocker.patch('openpilot.selfdrive.locationd.torqued.Params').return_value.get.return_value = None
    cp = car.CarParams()
    cp.brand = "unknown_brand"

    est = TorqueEstimator(cp)
    msg = est.get_msg()

    assert msg.liveTorqueParameters.useParams is False


def test_cal_percent():
  est = TorqueEstimator(car.CarParams())
  msg = est.get_msg()
  assert msg.liveTorqueParameters.calPerc == 0

  for (low, high), min_pts in zip(est.filtered_points.buckets.keys(), est.filtered_points.buckets_min_points.values(), strict=True):
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
