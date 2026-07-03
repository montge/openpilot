"""Tests for locationd/helpers.py - location helper classes and functions."""

import numpy as np
import pytest

from openpilot.selfdrive.locationd.helpers import (
  fft_next_good_size,
  parabolic_peak_interp,
  rotate_cov,
  rotate_std,
  NPQueue,
  PointBuckets,
  Measurement,
  Pose,
  PoseCalibrator,
)


class TestFftNextGoodSize:
  """Test fft_next_good_size function."""

  def test_small_values(self):
    """Test small input values are returned as-is."""
    for n in range(1, 7):
      result = fft_next_good_size(n)
      assert result >= n
      assert result <= 6

  def test_returns_at_least_n(self):
    """Test result is always >= n."""
    for n in [7, 10, 100, 1000]:
      result = fft_next_good_size(n)
      assert result >= n

  def test_returns_composite(self):
    """Test result is composite of 2, 3, 5, 7, 11."""
    result = fft_next_good_size(100)
    # Result should be factorizable by 2, 3, 5, 7, 11 only
    n = result
    for prime in [2, 3, 5, 7, 11]:
      while n % prime == 0:
        n //= prime
    assert n == 1

  def test_cached(self):
    """Test function is cached."""
    # Call twice, should return same result
    result1 = fft_next_good_size(123)
    result2 = fft_next_good_size(123)
    assert result1 == result2


class TestParabolicPeakInterp:
  """Test parabolic_peak_interp function."""

  def test_boundary_index_zero(self):
    """Test returns index 0 when max_index is 0."""
    R = np.array([1.0, 0.5, 0.2])
    result = parabolic_peak_interp(R, 0)
    assert result == 0

  def test_boundary_index_end(self):
    """Test returns last index when max_index is at end."""
    R = np.array([0.2, 0.5, 1.0])
    result = parabolic_peak_interp(R, 2)
    assert result == 2

  def test_symmetric_peak(self):
    """Test symmetric peak returns center index."""
    R = np.array([0.5, 1.0, 0.5])
    result = parabolic_peak_interp(R, 1)
    assert abs(result - 1.0) < 1e-5

  def test_asymmetric_peak_right(self):
    """Test asymmetric peak shifts right."""
    R = np.array([0.3, 1.0, 0.8])
    result = parabolic_peak_interp(R, 1)
    assert result > 1.0

  def test_asymmetric_peak_left(self):
    """Test asymmetric peak shifts left."""
    R = np.array([0.8, 1.0, 0.3])
    result = parabolic_peak_interp(R, 1)
    assert result < 1.0


class TestRotateCov:
  """Test rotate_cov function."""

  def test_identity_rotation(self):
    """Test identity rotation leaves covariance unchanged."""
    rot = np.eye(3)
    cov = np.array([[1, 0.1, 0], [0.1, 2, 0.2], [0, 0.2, 3]])
    result = rotate_cov(rot, cov)
    np.testing.assert_array_almost_equal(result, cov)

  def test_rotation_preserves_trace(self):
    """Test rotation preserves trace of covariance."""
    # 90 degree rotation around z
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    cov = np.diag([1.0, 2.0, 3.0])
    result = rotate_cov(rot, cov)
    assert abs(np.trace(result) - np.trace(cov)) < 1e-10


class TestRotateStd:
  """Test rotate_std function."""

  def test_identity_rotation(self):
    """Test identity rotation leaves std unchanged."""
    rot = np.eye(3)
    std = np.array([1.0, 2.0, 3.0])
    result = rotate_std(rot, std)
    np.testing.assert_array_almost_equal(result, std)

  def test_returns_positive(self):
    """Test result is always positive."""
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    std = np.array([1.0, 2.0, 3.0])
    result = rotate_std(rot, std)
    assert np.all(result >= 0)


class TestNPQueue:
  """Test NPQueue class."""

  def test_init_empty(self):
    """Test NPQueue starts empty."""
    q = NPQueue(maxlen=10, rowsize=3)
    assert len(q) == 0

  def test_append_single(self):
    """Test appending a single element."""
    q = NPQueue(maxlen=10, rowsize=3)
    q.append([1.0, 2.0, 3.0])
    assert len(q) == 1

  def test_append_multiple(self):
    """Test appending multiple elements."""
    q = NPQueue(maxlen=10, rowsize=3)
    for i in range(5):
      q.append([float(i), float(i), float(i)])
    assert len(q) == 5

  def test_append_exceeds_maxlen(self):
    """Test appending beyond maxlen drops oldest."""
    q = NPQueue(maxlen=3, rowsize=2)
    for i in range(5):
      q.append([float(i), float(i)])
    assert len(q) == 3
    # First element should now be [2, 2] (indices 2, 3, 4 remain)
    np.testing.assert_array_almost_equal(q.arr[0], [2.0, 2.0])

  def test_fifo_order(self):
    """Test FIFO order is maintained."""
    q = NPQueue(maxlen=5, rowsize=1)
    for i in range(5):
      q.append([float(i)])
    np.testing.assert_array_almost_equal(q.arr[:, 0], [0, 1, 2, 3, 4])


class TestPointBuckets:
  """Test PointBuckets class."""

  def _create_buckets(self):
    """Create a simple PointBuckets instance."""
    x_bounds = [(-1.0, 0.0), (0.0, 1.0)]
    min_points = [2, 2]
    return PointBuckets(x_bounds=x_bounds, min_points=min_points, min_points_total=3, points_per_bucket=10, rowsize=2)

  def test_init_empty(self):
    """Test PointBuckets starts empty."""
    buckets = self._create_buckets()
    assert len(buckets) == 0

  def test_is_valid_false_initially(self):
    """Test is_valid returns False when empty."""
    buckets = self._create_buckets()
    assert not buckets.is_valid()

  def test_is_calculable_false_initially(self):
    """Test is_calculable returns False when empty."""
    buckets = self._create_buckets()
    assert not buckets.is_calculable()

  def test_get_valid_percent_zero_initially(self):
    """Test get_valid_percent returns 0 when empty."""
    buckets = self._create_buckets()
    assert buckets.get_valid_percent() == 0

  def test_add_point_not_implemented(self):
    """Test add_point raises NotImplementedError in base class."""
    buckets = self._create_buckets()
    with pytest.raises(NotImplementedError):
      buckets.add_point(0.5, 0.5)


class TestPointBucketsGetPoints:
  """Test PointBuckets get_points and load_points methods."""

  def _create_concrete_buckets(self):
    """Create a concrete PointBuckets subclass for testing."""

    class TestBuckets(PointBuckets):
      def add_point(self, x, y):
        for bound_min, bound_max in self.x_bounds:
          if bound_min <= x < bound_max:
            self.buckets[(bound_min, bound_max)].append([x, y])
            break

    x_bounds = [(-1.0, 0.0), (0.0, 1.0)]
    min_points = [2, 2]
    return TestBuckets(x_bounds=x_bounds, min_points=min_points, min_points_total=3, points_per_bucket=10, rowsize=2)

  def test_get_points_returns_all(self):
    """Test get_points returns all points when num_points is None."""
    buckets = self._create_concrete_buckets()
    buckets.add_point(-0.5, 1.0)
    buckets.add_point(-0.3, 2.0)
    buckets.add_point(0.5, 3.0)

    points = buckets.get_points()

    assert points.shape[0] == 3
    assert points.shape[1] == 2

  def test_get_points_with_limit(self):
    """Test get_points with num_points limit."""
    buckets = self._create_concrete_buckets()
    for i in range(5):
      buckets.add_point(-0.5, float(i))

    points = buckets.get_points(num_points=3)

    assert points.shape[0] == 3

  def test_load_points(self):
    """Test load_points loads multiple points."""
    buckets = self._create_concrete_buckets()
    points = [[-0.5, 1.0], [-0.3, 2.0], [0.5, 3.0]]

    buckets.load_points(points)

    assert len(buckets) == 3


class TestMeasurement:
  """Test Measurement class."""

  def test_init(self):
    """Test Measurement initialization."""
    xyz = np.array([1.0, 2.0, 3.0])
    xyz_std = np.array([0.1, 0.2, 0.3])
    m = Measurement(xyz, xyz_std)
    np.testing.assert_array_equal(m.xyz, xyz)
    np.testing.assert_array_equal(m.xyz_std, xyz_std)

  def test_xyz_properties(self):
    """Test x, y, z properties."""
    m = Measurement(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
    assert m.x == 1.0
    assert m.y == 2.0
    assert m.z == 3.0

  def test_std_properties(self):
    """Test x_std, y_std, z_std properties."""
    m = Measurement(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
    assert m.x_std == 0.1
    assert m.y_std == 0.2
    assert m.z_std == 0.3

  def test_rpy_aliases(self):
    """Test roll, pitch, yaw are aliases for x, y, z."""
    m = Measurement(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
    assert m.roll == m.x
    assert m.pitch == m.y
    assert m.yaw == m.z

  def test_from_measurement_xyz(self, mocker):
    """Test from_measurement_xyz class method."""
    mock_measurement = mocker.MagicMock()
    mock_measurement.x = 1.0
    mock_measurement.y = 2.0
    mock_measurement.z = 3.0
    mock_measurement.xStd = 0.1
    mock_measurement.yStd = 0.2
    mock_measurement.zStd = 0.3

    result = Measurement.from_measurement_xyz(mock_measurement)

    np.testing.assert_array_almost_equal(result.xyz, [1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(result.xyz_std, [0.1, 0.2, 0.3])


class TestPose:
  """Test Pose class."""

  def test_init(self):
    """Test Pose initialization."""
    orientation = Measurement(np.array([0.1, 0.2, 0.3]), np.array([0.01, 0.02, 0.03]))
    velocity = Measurement(np.array([10.0, 0.0, 0.0]), np.array([0.1, 0.1, 0.1]))
    acceleration = Measurement(np.array([1.0, 0.0, 0.0]), np.array([0.1, 0.1, 0.1]))
    angular_velocity = Measurement(np.array([0.0, 0.0, 0.1]), np.array([0.01, 0.01, 0.01]))

    pose = Pose(orientation, velocity, acceleration, angular_velocity)

    assert pose.orientation is orientation
    assert pose.velocity is velocity
    assert pose.acceleration is acceleration
    assert pose.angular_velocity is angular_velocity

  def test_from_live_pose(self, mocker):
    """Test from_live_pose class method."""

    # Create mock XYZMeasurement
    def mock_xyz(x, y, z, xstd, ystd, zstd):
      m = mocker.MagicMock()
      m.x, m.y, m.z = x, y, z
      m.xStd, m.yStd, m.zStd = xstd, ystd, zstd
      return m

    mock_live_pose = mocker.MagicMock()
    mock_live_pose.orientationNED = mock_xyz(0.1, 0.2, 0.3, 0.01, 0.02, 0.03)
    mock_live_pose.velocityDevice = mock_xyz(10.0, 0.0, 0.0, 0.1, 0.1, 0.1)
    mock_live_pose.accelerationDevice = mock_xyz(1.0, 0.0, 0.0, 0.1, 0.1, 0.1)
    mock_live_pose.angularVelocityDevice = mock_xyz(0.0, 0.0, 0.1, 0.01, 0.01, 0.01)

    result = Pose.from_live_pose(mock_live_pose)

    assert isinstance(result, Pose)
    np.testing.assert_array_almost_equal(result.orientation.xyz, [0.1, 0.2, 0.3])
    np.testing.assert_array_almost_equal(result.velocity.xyz, [10.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(result.acceleration.xyz, [1.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(result.angular_velocity.xyz, [0.0, 0.0, 0.1])


class TestPoseCalibrator:
  """Test PoseCalibrator class."""

  def test_init(self):
    """Test PoseCalibrator initialization."""
    pc = PoseCalibrator()
    assert not pc.calib_valid
    np.testing.assert_array_equal(pc.calib_from_device, np.eye(3))

  def test_transform_with_identity(self):
    """Test transform with identity calibration."""
    pc = PoseCalibrator()
    m = Measurement(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
    result = pc._transform_calib_from_device(m)
    np.testing.assert_array_almost_equal(result.xyz, m.xyz)

  def test_build_calibrated_pose(self):
    """Test build_calibrated_pose returns Pose."""
    pc = PoseCalibrator()
    orientation = Measurement(np.array([0.0, 0.0, 0.0]), np.array([0.01, 0.01, 0.01]))
    velocity = Measurement(np.array([10.0, 0.0, 0.0]), np.array([0.1, 0.1, 0.1]))
    acceleration = Measurement(np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.1, 0.1]))
    angular_velocity = Measurement(np.array([0.0, 0.0, 0.0]), np.array([0.01, 0.01, 0.01]))
    pose = Pose(orientation, velocity, acceleration, angular_velocity)

    result = pc.build_calibrated_pose(pose)

    assert isinstance(result, Pose)

  def test_feed_live_calib(self, mocker):
    """Test feed_live_calib updates calibration."""
    pc = PoseCalibrator()

    # Create mock live calibration
    live_calib = mocker.MagicMock()
    live_calib.rpyCalib = [0.0, 0.0, 0.0]
    live_calib.calStatus = 1  # calibrated

    pc.feed_live_calib(live_calib)

    # After feeding identity calibration, calib_from_device should still be ~identity
    np.testing.assert_array_almost_equal(pc.calib_from_device, np.eye(3))
