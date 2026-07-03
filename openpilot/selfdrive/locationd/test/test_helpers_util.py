"""Tests for selfdrive/locationd/helpers.py - utility functions and classes."""

import numpy as np

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
    """Test small values return themselves."""
    for n in range(1, 7):
      result = fft_next_good_size(n)
      assert result == n

  def test_power_of_two(self):
    """Test powers of two."""
    assert fft_next_good_size(8) == 8
    assert fft_next_good_size(16) == 16

  def test_composite_of_small_primes(self):
    """Test composites of small primes."""
    # 12 = 2^2 * 3
    assert fft_next_good_size(12) == 12
    # 15 = 3 * 5
    assert fft_next_good_size(15) == 15

  def test_finds_next_good_size(self):
    """Test finds next good size for non-composite."""
    # 13 is prime, next good size should be >= 13
    result = fft_next_good_size(13)
    assert result >= 13
    # Should be a composite of 2, 3, 5, 7, 11
    assert result == 14  # 2 * 7

  def test_larger_values(self):
    """Test larger values."""
    result = fft_next_good_size(100)
    assert result >= 100

  def test_cached(self):
    """Test function is cached (same input gives same result)."""
    result1 = fft_next_good_size(50)
    result2 = fft_next_good_size(50)
    assert result1 == result2


class TestParabolicPeakInterp:
  """Test parabolic_peak_interp function."""

  def test_edge_case_zero_index(self):
    """Test returns index 0 when max_index is 0."""
    R = [5.0, 3.0, 1.0]
    result = parabolic_peak_interp(R, 0)
    assert result == 0

  def test_edge_case_last_index(self):
    """Test returns last index when max_index is last."""
    R = [1.0, 3.0, 5.0]
    result = parabolic_peak_interp(R, 2)
    assert result == 2

  def test_symmetric_peak(self):
    """Test symmetric peak returns center."""
    R = [1.0, 5.0, 1.0]
    result = parabolic_peak_interp(R, 1)
    # Symmetric peak should return center index
    assert abs(result - 1.0) < 1e-5

  def test_asymmetric_peak_left(self):
    """Test asymmetric peak interpolates left."""
    R = [3.0, 5.0, 1.0]
    result = parabolic_peak_interp(R, 1)
    # Peak should be left of center
    assert result < 1.0

  def test_asymmetric_peak_right(self):
    """Test asymmetric peak interpolates right."""
    R = [1.0, 5.0, 3.0]
    result = parabolic_peak_interp(R, 1)
    # Peak should be right of center
    assert result > 1.0


class TestRotateCov:
  """Test rotate_cov function."""

  def test_identity_rotation(self):
    """Test identity rotation returns same covariance."""
    cov = np.array([[1.0, 0.0], [0.0, 2.0]])
    rot = np.eye(2)

    result = rotate_cov(rot, cov)

    np.testing.assert_array_almost_equal(result, cov)

  def test_rotation_shape(self):
    """Test rotation preserves shape."""
    cov = np.array([[1.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 3.0]])
    rot = np.eye(3)

    result = rotate_cov(rot, cov)

    assert result.shape == (3, 3)


class TestRotateStd:
  """Test rotate_std function."""

  def test_identity_rotation(self):
    """Test identity rotation returns same std."""
    std = np.array([1.0, 2.0, 3.0])
    rot = np.eye(3)

    result = rotate_std(rot, std)

    np.testing.assert_array_almost_equal(result, std)

  def test_output_shape(self):
    """Test output shape matches input."""
    std = np.array([1.0, 2.0])
    rot = np.eye(2)

    result = rotate_std(rot, std)

    assert result.shape == (2,)


class TestNPQueue:
  """Test NPQueue class."""

  def test_init_empty(self):
    """Test queue initializes empty."""
    q = NPQueue(maxlen=10, rowsize=3)
    assert len(q) == 0

  def test_append_single(self):
    """Test appending single point."""
    q = NPQueue(maxlen=10, rowsize=3)
    q.append([1.0, 2.0, 3.0])
    assert len(q) == 1

  def test_append_multiple(self):
    """Test appending multiple points."""
    q = NPQueue(maxlen=10, rowsize=3)
    q.append([1.0, 2.0, 3.0])
    q.append([4.0, 5.0, 6.0])
    assert len(q) == 2

  def test_append_beyond_maxlen(self):
    """Test appending beyond maxlen rolls buffer."""
    q = NPQueue(maxlen=2, rowsize=2)
    q.append([1.0, 2.0])
    q.append([3.0, 4.0])
    q.append([5.0, 6.0])

    assert len(q) == 2
    # Should keep last 2 items
    np.testing.assert_array_almost_equal(q.arr[0], [3.0, 4.0])
    np.testing.assert_array_almost_equal(q.arr[1], [5.0, 6.0])

  def test_arr_shape(self):
    """Test arr has correct shape."""
    q = NPQueue(maxlen=10, rowsize=5)
    q.append([1.0, 2.0, 3.0, 4.0, 5.0])
    q.append([6.0, 7.0, 8.0, 9.0, 10.0])

    assert q.arr.shape == (2, 5)


class TestPointBuckets:
  """Test PointBuckets class."""

  def _create_buckets(self):
    """Create a test PointBuckets instance."""
    x_bounds = [(0.0, 5.0), (5.0, 10.0)]
    min_points = [2, 2]
    return PointBuckets(x_bounds=x_bounds, min_points=min_points, min_points_total=4, points_per_bucket=10, rowsize=2)

  def test_init_empty(self):
    """Test buckets initialize empty."""
    buckets = self._create_buckets()
    assert len(buckets) == 0

  def test_is_valid_empty(self):
    """Test is_valid returns False when empty."""
    buckets = self._create_buckets()
    assert not buckets.is_valid()

  def test_is_calculable_empty(self):
    """Test is_calculable returns False when empty."""
    buckets = self._create_buckets()
    assert not buckets.is_calculable()

  def test_get_valid_percent_empty(self):
    """Test get_valid_percent returns 0 when empty."""
    buckets = self._create_buckets()
    assert buckets.get_valid_percent() == 0

  def test_get_points_empty(self):
    """Test get_points on empty buckets."""

    # Add a point using load_points
    class TestBuckets(PointBuckets):
      def add_point(self, x, y):
        for bounds in self.x_bounds:
          if bounds[0] <= x < bounds[1]:
            self.buckets[bounds].append([x, y])
            break

    x_bounds = [(0.0, 10.0)]
    min_points = [1]
    test_buckets = TestBuckets(x_bounds=x_bounds, min_points=min_points, min_points_total=1, points_per_bucket=10, rowsize=2)
    test_buckets.add_point(5.0, 1.0)
    points = test_buckets.get_points()
    assert len(points) == 1

  def test_load_points(self):
    """Test load_points adds multiple points."""

    class TestBuckets(PointBuckets):
      def add_point(self, x, y):
        for bounds in self.x_bounds:
          if bounds[0] <= x < bounds[1]:
            self.buckets[bounds].append([x, y])
            break

    x_bounds = [(0.0, 10.0)]
    min_points = [1]
    test_buckets = TestBuckets(x_bounds=x_bounds, min_points=min_points, min_points_total=1, points_per_bucket=10, rowsize=2)

    # Use load_points to add multiple points at once
    points_to_load = [[1.0, 0.5], [5.0, 1.5], [8.0, 2.5]]
    test_buckets.load_points(points_to_load)

    points = test_buckets.get_points()
    assert len(points) == 3

  def test_get_points_with_num_points(self):
    """Test get_points returns subset when num_points is provided."""

    class TestBuckets(PointBuckets):
      def add_point(self, x, y):
        for bounds in self.x_bounds:
          if bounds[0] <= x < bounds[1]:
            self.buckets[bounds].append([x, y])
            break

    x_bounds = [(0.0, 10.0)]
    min_points = [1]
    test_buckets = TestBuckets(x_bounds=x_bounds, min_points=min_points, min_points_total=1, points_per_bucket=20, rowsize=2)

    # Add many points
    for i in range(10):
      test_buckets.add_point(float(i), float(i) * 2)

    # Get a subset of points
    subset = test_buckets.get_points(num_points=5)
    assert len(subset) == 5

    # When num_points > total, return all
    all_points = test_buckets.get_points(num_points=20)
    assert len(all_points) == 10


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

  def test_roll_pitch_yaw_aliases(self):
    """Test roll, pitch, yaw are aliases for x, y, z."""
    m = Measurement(np.array([0.1, 0.2, 0.3]), np.array([0.01, 0.02, 0.03]))

    assert m.roll == m.x
    assert m.pitch == m.y
    assert m.yaw == m.z

  def test_from_measurement_xyz(self, mocker):
    """Test from_measurement_xyz class method."""
    mock_msg = mocker.MagicMock()
    mock_msg.x = 1.0
    mock_msg.y = 2.0
    mock_msg.z = 3.0
    mock_msg.xStd = 0.1
    mock_msg.yStd = 0.2
    mock_msg.zStd = 0.3

    m = Measurement.from_measurement_xyz(mock_msg)

    assert m.x == 1.0
    assert m.y == 2.0
    assert m.z == 3.0
    assert m.x_std == 0.1
    assert m.y_std == 0.2
    assert m.z_std == 0.3


class TestPose:
  """Test Pose class."""

  def test_init(self):
    """Test Pose initialization."""
    orientation = Measurement(np.array([0.1, 0.2, 0.3]), np.array([0.01, 0.01, 0.01]))
    velocity = Measurement(np.array([1.0, 0.0, 0.0]), np.array([0.1, 0.1, 0.1]))
    acceleration = Measurement(np.array([0.0, 0.0, 9.8]), np.array([0.1, 0.1, 0.1]))
    angular_velocity = Measurement(np.array([0.0, 0.0, 0.1]), np.array([0.01, 0.01, 0.01]))

    pose = Pose(orientation, velocity, acceleration, angular_velocity)

    assert pose.orientation == orientation
    assert pose.velocity == velocity
    assert pose.acceleration == acceleration
    assert pose.angular_velocity == angular_velocity

  def test_from_live_pose(self, mocker):
    """Test Pose.from_live_pose class method."""
    # Create mock XYZMeasurement objects
    def make_mock_xyz(x, y, z, x_std, y_std, z_std):
      m = mocker.MagicMock()
      m.x = x
      m.y = y
      m.z = z
      m.xStd = x_std
      m.yStd = y_std
      m.zStd = z_std
      return m

    mock_live_pose = mocker.MagicMock()
    mock_live_pose.orientationNED = make_mock_xyz(0.1, 0.2, 0.3, 0.01, 0.02, 0.03)
    mock_live_pose.velocityDevice = make_mock_xyz(10.0, 0.5, 0.1, 0.1, 0.1, 0.1)
    mock_live_pose.accelerationDevice = make_mock_xyz(0.0, 0.0, 9.8, 0.1, 0.1, 0.1)
    mock_live_pose.angularVelocityDevice = make_mock_xyz(0.0, 0.0, 0.05, 0.01, 0.01, 0.01)

    pose = Pose.from_live_pose(mock_live_pose)

    assert isinstance(pose, Pose)
    assert pose.orientation.x == 0.1
    assert pose.velocity.x == 10.0
    assert pose.acceleration.z == 9.8
    assert pose.angular_velocity.z == 0.05


class TestPoseCalibrator:
  """Test PoseCalibrator class."""

  def test_init(self):
    """Test PoseCalibrator initialization."""
    pc = PoseCalibrator()

    assert not pc.calib_valid
    np.testing.assert_array_almost_equal(pc.calib_from_device, np.eye(3))

  def test_build_calibrated_pose_identity(self):
    """Test build_calibrated_pose with identity calibration."""
    pc = PoseCalibrator()

    orientation = Measurement(np.array([0.0, 0.0, 0.0]), np.array([0.01, 0.01, 0.01]))
    velocity = Measurement(np.array([1.0, 0.0, 0.0]), np.array([0.1, 0.1, 0.1]))
    acceleration = Measurement(np.array([0.0, 0.0, 9.8]), np.array([0.1, 0.1, 0.1]))
    angular_velocity = Measurement(np.array([0.0, 0.0, 0.1]), np.array([0.01, 0.01, 0.01]))

    pose = Pose(orientation, velocity, acceleration, angular_velocity)
    calibrated = pc.build_calibrated_pose(pose)

    # With identity calibration, velocities should be unchanged
    np.testing.assert_array_almost_equal(calibrated.velocity.xyz, velocity.xyz)
