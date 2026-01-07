"""Property-based tests for locationd/helpers.py using Hypothesis.

These tests verify mathematical properties that should hold for all valid inputs,
complementing the unit tests with broader coverage.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from openpilot.selfdrive.locationd.helpers import (
  fft_next_good_size,
  parabolic_peak_interp,
  rotate_cov,
  rotate_std,
  NPQueue,
  Measurement,
)

HYPOTHESIS_SETTINGS = settings(max_examples=100, deadline=None)


class TestFftNextGoodSizeProperties:
  """Property-based tests for fft_next_good_size."""

  @given(n=st.integers(min_value=1, max_value=10000))
  @HYPOTHESIS_SETTINGS
  def test_result_at_least_n(self, n):
    """Property: Result is always >= n."""
    result = fft_next_good_size(n)
    assert result >= n

  @given(n=st.integers(min_value=1, max_value=1000))
  @HYPOTHESIS_SETTINGS
  def test_result_is_good_composite(self, n):
    """Property: Result is factorizable by only 2, 3, 5, 7, 11."""
    result = fft_next_good_size(n)
    remaining = result
    for prime in [2, 3, 5, 7, 11]:
      while remaining % prime == 0:
        remaining //= prime
    assert remaining == 1, f"{result} has prime factor > 11"

  @given(n=st.integers(min_value=1, max_value=10000))
  @HYPOTHESIS_SETTINGS
  def test_deterministic(self, n):
    """Property: Function is deterministic (same input gives same output)."""
    result1 = fft_next_good_size(n)
    result2 = fft_next_good_size(n)
    assert result1 == result2

  @given(n=st.integers(min_value=1, max_value=1000))
  @HYPOTHESIS_SETTINGS
  def test_bounded_above(self, n):
    """Property: Result is at most 2n (reasonable upper bound)."""
    result = fft_next_good_size(n)
    assert result <= 2 * n


class TestParabolicPeakInterpProperties:
  """Property-based tests for parabolic_peak_interp."""

  @given(
    left=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False),
    center=st.floats(min_value=1.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    right=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_result_near_peak(self, left, center, right):
    """Property: Result is within 1 of max_index for valid peaks."""
    R = np.array([left, center, right])
    result = parabolic_peak_interp(R, 1)
    assert 0 <= result <= 2

  @given(value=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False))
  @HYPOTHESIS_SETTINGS
  def test_symmetric_returns_center(self, value):
    """Property: Symmetric peak returns exact center."""
    R = np.array([value * 0.5, value, value * 0.5])
    result = parabolic_peak_interp(R, 1)
    assert abs(result - 1.0) < 1e-10

  @given(
    center=st.floats(min_value=1.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    offset=st.floats(min_value=0.01, max_value=0.4, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_higher_right_shifts_right(self, center, offset):
    """Property: Higher right neighbor shifts interpolated peak right."""
    left = center * 0.5
    right = center * 0.5 + offset
    R = np.array([left, center, right])
    result = parabolic_peak_interp(R, 1)
    assert result > 1.0

  @given(
    center=st.floats(min_value=1.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    offset=st.floats(min_value=0.01, max_value=0.4, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_higher_left_shifts_left(self, center, offset):
    """Property: Higher left neighbor shifts interpolated peak left."""
    left = center * 0.5 + offset
    right = center * 0.5
    R = np.array([left, center, right])
    result = parabolic_peak_interp(R, 1)
    assert result < 1.0


class TestRotateCovProperties:
  """Property-based tests for rotate_cov."""

  @given(
    angle=st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False),
    var1=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    var2=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    var3=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_preserves_trace(self, angle, var1, var2, var3):
    """Property: Rotation preserves trace of covariance matrix."""
    # Create rotation around z-axis
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    cov = np.diag([var1, var2, var3])

    result = rotate_cov(rot, cov)

    assert abs(np.trace(result) - np.trace(cov)) < 1e-10

  @given(
    var1=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    var2=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    var3=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_identity_unchanged(self, var1, var2, var3):
    """Property: Identity rotation leaves covariance unchanged."""
    rot = np.eye(3)
    cov = np.diag([var1, var2, var3])

    result = rotate_cov(rot, cov)

    np.testing.assert_array_almost_equal(result, cov)

  @given(
    angle=st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False),
    var1=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    var2=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    var3=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_positive_semidefinite(self, angle, var1, var2, var3):
    """Property: Rotated covariance remains positive semi-definite."""
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    cov = np.diag([var1, var2, var3])

    result = rotate_cov(rot, cov)

    eigenvalues = np.linalg.eigvalsh(result)
    assert np.all(eigenvalues >= -1e-10), "Eigenvalues must be non-negative"


class TestRotateStdProperties:
  """Property-based tests for rotate_std."""

  @given(
    angle=st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False),
    std1=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    std2=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    std3=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_always_positive(self, angle, std1, std2, std3):
    """Property: Rotated std values are always positive."""
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    std = np.array([std1, std2, std3])

    result = rotate_std(rot, std)

    assert np.all(result >= 0)

  @given(
    std1=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    std2=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    std3=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_identity_unchanged(self, std1, std2, std3):
    """Property: Identity rotation leaves std unchanged."""
    rot = np.eye(3)
    std = np.array([std1, std2, std3])

    result = rotate_std(rot, std)

    np.testing.assert_array_almost_equal(result, std)

  @given(
    angle=st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False),
    std1=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    std2=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    std3=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_finite_output(self, angle, std1, std2, std3):
    """Property: Output is always finite."""
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    std = np.array([std1, std2, std3])

    result = rotate_std(rot, std)

    assert np.all(np.isfinite(result))


class TestNPQueueProperties:
  """Property-based tests for NPQueue."""

  @given(
    maxlen=st.integers(min_value=1, max_value=100),
    rowsize=st.integers(min_value=1, max_value=10),
    num_appends=st.integers(min_value=0, max_value=200),
  )
  @HYPOTHESIS_SETTINGS
  def test_length_bounded(self, maxlen, rowsize, num_appends):
    """Property: Queue length never exceeds maxlen."""
    q = NPQueue(maxlen=maxlen, rowsize=rowsize)

    for i in range(num_appends):
      q.append([float(i)] * rowsize)

    assert len(q) <= maxlen

  @given(
    maxlen=st.integers(min_value=1, max_value=50),
    rowsize=st.integers(min_value=1, max_value=5),
    num_appends=st.integers(min_value=1, max_value=100),
  )
  @HYPOTHESIS_SETTINGS
  def test_correct_length(self, maxlen, rowsize, num_appends):
    """Property: Queue length is min(num_appends, maxlen)."""
    q = NPQueue(maxlen=maxlen, rowsize=rowsize)

    for i in range(num_appends):
      q.append([float(i)] * rowsize)

    expected_len = min(num_appends, maxlen)
    assert len(q) == expected_len

  @given(
    maxlen=st.integers(min_value=2, max_value=20),
    rowsize=st.integers(min_value=1, max_value=5),
  )
  @HYPOTHESIS_SETTINGS
  def test_fifo_order(self, maxlen, rowsize):
    """Property: FIFO order is maintained after filling."""
    q = NPQueue(maxlen=maxlen, rowsize=rowsize)

    # Append more than maxlen to test overflow
    for i in range(maxlen + 5):
      q.append([float(i)] * rowsize)

    # First element should be 5.0 (indices 0-4 dropped)
    assert q.arr[0, 0] == 5.0
    # Last element should be maxlen + 4
    assert q.arr[-1, 0] == float(maxlen + 4)


class TestMeasurementProperties:
  """Property-based tests for Measurement class."""

  @given(
    x=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    z=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    x_std=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    y_std=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    z_std=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_properties_match_array(self, x, y, z, x_std, y_std, z_std):
    """Property: x,y,z properties match xyz array indices."""
    m = Measurement(np.array([x, y, z]), np.array([x_std, y_std, z_std]))

    assert m.x == m.xyz[0]
    assert m.y == m.xyz[1]
    assert m.z == m.xyz[2]
    assert m.x_std == m.xyz_std[0]
    assert m.y_std == m.xyz_std[1]
    assert m.z_std == m.xyz_std[2]

  @given(
    x=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    z=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_rpy_aliases(self, x, y, z):
    """Property: roll,pitch,yaw are aliases for x,y,z."""
    m = Measurement(np.array([x, y, z]), np.array([0.1, 0.1, 0.1]))

    assert m.roll == m.x
    assert m.pitch == m.y
    assert m.yaw == m.z
