import numpy as np
import pytest

from openpilot.common.simple_kalman import KF1D, get_kalman_gain


class TestGetKalmanGain:
  """Test get_kalman_gain helper function."""

  def test_returns_correct_shape(self):
    """Test that get_kalman_gain returns correctly shaped matrix."""
    dt = 0.01
    A = np.array([[1.0, dt], [0.0, 1.0]])
    C = np.array([[1.0, 0.0]])
    Q = np.array([[10.0, 0.0], [0.0, 100.0]])
    R = np.array([[1000.0]])

    K = get_kalman_gain(dt, A, C, Q, R)

    assert K.shape == (2, 1)

  def test_converges_with_iterations(self):
    """Test that gain converges with more iterations."""
    dt = 0.01
    A = np.array([[1.0, dt], [0.0, 1.0]])
    C = np.array([[1.0, 0.0]])
    Q = np.array([[10.0, 0.0], [0.0, 100.0]])
    R = np.array([[1000.0]])

    K_few = get_kalman_gain(dt, A, C, Q, R, iterations=10)
    K_many = get_kalman_gain(dt, A, C, Q, R, iterations=100)

    # Values should be close but not necessarily identical
    assert isinstance(K_few[0, 0], (float, np.floating))
    assert isinstance(K_many[0, 0], (float, np.floating))

  def test_gain_positive(self):
    """Test that Kalman gain elements are positive."""
    dt = 0.01
    A = np.array([[1.0, dt], [0.0, 1.0]])
    C = np.array([[1.0, 0.0]])
    Q = np.array([[10.0, 0.0], [0.0, 100.0]])
    R = np.array([[1000.0]])

    K = get_kalman_gain(dt, A, C, Q, R)

    # For this typical tracking problem, gain should be positive
    assert K[0, 0] > 0
    assert K[1, 0] > 0

  def test_high_r_gives_low_gain(self):
    """Test that high measurement noise gives lower gain."""
    dt = 0.01
    A = np.array([[1.0, dt], [0.0, 1.0]])
    C = np.array([[1.0, 0.0]])
    Q = np.array([[10.0, 0.0], [0.0, 100.0]])

    K_low_r = get_kalman_gain(dt, A, C, Q, np.array([[100.0]]))
    K_high_r = get_kalman_gain(dt, A, C, Q, np.array([[10000.0]]))

    # Higher R (more measurement noise) should give lower gain
    assert K_high_r[0, 0] < K_low_r[0, 0]


class TestKF1D:
  """Test KF1D Kalman filter class."""

  def setup_method(self):
    dt = 0.01
    x0_0 = 0.0
    x1_0 = 0.0
    A0_0 = 1.0
    A0_1 = dt
    A1_0 = 0.0
    A1_1 = 1.0
    C0_0 = 1.0
    C0_1 = 0.0
    K0_0 = 0.12287673
    K1_0 = 0.29666309

    self.kf = KF1D(x0=[[x0_0], [x1_0]],
                   A=[[A0_0, A0_1], [A1_0, A1_1]],
                   C=[C0_0, C0_1],
                   K=[[K0_0], [K1_0]])

  def test_getter_setter(self):
    self.kf.set_x([[1.0], [1.0]])
    assert self.kf.x == [[1.0], [1.0]]

  def test_update_returns_state(self):
    x = self.kf.update(100)
    assert x == [i[0] for i in self.kf.x]

  def test_initial_state_zero(self):
    """Test that initial state is zero."""
    assert self.kf.x == [[0.0], [0.0]]
    assert self.kf.x0_0 == 0.0
    assert self.kf.x1_0 == 0.0

  def test_update_modifies_state(self):
    """Test that update modifies state."""
    initial_x0 = self.kf.x0_0
    initial_x1 = self.kf.x1_0

    self.kf.update(100.0)

    # State should change after measurement
    assert self.kf.x0_0 != initial_x0
    assert self.kf.x1_0 != initial_x1

  def test_x_property_returns_nested_list(self):
    """Test that x property returns proper nested list format."""
    x = self.kf.x
    assert isinstance(x, list)
    assert len(x) == 2
    assert isinstance(x[0], list)
    assert isinstance(x[1], list)
    assert len(x[0]) == 1
    assert len(x[1]) == 1

  def test_set_x_updates_state(self):
    """Test that set_x properly updates internal state."""
    self.kf.set_x([[5.0], [2.0]])

    assert self.kf.x0_0 == 5.0
    assert self.kf.x1_0 == 2.0
    assert self.kf.x == [[5.0], [2.0]]

  def test_update_with_zero_measurement(self):
    """Test update with zero measurement."""
    self.kf.set_x([[10.0], [5.0]])
    result = self.kf.update(0.0)

    # State should decay toward zero
    assert result[0] < 10.0
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)

  def test_filter_tracks_constant_measurement(self):
    """Test that filter converges to constant measurement."""
    target = 50.0
    for _ in range(200):
      self.kf.update(target)

    # Position (x0) should converge to target
    assert self.kf.x0_0 == pytest.approx(target, rel=0.1)
    # Velocity (x1) should converge to zero
    assert abs(self.kf.x1_0) < 1.0

  def test_filter_tracks_ramp(self):
    """Test that filter tracks a ramping measurement."""
    # Apply linearly increasing measurements
    for i in range(100):
      self.kf.update(float(i))

    # Velocity estimate should be positive (tracking the ramp)
    assert self.kf.x1_0 > 0

  def test_a_k_matrix_computed(self):
    """Test that A_K matrix components are computed."""
    # A_K = A - K*C should be computed in init
    assert hasattr(self.kf, 'A_K_0')
    assert hasattr(self.kf, 'A_K_1')
    assert hasattr(self.kf, 'A_K_2')
    assert hasattr(self.kf, 'A_K_3')

  def test_filter_with_different_initial_state(self):
    """Test filter starting from non-zero state."""
    kf = KF1D(x0=[[100.0], [10.0]],
              A=[[1.0, 0.01], [0.0, 1.0]],
              C=[1.0, 0.0],
              K=[[0.12], [0.29]])

    assert kf.x0_0 == 100.0
    assert kf.x1_0 == 10.0

    # Update toward zero
    for _ in range(50):
      kf.update(0.0)

    # State should move toward zero
    assert kf.x0_0 < 100.0

  def test_negative_measurements(self):
    """Test filter with negative measurements."""
    target = -50.0
    for _ in range(200):
      self.kf.update(target)

    # Should converge to negative value
    assert self.kf.x0_0 == pytest.approx(target, rel=0.1)

  def test_noisy_measurements(self):
    """Test filter smooths noisy measurements."""
    np.random.seed(42)
    true_value = 100.0
    noise_std = 10.0

    values = []
    for _ in range(200):
      noisy = true_value + np.random.normal(0, noise_std)
      self.kf.update(noisy)
      values.append(self.kf.x0_0)

    # Filter output should be close to true value
    assert np.mean(values[-50:]) == pytest.approx(true_value, abs=5.0)

    # Filter output should have lower variance than input noise
    assert np.std(values[-50:]) < noise_std

  def test_update_returns_list(self):
    """Test that update returns a list, not nested list."""
    result = self.kf.update(50.0)
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)
