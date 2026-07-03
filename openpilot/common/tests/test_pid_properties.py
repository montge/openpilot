"""Property-based tests for common/pid.py using Hypothesis.

These tests verify mathematical properties that should hold for all valid inputs,
complementing the unit tests with broader coverage.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from openpilot.common.pid import PIDController

HYPOTHESIS_SETTINGS = settings(max_examples=100, deadline=None)


class TestPIDControllerProperties:
  """Property-based tests for PIDController."""

  @given(
    k_p=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    k_i=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    error=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    pos_limit=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_output_within_limits(self, k_p, k_i, error, pos_limit):
    """Property: Output is always within configured limits."""
    pid = PIDController(k_p=k_p, k_i=k_i, pos_limit=pos_limit, neg_limit=-pos_limit)

    output = pid.update(error)

    assert -pos_limit <= output <= pos_limit

  @given(
    k_p=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    k_i=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    error=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_reset_zeros_state(self, k_p, k_i, error):
    """Property: Reset zeros all internal state."""
    pid = PIDController(k_p=k_p, k_i=k_i)

    # Run a few updates to build up state
    for _ in range(5):
      pid.update(error)

    pid.reset()

    assert pid.p == 0.0
    assert pid.i == 0.0
    assert pid.d == 0.0
    assert pid.f == 0.0
    assert pid.control == 0

  @given(
    k_p=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    error=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_proportional_term_correct(self, k_p, error):
    """Property: P term equals k_p * error."""
    pid = PIDController(k_p=k_p, k_i=0.0)

    pid.update(error)

    assert abs(pid.p - k_p * error) < 1e-10

  @given(
    k_p=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
    k_i=st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
    error=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_integral_accumulates(self, k_p, k_i, error):
    """Property: I term accumulates with same-sign errors."""
    pid = PIDController(k_p=k_p, k_i=k_i, pos_limit=1000, neg_limit=-1000)

    prev_i = 0.0
    for _ in range(5):
      pid.update(error)
      # Integral should grow when error is positive (within limits)
      if pid.i != prev_i:  # Only check if not saturated
        assert pid.i > prev_i
      prev_i = pid.i

  @given(
    k_p=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
    k_i=st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
    error=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_freeze_integrator_works(self, k_p, k_i, error):
    """Property: Integrator doesn't change when frozen."""
    pid = PIDController(k_p=k_p, k_i=k_i, pos_limit=1000, neg_limit=-1000)

    # First update to set initial state
    pid.update(error)
    i_before = pid.i

    # Update with freeze
    pid.update(error, freeze_integrator=True)

    assert pid.i == i_before

  @given(
    k_p=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
    feedforward=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_feedforward_added(self, k_p, feedforward):
    """Property: Feedforward is added to output."""
    pid = PIDController(k_p=k_p, k_i=0.0, pos_limit=1000, neg_limit=-1000)

    pid.update(error=0.0, feedforward=feedforward)

    assert pid.f == feedforward
    # With zero error and gains, output should be feedforward
    assert abs(pid.control - feedforward) < 1e-10

  @given(
    k_d=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
    error_rate=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_derivative_term_correct(self, k_d, error_rate):
    """Property: D term equals k_d * error_rate."""
    pid = PIDController(k_p=0.0, k_i=0.0, k_d=k_d, pos_limit=1000, neg_limit=-1000)

    pid.update(error=0.0, error_rate=error_rate)

    assert abs(pid.d - k_d * error_rate) < 1e-10

  @given(
    speed=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_speed_interpolation(self, speed):
    """Property: Gain interpolation works with speed-dependent gains."""
    # Speed-dependent kp: low at low speed, high at high speed
    k_p = [[0, 20, 40], [0.5, 1.0, 1.5]]
    pid = PIDController(k_p=k_p, k_i=0.0)

    pid.update(error=1.0, speed=speed)

    # k_p should be interpolated based on speed
    expected_kp = np.interp(speed, k_p[0], k_p[1])
    assert abs(pid.k_p - expected_kp) < 1e-10


class TestPIDControllerAntiWindup:
  """Property tests for anti-windup behavior."""

  @given(
    k_i=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
    pos_limit=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
    error=st.floats(min_value=10.0, max_value=100.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_integrator_anti_windup(self, k_i, pos_limit, error):
    """Property: Integrator doesn't windup beyond limits."""
    pid = PIDController(k_p=0.0, k_i=k_i, pos_limit=pos_limit, neg_limit=-pos_limit)

    # Run many updates with large positive error
    for _ in range(100):
      output = pid.update(error)

    # Output should still be within limits
    assert -pos_limit <= output <= pos_limit
    # Integral term should not grow unboundedly
    assert abs(pid.i) <= pos_limit * 10  # Reasonable bound

  @given(
    k_p=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
    k_i=st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False),
    pos_limit=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_saturated_output_finite(self, k_p, k_i, pos_limit):
    """Property: Output remains finite even with extreme inputs."""
    pid = PIDController(k_p=k_p, k_i=k_i, pos_limit=pos_limit, neg_limit=-pos_limit)

    # Large errors
    for error in [1e6, -1e6, 1e3, -1e3]:
      output = pid.update(error)
      assert np.isfinite(output)
      assert -pos_limit <= output <= pos_limit
