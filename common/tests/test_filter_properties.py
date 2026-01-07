"""Property-based tests for common/filter_simple.py using Hypothesis.

These tests verify mathematical properties that should hold for all valid inputs,
complementing the unit tests with broader coverage.
"""

import math

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from openpilot.common.filter_simple import FirstOrderFilter, BounceFilter

HYPOTHESIS_SETTINGS = settings(max_examples=100, deadline=None)


class TestFirstOrderFilterProperties:
  """Property-based tests for FirstOrderFilter."""

  @given(
    x0=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    rc=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    dt=st.floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False),
    new_x=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_output_between_old_and_new(self, x0, rc, dt, new_x):
    """Property: Output is always between previous and new value."""
    f = FirstOrderFilter(x0, rc, dt)

    result = f.update(new_x)

    lower = min(x0, new_x)
    upper = max(x0, new_x)
    # Allow small epsilon for floating point
    assert lower - 1e-10 <= result <= upper + 1e-10

  @given(
    x0=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    rc=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    dt=st.floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_same_value_unchanged(self, x0, rc, dt):
    """Property: Updating with same value returns that value."""
    f = FirstOrderFilter(x0, rc, dt)

    result = f.update(x0)

    assert abs(result - x0) < 1e-10

  @given(
    x0=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    rc=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    dt=st.floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_alpha_in_valid_range(self, x0, rc, dt):
    """Property: Alpha is always between 0 and 1."""
    f = FirstOrderFilter(x0, rc, dt)

    assert 0 < f.alpha < 1

  @given(
    x0=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    dt=st.floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False),
    new_x=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_small_rc_follows_input(self, x0, dt, new_x):
    """Property: Very small RC follows input closely."""
    f = FirstOrderFilter(x0, rc=0.001, dt=dt)

    result = f.update(new_x)

    # With small RC, alpha is close to 1, so result should be close to new_x
    assert abs(result - new_x) < abs(x0 - new_x) * 0.5 + 1e-5

  @given(
    x0=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    dt=st.floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False),
    new_x=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_large_rc_smooths_heavily(self, x0, dt, new_x):
    """Property: Large RC heavily smooths the input."""
    assume(abs(x0 - new_x) > 1.0)  # Need meaningful difference

    f = FirstOrderFilter(x0, rc=100.0, dt=dt)

    result = f.update(new_x)

    # With large RC, result should be close to x0
    assert abs(result - x0) < abs(new_x - x0)

  @given(
    x0=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    rc=st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
    dt=st.floats(min_value=0.01, max_value=0.1, allow_nan=False, allow_infinity=False),
    target=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_converges_to_constant_input(self, x0, rc, dt, target):
    """Property: Filter converges to constant input over time."""
    assume(abs(x0 - target) > 0.1)  # Need meaningful difference

    f = FirstOrderFilter(x0, rc, dt)

    # Run many updates with same target
    for _ in range(500):
      result = f.update(target)

    # Should converge close to target (tolerance scales with initial difference)
    max_error = abs(x0 - target) * 0.01  # 1% of initial error
    assert abs(result - target) < max(max_error, 0.5)

  @given(
    x0=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    rc=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    dt=st.floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_uninitialized_takes_first_value(self, x0, rc, dt):
    """Property: Uninitialized filter takes first update value directly."""
    f = FirstOrderFilter(x0, rc, dt, initialized=False)

    new_x = x0 + 50.0  # Different value
    result = f.update(new_x)

    assert result == new_x
    assert f.initialized


class TestFirstOrderFilterUpdateAlpha:
  """Property tests for alpha updates."""

  @given(
    x0=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    rc1=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
    rc2=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
    dt=st.floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_update_alpha_changes_behavior(self, x0, rc1, rc2, dt):
    """Property: Changing RC changes filter behavior."""
    assume(abs(rc1 - rc2) > 0.5)  # Need meaningful difference

    f = FirstOrderFilter(x0, rc1, dt)
    alpha1 = f.alpha

    f.update_alpha(rc2)
    alpha2 = f.alpha

    # Different RC should give different alpha
    assert alpha1 != alpha2


class TestBounceFilterProperties:
  """Property-based tests for BounceFilter."""

  @given(
    x0=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    rc=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
    dt=st.floats(min_value=0.01, max_value=0.05, allow_nan=False, allow_infinity=False),
    new_x=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_output_finite(self, x0, rc, dt, new_x):
    """Property: Output is always finite."""
    f = BounceFilter(x0, rc, dt)

    for _ in range(10):
      result = f.update(new_x)
      assert math.isfinite(result)

  @given(
    x0=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    rc=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
    dt=st.floats(min_value=0.01, max_value=0.05, allow_nan=False, allow_infinity=False),
    target=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_converges_eventually(self, x0, rc, dt, target):
    """Property: BounceFilter eventually settles near target."""
    assume(abs(x0 - target) > 1.0)  # Need meaningful difference

    f = BounceFilter(x0, rc, dt)

    # Run many updates
    for _ in range(500):
      result = f.update(target)

    # Should settle near target (with some bounce tolerance)
    assert abs(result - target) < abs(x0 - target) * 0.5

  @given(
    x0=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    rc=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
    dt=st.floats(min_value=0.01, max_value=0.05, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_velocity_dampens(self, x0, rc, dt):
    """Property: Velocity filter dampens over time with no input change."""
    f = BounceFilter(x0, rc, dt)

    # Initial update to create some velocity
    f.update(x0 + 10.0)

    # Keep updating with same value - velocity should dampen
    for _ in range(100):
      f.update(x0 + 10.0)

    # Velocity should be very small or zero
    assert abs(f.velocity.x) < 1.0
