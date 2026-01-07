"""Property-based tests for drive_helpers using Hypothesis.

These tests verify mathematical properties that should hold for all valid inputs,
complementing the unit tests with broader coverage.
"""

import math

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from openpilot.selfdrive.controls.lib.drive_helpers import (
  clamp,
  smooth_value,
  clip_curvature,
  curv_from_psis,
  MAX_CURVATURE,
)

HYPOTHESIS_SETTINGS = settings(max_examples=100, deadline=None)


class TestClampProperties:
  """Property-based tests for clamp function."""

  @given(
    val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    max_val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_result_always_within_bounds(self, val, min_val, max_val):
    """Property: Result is always between min and max."""
    assume(min_val <= max_val)
    result, _ = clamp(val, min_val, max_val)
    assert min_val <= result <= max_val

  @given(
    val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    max_val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_clamped_flag_correct(self, val, min_val, max_val):
    """Property: Clamped flag correctly indicates if value was modified."""
    assume(min_val <= max_val)
    result, clamped = clamp(val, min_val, max_val)

    if min_val <= val <= max_val:
      assert not clamped
      assert result == val
    else:
      assert clamped

  @given(
    val=st.floats(allow_nan=False, allow_infinity=False),
    min_val=st.floats(allow_nan=False, allow_infinity=False),
    max_val=st.floats(allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_result_is_float(self, val, min_val, max_val):
    """Property: Result is always a float."""
    assume(min_val <= max_val)
    result, _ = clamp(val, min_val, max_val)
    assert isinstance(result, float)

  @given(
    val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    bound=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_idempotent(self, val, bound):
    """Property: Clamping twice gives same result as once."""
    assume(bound >= 0)
    result1, _ = clamp(val, -bound, bound)
    result2, _ = clamp(result1, -bound, bound)
    assert result1 == result2


class TestSmoothValueProperties:
  """Property-based tests for smooth_value function."""

  @given(
    val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    prev_val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    tau=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_result_between_values(self, val, prev_val, tau):
    """Property: Result is between val and prev_val (or equal to one)."""
    result = smooth_value(val, prev_val, tau)
    lower = min(val, prev_val)
    upper = max(val, prev_val)
    # Allow small floating point error
    assert lower - 1e-10 <= result <= upper + 1e-10

  @given(
    val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    tau=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_same_values_unchanged(self, val, tau):
    """Property: Smoothing identical values returns that value."""
    result = smooth_value(val, val, tau)
    assert abs(result - val) < 1e-10

  @given(
    val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    prev_val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_tau_zero_returns_val(self, val, prev_val):
    """Property: tau=0 returns current value."""
    result = smooth_value(val, prev_val, 0.0)
    assert result == val

  @given(
    val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    prev_val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    tau=st.floats(min_value=-100.0, max_value=-0.001, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_negative_tau_returns_val(self, val, prev_val, tau):
    """Property: Negative tau returns current value."""
    result = smooth_value(val, prev_val, tau)
    assert result == val


class TestClipCurvatureProperties:
  """Property-based tests for clip_curvature function."""

  @given(
    v_ego=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
    prev_curvature=st.floats(min_value=-MAX_CURVATURE, max_value=MAX_CURVATURE, allow_nan=False, allow_infinity=False),
    new_curvature=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
    roll=st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_result_within_max_curvature(self, v_ego, prev_curvature, new_curvature, roll):
    """Property: Result curvature never exceeds MAX_CURVATURE."""
    result, _ = clip_curvature(v_ego, prev_curvature, new_curvature, roll)
    assert -MAX_CURVATURE <= result <= MAX_CURVATURE

  @given(
    v_ego=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
    prev_curvature=st.floats(min_value=-MAX_CURVATURE, max_value=MAX_CURVATURE, allow_nan=False, allow_infinity=False),
    new_curvature=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
    roll=st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_result_is_float(self, v_ego, prev_curvature, new_curvature, roll):
    """Property: Result is always a float."""
    result, _ = clip_curvature(v_ego, prev_curvature, new_curvature, roll)
    assert isinstance(result, float)
    assert math.isfinite(result)

  @given(
    v_ego=st.floats(min_value=10.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    curvature=st.floats(min_value=-0.02, max_value=0.02, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_small_curvature_not_jerk_limited(self, v_ego, curvature):
    """Property: Small curvature requests are not jerk-limited."""
    # For small curvatures at reasonable speeds, jerk limiting shouldn't apply
    # when requesting the same curvature (no rate of change)
    result, _ = clip_curvature(v_ego, curvature, curvature, roll=0.0)
    # Result should be close to requested (may differ due to accel limits)
    assert abs(result - curvature) < 0.05

  @given(
    v_ego=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    prev_curvature=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
    new_curvature=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_low_speed_handled(self, v_ego, prev_curvature, new_curvature):
    """Property: Low speeds are handled without errors."""
    # Should not raise, speed is clamped to MIN_SPEED internally
    result, _ = clip_curvature(v_ego, prev_curvature, new_curvature, roll=0.0)
    assert math.isfinite(result)


class TestCurvFromPsisProperties:
  """Property-based tests for curv_from_psis function."""

  @given(
    psi_target=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    psi_rate=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    vego=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
    action_t=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_result_finite(self, psi_target, psi_rate, vego, action_t):
    """Property: Result is always finite."""
    result = curv_from_psis(psi_target, psi_rate, vego, action_t)
    assert math.isfinite(result)

  @given(
    vego=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
    action_t=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_zero_inputs_zero_output(self, vego, action_t):
    """Property: Zero psi_target and psi_rate gives zero curvature."""
    result = curv_from_psis(0.0, 0.0, vego, action_t)
    assert abs(result) < 1e-10

  @given(
    psi_target=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    vego=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    action_t=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_positive_psi_gives_positive_curv(self, psi_target, vego, action_t):
    """Property: Positive psi_target with zero rate gives positive curvature."""
    result = curv_from_psis(psi_target, 0.0, vego, action_t)
    assert result > 0

  @given(
    psi_target=st.floats(min_value=-1.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
    vego=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    action_t=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_negative_psi_gives_negative_curv(self, psi_target, vego, action_t):
    """Property: Negative psi_target with zero rate gives negative curvature."""
    result = curv_from_psis(psi_target, 0.0, vego, action_t)
    assert result < 0

  @given(
    vego=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    psi_target=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
    psi_rate=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
  )
  @HYPOTHESIS_SETTINGS
  def test_low_speed_clamped(self, vego, psi_target, psi_rate):
    """Property: Low speeds are clamped to MIN_SPEED."""
    result = curv_from_psis(psi_target, psi_rate, vego, 0.1)
    assert math.isfinite(result)
