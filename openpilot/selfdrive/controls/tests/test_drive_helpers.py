import pytest
import numpy as np

from openpilot.selfdrive.controls.lib.drive_helpers import (
  clamp,
  smooth_value,
  clip_curvature,
  get_accel_from_plan,
  should_stop,
  curv_from_psis,
  get_curvature_from_plan,
  MIN_SPEED,
  MAX_CURVATURE,
  MAX_LATERAL_JERK,
  MAX_LATERAL_ACCEL_NO_ROLL,
)
from openpilot.common.realtime import DT_CTRL, DT_MDL


class TestClamp:
  """Tests for the clamp utility function."""

  def test_value_within_bounds(self):
    """Value within bounds should be unchanged and not flagged as clamped."""
    val, clamped = clamp(5.0, 0.0, 10.0)
    assert val == 5.0
    assert clamped is False

  def test_value_below_min(self):
    """Value below min should be clamped to min."""
    val, clamped = clamp(-5.0, 0.0, 10.0)
    assert val == 0.0
    assert clamped is True

  def test_value_above_max(self):
    """Value above max should be clamped to max."""
    val, clamped = clamp(15.0, 0.0, 10.0)
    assert val == 10.0
    assert clamped is True

  def test_value_at_min_boundary(self):
    """Value exactly at min boundary should not be flagged as clamped."""
    val, clamped = clamp(0.0, 0.0, 10.0)
    assert val == 0.0
    assert clamped is False

  def test_value_at_max_boundary(self):
    """Value exactly at max boundary should not be flagged as clamped."""
    val, clamped = clamp(10.0, 0.0, 10.0)
    assert val == 10.0
    assert clamped is False

  def test_negative_range(self):
    """Clamping should work with negative ranges."""
    val, clamped = clamp(-15.0, -20.0, -10.0)
    assert val == -15.0
    assert clamped is False


class TestSmoothValue:
  """Tests for the smooth_value exponential filter function."""

  def test_instant_smoothing_when_tau_zero(self):
    """With tau=0, output should equal input immediately."""
    result = smooth_value(val=10.0, prev_val=0.0, tau=0.0)
    assert result == pytest.approx(10.0, abs=0.001)

  def test_slow_smoothing_with_large_tau(self):
    """With large tau, output should be close to prev_val."""
    result = smooth_value(val=10.0, prev_val=0.0, tau=100.0)
    # With very large tau, alpha is small, so result stays close to prev_val
    assert result < 1.0  # Mostly previous value

  def test_smoothing_converges_over_time(self):
    """Repeated smoothing should converge to target value."""
    val = 10.0
    prev = 0.0
    tau = 0.5
    for _ in range(100):
      prev = smooth_value(val=val, prev_val=prev, tau=tau, dt=DT_MDL)
    assert prev == pytest.approx(val, abs=0.01)

  def test_smoothing_with_custom_dt(self):
    """Custom dt should affect smoothing rate."""
    # Larger dt = faster convergence
    result_small_dt = smooth_value(val=10.0, prev_val=0.0, tau=1.0, dt=0.01)
    result_large_dt = smooth_value(val=10.0, prev_val=0.0, tau=1.0, dt=0.1)
    assert result_large_dt > result_small_dt  # Larger dt = faster convergence


class TestClipCurvature:
  """Tests for the clip_curvature function."""

  def test_curvature_rate_limited_from_zero(self):
    """Curvature change from zero should be rate limited by jerk."""
    v_ego = 10.0
    prev_curv = 0.0
    new_curv = 0.1  # Large change from zero
    max_rate = MAX_LATERAL_JERK / (max(v_ego, MIN_SPEED) ** 2)
    max_delta = max_rate * DT_CTRL

    result, _ = clip_curvature(v_ego, prev_curv, new_curv, roll=0.0)
    # Result should be limited by jerk rate from prev
    assert result == pytest.approx(prev_curv + max_delta, abs=0.001)

  def test_curvature_jerk_limiting(self):
    """Large curvature change should be rate limited."""
    v_ego = 10.0
    prev_curv = 0.0
    new_curv = 0.1  # Large change
    max_rate = MAX_LATERAL_JERK / (v_ego**2)
    max_delta = max_rate * DT_CTRL

    result, limited = clip_curvature(v_ego, prev_curv, new_curv, roll=0.0)
    # Result should be limited by jerk rate
    assert result <= prev_curv + max_delta + 0.001

  def test_curvature_lateral_accel_limiting(self):
    """Curvature exceeding lateral accel limit should be clamped."""
    v_ego = 10.0
    # Calculate a curvature that exceeds lateral accel limit
    max_lat_accel = MAX_LATERAL_ACCEL_NO_ROLL
    max_curv_from_accel = max_lat_accel / (v_ego**2)

    # When prev and new are the same, no jerk limiting applies
    new_curv = max_curv_from_accel * 2
    result, limited = clip_curvature(v_ego, prev_curvature=new_curv, new_curvature=new_curv, roll=0.0)
    assert result <= max_curv_from_accel + 0.001
    assert limited

  def test_max_curvature_clipping(self):
    """Curvature should be clipped to MAX_CURVATURE."""
    v_ego = 5.0  # Low speed for high curvature capability
    new_curv = 0.5  # Way beyond MAX_CURVATURE (0.2)
    result, limited = clip_curvature(v_ego, prev_curvature=new_curv, new_curvature=new_curv, roll=0.0)
    assert result <= MAX_CURVATURE
    assert limited

  def test_low_speed_uses_min_speed(self):
    """Very low speed should use MIN_SPEED for calculations."""
    v_ego = 0.1  # Very low
    new_curv = 0.05
    result, limited = clip_curvature(v_ego, prev_curvature=new_curv, new_curvature=new_curv, roll=0.0)
    # Should not crash and should use MIN_SPEED internally
    assert result is not None

  def test_roll_compensation_positive(self):
    """Positive roll should increase max lateral accel allowance."""
    v_ego = 10.0
    new_curv = 0.03
    # Positive roll (banking into the turn) allows more lateral accel
    result_no_roll, _ = clip_curvature(v_ego, prev_curvature=new_curv, new_curvature=new_curv, roll=0.0)
    result_with_roll, _ = clip_curvature(v_ego, prev_curvature=new_curv, new_curvature=new_curv, roll=0.1)
    # With positive roll, should allow same or higher curvature
    assert result_with_roll >= result_no_roll - 0.001

  def test_negative_curvature_lateral_accel_limiting(self):
    """Negative curvature (left turn) should also be clamped by lateral accel."""
    v_ego = 10.0
    # Curvature that exceeds lateral accel limit (negative)
    max_lat_accel = MAX_LATERAL_ACCEL_NO_ROLL
    max_curv_from_accel = max_lat_accel / (v_ego**2)
    new_curv = -max_curv_from_accel * 2  # Exceeds limit
    result, limited = clip_curvature(v_ego, prev_curvature=new_curv, new_curvature=new_curv, roll=0.0)
    assert result >= -max_curv_from_accel - 0.001
    assert limited


class TestGetAccelFromPlan:
  """Tests for the get_accel_from_plan function.

  fork: upstream split the stop decision out of get_accel_from_plan -- it now returns
  a_target alone, and should_stop(v_ego, a_target) is a separate helper with the
  vEgoStopping threshold folded in as a constant. TestShouldStop below covers it.
  """

  def test_valid_plan_interpolation(self):
    """Valid plan should interpolate acceleration correctly."""
    t_idxs = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    speeds = np.array([10.0, 10.5, 11.0, 11.5, 12.0])
    accels = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    a_target = get_accel_from_plan(speeds, accels, t_idxs, action_t=0.05)
    assert a_target != 0.0

  def test_invalid_plan_returns_zero(self):
    """Plan with mismatched lengths should return zero accel."""
    t_idxs = np.array([0.0, 0.5, 1.0])
    speeds = np.array([10.0, 10.5])  # Wrong length
    accels = np.array([1.0, 1.0])

    a_target = get_accel_from_plan(speeds, accels, t_idxs, action_t=0.05)
    assert a_target == 0.0


class TestShouldStop:
  """Tests for the should_stop helper."""

  def test_stops_when_slow_and_decelerating(self):
    """Stop when v_ego is below the threshold and a_target is not commanding accel."""
    assert should_stop(v_ego=0.01, a_target=0.0)

  def test_no_stop_when_moving(self):
    """Do not stop while still moving at speed."""
    assert not should_stop(v_ego=10.0, a_target=0.0)

  def test_no_stop_when_accelerating(self):
    """Do not stop when a_target commands meaningful acceleration."""
    assert not should_stop(v_ego=0.01, a_target=1.0)

  def test_threshold_boundaries(self):
    """Either condition alone is not enough to stop."""
    assert not should_stop(v_ego=0.3, a_target=0.0)   # at the speed threshold
    assert not should_stop(v_ego=0.0, a_target=0.1)   # at the accel threshold
    assert should_stop(v_ego=0.29, a_target=0.09)


class TestCurvFromPsis:
  """Tests for the curv_from_psis helper function."""

  def test_zero_values_returns_zero(self):
    """Zero inputs should return zero curvature."""
    result = curv_from_psis(psi_target=0.0, psi_rate=0.0, vego=10.0, action_t=0.1)
    assert result == pytest.approx(0.0, abs=0.001)

  def test_low_speed_clamps_to_min(self):
    """Very low speed should be clamped to MIN_SPEED."""
    # This should not cause division by zero
    result = curv_from_psis(psi_target=0.1, psi_rate=0.0, vego=0.01, action_t=0.1)
    assert np.isfinite(result)

  def test_positive_psi_target(self):
    """Positive psi_target should produce curvature."""
    result = curv_from_psis(psi_target=0.1, psi_rate=0.0, vego=10.0, action_t=0.5)
    assert result > 0.0


class TestGetCurvatureFromPlan:
  """Tests for the get_curvature_from_plan function."""

  def test_curvature_from_plan(self):
    """Should interpolate yaw and compute curvature."""
    t_idxs = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    yaws = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    yaw_rates = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

    result = get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego=10.0, action_t=0.1)
    assert np.isfinite(result)

  def test_stationary_plan_zero_curvature(self):
    """Plan with zero yaws should produce zero curvature."""
    t_idxs = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    yaws = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    yaw_rates = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    result = get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego=10.0, action_t=0.1)
    assert result == pytest.approx(0.0, abs=0.001)
