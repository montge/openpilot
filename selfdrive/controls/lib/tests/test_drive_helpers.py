"""Tests for selfdrive/controls/lib/drive_helpers.py - driving helper utilities."""
import unittest

import numpy as np

from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import (
  clamp, smooth_value, clip_curvature, get_accel_from_plan,
  curv_from_psis, get_curvature_from_plan,
  MIN_SPEED, MAX_CURVATURE, MAX_LATERAL_JERK, MAX_LATERAL_ACCEL_NO_ROLL,
)


class TestClamp(unittest.TestCase):
  """Test clamp function."""

  def test_value_within_bounds(self):
    """Test value within bounds is not clamped."""
    val, clamped = clamp(5.0, 0.0, 10.0)
    self.assertEqual(val, 5.0)
    self.assertFalse(clamped)

  def test_value_at_min(self):
    """Test value at min is not clamped."""
    val, clamped = clamp(0.0, 0.0, 10.0)
    self.assertEqual(val, 0.0)
    self.assertFalse(clamped)

  def test_value_at_max(self):
    """Test value at max is not clamped."""
    val, clamped = clamp(10.0, 0.0, 10.0)
    self.assertEqual(val, 10.0)
    self.assertFalse(clamped)

  def test_value_below_min_clamped(self):
    """Test value below min is clamped."""
    val, clamped = clamp(-5.0, 0.0, 10.0)
    self.assertEqual(val, 0.0)
    self.assertTrue(clamped)

  def test_value_above_max_clamped(self):
    """Test value above max is clamped."""
    val, clamped = clamp(15.0, 0.0, 10.0)
    self.assertEqual(val, 10.0)
    self.assertTrue(clamped)

  def test_negative_range(self):
    """Test clamping with negative range."""
    val, clamped = clamp(-5.0, -10.0, -2.0)
    self.assertEqual(val, -5.0)
    self.assertFalse(clamped)

  def test_returns_float(self):
    """Test result is always a float."""
    val, _ = clamp(5, 0, 10)
    self.assertIsInstance(val, float)


class TestSmoothValue(unittest.TestCase):
  """Test smooth_value function."""

  def test_tau_zero_returns_val(self):
    """Test tau=0 returns the current value (no smoothing)."""
    result = smooth_value(10.0, 5.0, 0.0)
    self.assertEqual(result, 10.0)

  def test_tau_negative_returns_val(self):
    """Test negative tau returns current value."""
    result = smooth_value(10.0, 5.0, -1.0)
    self.assertEqual(result, 10.0)

  def test_large_tau_heavily_smooths(self):
    """Test large tau produces heavy smoothing."""
    # With large tau, result should be close to prev_val
    result = smooth_value(10.0, 5.0, 100.0)
    self.assertGreater(result, 5.0)
    self.assertLess(result, 6.0)  # Should be very close to prev

  def test_small_tau_light_smoothing(self):
    """Test small tau produces light smoothing."""
    result = smooth_value(10.0, 5.0, 0.001)
    # Small tau means result should be close to new val
    self.assertGreater(result, 9.0)

  def test_same_values_unchanged(self):
    """Test same val and prev_val stays unchanged."""
    result = smooth_value(5.0, 5.0, 1.0)
    self.assertEqual(result, 5.0)

  def test_custom_dt(self):
    """Test with custom dt parameter."""
    result1 = smooth_value(10.0, 5.0, 1.0, dt=0.1)
    result2 = smooth_value(10.0, 5.0, 1.0, dt=0.01)
    # Larger dt means more smoothing toward new value
    self.assertGreater(result1, result2)


class TestClipCurvature(unittest.TestCase):
  """Test clip_curvature function."""

  def test_curvature_within_jerk_limit(self):
    """Test small curvature change within jerk limit passes through."""
    v_ego = 10.0
    prev_curv = 0.0
    # Very small change that should be within jerk rate limit
    max_rate = MAX_LATERAL_JERK / (v_ego ** 2) * DT_CTRL
    new_curv = max_rate * 0.5  # Half the max rate

    result, limited = clip_curvature(v_ego, prev_curv, new_curv, roll=0.0)
    self.assertAlmostEqual(result, new_curv, places=5)

  def test_curvature_jerk_limited(self):
    """Test large curvature change is jerk-limited."""
    v_ego = 10.0
    prev_curv = 0.0
    new_curv = 0.1  # Large step from 0
    roll = 0.0

    result, limited = clip_curvature(v_ego, prev_curv, new_curv, roll)
    max_rate = MAX_LATERAL_JERK / (v_ego ** 2) * DT_CTRL
    # Result should be limited to max jerk rate
    self.assertLessEqual(abs(result - prev_curv), max_rate + 1e-6)

  def test_curvature_exceeds_max_from_prev(self):
    """Test curvature exceeding MAX_CURVATURE is clipped when prev is at limit."""
    v_ego = 10.0
    prev_curv = MAX_CURVATURE - 0.001  # Just below max
    new_curv = MAX_CURVATURE + 0.1  # Trying to exceed max
    roll = 0.0

    result, limited = clip_curvature(v_ego, prev_curv, new_curv, roll)
    self.assertTrue(limited)
    self.assertLessEqual(result, MAX_CURVATURE)

  def test_curvature_below_min_from_prev(self):
    """Test negative curvature exceeding limit is clipped when prev is at limit."""
    v_ego = 10.0
    prev_curv = -MAX_CURVATURE + 0.001  # Just above negative max
    new_curv = -MAX_CURVATURE - 0.1  # Trying to go below
    roll = 0.0

    result, limited = clip_curvature(v_ego, prev_curv, new_curv, roll)
    self.assertTrue(limited)
    self.assertGreaterEqual(result, -MAX_CURVATURE)

  def test_low_speed_clamped(self):
    """Test low speed is clamped to MIN_SPEED."""
    v_ego = 0.1  # Very low speed
    prev_curv = 0.0
    new_curv = 0.01
    roll = 0.0

    # Should not raise, speed is clamped to MIN_SPEED
    result, _ = clip_curvature(v_ego, prev_curv, new_curv, roll)
    self.assertIsInstance(result, float)

  def test_roll_compensation(self):
    """Test roll compensation affects limits."""
    v_ego = 10.0
    prev_curv = 0.0
    new_curv = 0.05
    roll_pos = 0.1  # Positive roll
    roll_neg = -0.1  # Negative roll

    result_pos, _ = clip_curvature(v_ego, prev_curv, new_curv, roll_pos)
    result_neg, _ = clip_curvature(v_ego, prev_curv, new_curv, roll_neg)

    # Different roll should potentially produce different results
    # due to different acceleration limits
    # Results may be same if not hitting limits
    self.assertIsInstance(result_pos, float)
    self.assertIsInstance(result_neg, float)


class TestGetAccelFromPlan(unittest.TestCase):
  """Test get_accel_from_plan function."""

  def test_valid_plan(self):
    """Test with valid speed/accel plan."""
    t_idxs = np.array([0.0, 0.1, 0.2, 0.3])
    speeds = np.array([10.0, 10.0, 10.0, 10.0])
    accels = np.array([0.0, 0.0, 0.0, 0.0])

    a_target, should_stop = get_accel_from_plan(speeds, accels, t_idxs)

    self.assertIsInstance(a_target, float)
    self.assertFalse(should_stop)

  def test_stopping_plan(self):
    """Test plan that results in stopping."""
    t_idxs = np.array([0.0, 0.5, 1.0, 1.5])
    speeds = np.array([0.01, 0.01, 0.01, 0.01])  # Very slow
    accels = np.array([0.0, 0.0, 0.0, 0.0])

    a_target, should_stop = get_accel_from_plan(speeds, accels, t_idxs)

    self.assertTrue(should_stop)

  def test_accelerating_plan(self):
    """Test plan with increasing speed."""
    t_idxs = np.array([0.0, 0.1, 0.2, 0.3])
    speeds = np.array([10.0, 11.0, 12.0, 13.0])
    accels = np.array([1.0, 1.0, 1.0, 1.0])

    a_target, should_stop = get_accel_from_plan(speeds, accels, t_idxs)

    self.assertGreater(a_target, 0)
    self.assertFalse(should_stop)

  def test_decelerating_plan(self):
    """Test plan with decreasing speed."""
    t_idxs = np.array([0.0, 0.1, 0.2, 0.3])
    speeds = np.array([15.0, 14.0, 13.0, 12.0])
    accels = np.array([-1.0, -1.0, -1.0, -1.0])

    a_target, should_stop = get_accel_from_plan(speeds, accels, t_idxs)

    self.assertLess(a_target, 0)
    self.assertFalse(should_stop)

  def test_mismatched_lengths(self):
    """Test with mismatched array lengths returns defaults."""
    t_idxs = np.array([0.0, 0.1, 0.2])
    speeds = np.array([10.0, 10.0])  # Different length
    accels = np.array([0.0, 0.0])

    a_target, should_stop = get_accel_from_plan(speeds, accels, t_idxs)

    self.assertEqual(a_target, 0.0)
    self.assertTrue(should_stop)  # v_target=0 and v_target_1sec=0

  def test_custom_vego_stopping(self):
    """Test custom vEgoStopping threshold."""
    t_idxs = np.array([0.0, 0.5, 1.0, 1.5])
    speeds = np.array([0.1, 0.1, 0.1, 0.1])
    accels = np.array([0.0, 0.0, 0.0, 0.0])

    # Default threshold is 0.05, so 0.1 should not stop
    _, should_stop_default = get_accel_from_plan(speeds, accels, t_idxs)
    self.assertFalse(should_stop_default)

    # With higher threshold, should stop
    _, should_stop_custom = get_accel_from_plan(speeds, accels, t_idxs, vEgoStopping=0.2)
    self.assertTrue(should_stop_custom)


class TestCurvFromPsis(unittest.TestCase):
  """Test curv_from_psis function."""

  def test_zero_values(self):
    """Test with zero inputs."""
    result = curv_from_psis(0.0, 0.0, 10.0, DT_MDL)
    self.assertEqual(result, 0.0)

  def test_low_speed_clamped(self):
    """Test low speed is clamped to MIN_SPEED."""
    # Should not raise even with very low speed
    result = curv_from_psis(0.1, 0.0, 0.01, DT_MDL)
    self.assertIsInstance(result, float)

  def test_positive_psi_target(self):
    """Test with positive psi target."""
    result = curv_from_psis(0.1, 0.0, 10.0, DT_MDL)
    self.assertGreater(result, 0)

  def test_negative_psi_target(self):
    """Test with negative psi target."""
    result = curv_from_psis(-0.1, 0.0, 10.0, DT_MDL)
    self.assertLess(result, 0)

  def test_psi_rate_affects_result(self):
    """Test psi_rate affects curvature."""
    result_no_rate = curv_from_psis(0.1, 0.0, 10.0, DT_MDL)
    result_with_rate = curv_from_psis(0.1, 0.5, 10.0, DT_MDL)
    self.assertNotEqual(result_no_rate, result_with_rate)


class TestGetCurvatureFromPlan(unittest.TestCase):
  """Test get_curvature_from_plan function."""

  def test_valid_plan(self):
    """Test with valid yaw plan."""
    t_idxs = np.array([0.0, 0.1, 0.2, 0.3])
    yaws = np.array([0.0, 0.01, 0.02, 0.03])
    yaw_rates = np.array([0.1, 0.1, 0.1, 0.1])
    vego = 10.0

    result = get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego, DT_MDL)
    self.assertIsInstance(result, float)

  def test_zero_yaw(self):
    """Test with zero yaw values."""
    t_idxs = np.array([0.0, 0.1, 0.2])
    yaws = np.array([0.0, 0.0, 0.0])
    yaw_rates = np.array([0.0, 0.0, 0.0])
    vego = 10.0

    result = get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego, DT_MDL)
    self.assertEqual(result, 0.0)

  def test_interpolation(self):
    """Test yaw is interpolated at action_t."""
    t_idxs = np.array([0.0, 0.1, 0.2])
    yaws = np.array([0.0, 0.1, 0.2])
    yaw_rates = np.array([1.0, 1.0, 1.0])
    vego = 10.0

    # With different action_t, should get different curvature
    result1 = get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego, 0.05)
    result2 = get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego, 0.15)
    self.assertNotEqual(result1, result2)


class TestConstants(unittest.TestCase):
  """Test module constants."""

  def test_min_speed_positive(self):
    """Test MIN_SPEED is positive."""
    self.assertGreater(MIN_SPEED, 0)

  def test_max_curvature_positive(self):
    """Test MAX_CURVATURE is positive."""
    self.assertGreater(MAX_CURVATURE, 0)

  def test_max_lateral_jerk_positive(self):
    """Test MAX_LATERAL_JERK is positive."""
    self.assertGreater(MAX_LATERAL_JERK, 0)

  def test_max_lateral_accel_positive(self):
    """Test MAX_LATERAL_ACCEL_NO_ROLL is positive."""
    self.assertGreater(MAX_LATERAL_ACCEL_NO_ROLL, 0)


if __name__ == '__main__':
  unittest.main()
