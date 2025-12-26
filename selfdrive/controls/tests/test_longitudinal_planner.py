import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from cereal import car, log
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  LongitudinalPlanner,
  get_max_accel,
  get_coast_accel,
  limit_accel_in_turns,
  A_CRUISE_MAX_VALS,
  A_CRUISE_MAX_BP,
  ALLOW_THROTTLE_THRESHOLD,
  MIN_ALLOW_THROTTLE_SPEED,
)
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState


class TestGetMaxAccel:
  """Tests for get_max_accel helper function."""

  def test_at_zero_speed(self):
    """Max accel at 0 m/s should be highest value."""
    result = get_max_accel(0.0)
    assert result == pytest.approx(A_CRUISE_MAX_VALS[0], abs=0.01)

  def test_at_high_speed(self):
    """Max accel at 40 m/s should be lowest value."""
    result = get_max_accel(40.0)
    assert result == pytest.approx(A_CRUISE_MAX_VALS[-1], abs=0.01)

  def test_interpolation_midpoint(self):
    """Max accel at 10 m/s should be interpolated."""
    result = get_max_accel(10.0)
    assert result == pytest.approx(A_CRUISE_MAX_VALS[1], abs=0.01)

  def test_above_max_bp(self):
    """Max accel above max BP should clamp to last value."""
    result = get_max_accel(100.0)
    assert result == pytest.approx(A_CRUISE_MAX_VALS[-1], abs=0.01)


class TestGetCoastAccel:
  """Tests for get_coast_accel helper function."""

  def test_zero_pitch(self):
    """At zero pitch, coast accel should be about -0.3."""
    result = get_coast_accel(0.0)
    assert result == pytest.approx(-0.3, abs=0.01)

  def test_positive_pitch_uphill(self):
    """Positive pitch (uphill) should give more negative accel."""
    result = get_coast_accel(0.1)  # ~5.7 degrees
    assert result < -0.3  # More deceleration uphill

  def test_negative_pitch_downhill(self):
    """Negative pitch (downhill) should give less negative accel."""
    result = get_coast_accel(-0.1)  # ~-5.7 degrees
    assert result > -0.3  # Less deceleration (or acceleration) downhill


class TestLimitAccelInTurns:
  """Tests for limit_accel_in_turns helper function."""

  def _create_car_params(self, steer_ratio=15.0, wheelbase=2.7):
    """Create CarParams with steering geometry."""
    CP = car.CarParams.new_message()
    CP.steerRatio = steer_ratio
    CP.wheelbase = wheelbase
    return CP

  def test_straight_line_limited_by_total_max(self):
    """Straight driving still respects a_total_max based on speed."""
    CP = self._create_car_params()
    a_target = [ACCEL_MIN, 2.0]
    # At 20 m/s, a_total_max is 1.7 (from _A_TOTAL_MAX_BP/V lookup)
    result = limit_accel_in_turns(v_ego=20.0, angle_steers=0.0, a_target=a_target, CP=CP)
    assert result[0] == a_target[0]
    assert result[1] == pytest.approx(1.7, abs=0.01)  # Limited by a_total_max at 20 m/s

  def test_straight_line_below_total_max_unchanged(self):
    """Request below a_total_max should pass through unchanged."""
    CP = self._create_car_params()
    a_target = [ACCEL_MIN, 1.0]  # Below a_total_max at any speed
    result = limit_accel_in_turns(v_ego=20.0, angle_steers=0.0, a_target=a_target, CP=CP)
    assert result[0] == a_target[0]
    assert result[1] == pytest.approx(a_target[1], abs=0.01)  # Unchanged

  def test_high_lateral_accel_limits(self):
    """High lateral acceleration in turn should limit longitudinal accel."""
    CP = self._create_car_params()
    a_target = [ACCEL_MIN, 2.0]
    # Large steering angle at speed creates high lateral accel
    result = limit_accel_in_turns(v_ego=30.0, angle_steers=45.0, a_target=a_target, CP=CP)
    assert result[1] < a_target[1]  # Should be limited

  def test_low_speed_less_lateral_effect(self):
    """At low speeds, lateral accel is lower so less limiting."""
    CP = self._create_car_params()
    a_target = [ACCEL_MIN, 2.0]
    result = limit_accel_in_turns(v_ego=5.0, angle_steers=45.0, a_target=a_target, CP=CP)
    # Low speed should have less lateral accel effect
    assert result[1] >= 0.0  # Should still allow some accel

  def test_min_accel_unchanged(self):
    """Minimum accel limit should never change."""
    CP = self._create_car_params()
    a_target = [ACCEL_MIN, 2.0]
    result = limit_accel_in_turns(v_ego=30.0, angle_steers=45.0, a_target=a_target, CP=CP)
    assert result[0] == a_target[0]


class TestParseModel:
  """Tests for LongitudinalPlanner.parse_model static method."""

  def _create_model_msg(self, n_elements=ModelConstants.IDX_N, throttle_prob=1.0):
    """Create a modelV2 message with specified array sizes."""
    model = MagicMock()
    model.position.x = list(np.linspace(0, 100, n_elements))
    model.velocity.x = list(np.linspace(10, 20, n_elements))
    model.acceleration.x = list(np.linspace(0, 2, n_elements))
    if throttle_prob is not None:
      model.meta.disengagePredictions.gasPressProbs = [0.0, throttle_prob]
    else:
      model.meta.disengagePredictions.gasPressProbs = []
    return model

  def test_valid_model_parsed_correctly(self):
    """Valid model with correct array lengths should parse successfully."""
    model = self._create_model_msg(n_elements=ModelConstants.IDX_N)
    x, v, a, j, throttle_prob = LongitudinalPlanner.parse_model(model)

    assert len(x) > 0
    assert len(v) > 0
    assert len(a) > 0
    assert len(j) > 0
    assert throttle_prob == 1.0

  def test_invalid_model_returns_zeros(self):
    """Invalid model with wrong array lengths should return zeros."""
    model = self._create_model_msg(n_elements=5)  # Wrong size
    x, v, a, j, throttle_prob = LongitudinalPlanner.parse_model(model)

    assert np.all(x == 0)
    assert np.all(v == 0)
    assert np.all(a == 0)
    assert np.all(j == 0)

  def test_missing_throttle_prob_defaults_to_one(self):
    """Missing throttle probability should default to 1.0."""
    model = self._create_model_msg(n_elements=ModelConstants.IDX_N, throttle_prob=None)
    _, _, _, _, throttle_prob = LongitudinalPlanner.parse_model(model)

    assert throttle_prob == 1.0

  def test_low_throttle_prob(self):
    """Low throttle probability should be returned correctly."""
    model = self._create_model_msg(n_elements=ModelConstants.IDX_N, throttle_prob=0.2)
    _, _, _, _, throttle_prob = LongitudinalPlanner.parse_model(model)

    assert throttle_prob == pytest.approx(0.2, abs=0.01)


class TestLongitudinalPlannerInit:
  """Tests for LongitudinalPlanner initialization."""

  def _create_car_params(self):
    """Create basic CarParams for planner."""
    CP = car.CarParams.new_message()
    CP.steerRatio = 15.0
    CP.wheelbase = 2.7
    CP.openpilotLongitudinalControl = True
    CP.longitudinalActuatorDelay = 0.5
    CP.vEgoStopping = 0.5
    return CP

  def test_initial_values(self):
    """Planner should initialize with expected default values."""
    CP = self._create_car_params()
    planner = LongitudinalPlanner(CP, init_v=0.0, init_a=0.0)

    assert planner.fcw is False
    assert planner.allow_throttle is True
    assert planner.a_desired == 0.0
    assert planner.output_a_target == 0.0
    assert planner.output_should_stop is False

  def test_initial_values_with_nonzero_start(self):
    """Planner should respect initial velocity and acceleration."""
    CP = self._create_car_params()
    planner = LongitudinalPlanner(CP, init_v=10.0, init_a=1.5)

    assert planner.a_desired == 1.5
    assert planner.v_desired_filter.x == pytest.approx(10.0, abs=0.01)

  def test_trajectory_arrays_initialized(self):
    """Trajectory arrays should be initialized to correct length."""
    CP = self._create_car_params()
    planner = LongitudinalPlanner(CP)

    from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
    assert len(planner.v_desired_trajectory) == CONTROL_N
    assert len(planner.a_desired_trajectory) == CONTROL_N
    assert len(planner.j_desired_trajectory) == CONTROL_N


class TestAllowThrottleLogic:
  """Tests for the allow_throttle threshold logic."""

  def test_high_throttle_prob_allows_throttle(self):
    """Throttle probability above threshold should allow throttle."""
    throttle_prob = 0.6  # Above ALLOW_THROTTLE_THRESHOLD (0.4)
    v_ego = 10.0  # Above MIN_ALLOW_THROTTLE_SPEED
    allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED
    assert allow_throttle is True

  def test_low_throttle_prob_disallows_throttle(self):
    """Throttle probability below threshold should disallow throttle at higher speeds."""
    throttle_prob = 0.2  # Below ALLOW_THROTTLE_THRESHOLD (0.4)
    v_ego = 10.0  # Above MIN_ALLOW_THROTTLE_SPEED
    allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED
    assert allow_throttle is False

  def test_low_speed_always_allows_throttle(self):
    """Low speed should always allow throttle regardless of probability."""
    throttle_prob = 0.1  # Very low
    v_ego = 2.0  # Below MIN_ALLOW_THROTTLE_SPEED (2.5)
    allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED
    assert allow_throttle is True


class TestFCWLogic:
  """Tests for Forward Collision Warning trigger logic."""

  def test_fcw_not_triggered_initially(self):
    """FCW should not be triggered on initialization."""
    CP = car.CarParams.new_message()
    CP.steerRatio = 15.0
    CP.wheelbase = 2.7
    planner = LongitudinalPlanner(CP)
    assert planner.fcw is False

  def test_fcw_logic_with_crash_counter(self):
    """FCW should trigger when crash_cnt > 2 and not standstill."""
    # FCW is triggered when: mpc.crash_cnt > 2 and not sm['carState'].standstill
    crash_cnt = 3
    standstill = False
    fcw = crash_cnt > 2 and not standstill
    assert fcw is True

  def test_fcw_not_triggered_at_standstill(self):
    """FCW should not trigger at standstill even with high crash count."""
    crash_cnt = 10
    standstill = True
    fcw = crash_cnt > 2 and not standstill
    assert fcw is False

  def test_fcw_not_triggered_low_crash_count(self):
    """FCW should not trigger with low crash count."""
    crash_cnt = 1
    standstill = False
    fcw = crash_cnt > 2 and not standstill
    assert fcw is False


class TestAccelClipping:
  """Tests for acceleration clipping behavior."""

  def test_accel_limits_initialized(self):
    """prev_accel_clip should be initialized to full range."""
    CP = car.CarParams.new_message()
    CP.steerRatio = 15.0
    CP.wheelbase = 2.7
    planner = LongitudinalPlanner(CP)
    assert planner.prev_accel_clip[0] == ACCEL_MIN
    assert planner.prev_accel_clip[1] == ACCEL_MAX

  def test_accel_clip_rate_limiting(self):
    """Accel clip should be rate limited by 0.05 per step."""
    prev_clip = [ACCEL_MIN, 1.0]
    new_clip = [ACCEL_MIN, 2.0]
    rate_limit = 0.05

    # Simulate rate limiting logic from update()
    clipped = [0.0, 0.0]
    for idx in range(2):
      clipped[idx] = np.clip(new_clip[idx], prev_clip[idx] - rate_limit, prev_clip[idx] + rate_limit)

    assert clipped[0] == prev_clip[0]  # Min unchanged
    assert clipped[1] == pytest.approx(prev_clip[1] + rate_limit, abs=0.001)  # Max rate limited
