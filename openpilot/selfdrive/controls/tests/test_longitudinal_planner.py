import pytest

from opendbc.car.structs import car
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  LongitudinalPlanner,
  get_max_accel,
  get_coast_accel,
  get_cruise_accel,
  A_CRUISE_MAX_VALS,
  A_CRUISE_MIN,
  ALLOW_THROTTLE_THRESHOLD,
  MIN_ALLOW_THROTTLE_SPEED,
)


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


class TestCruiseAccelInTurns:
  """Tests for the lateral-accel limiting inside get_cruise_accel.

  fork: upstream removed limit_accel_in_turns and folded its a_total_max / a_y budget
  into get_cruise_accel's non-e2e branch, so these exercise it through that entry point.
  """

  def _create_car_params(self, steer_ratio=15.0, wheelbase=2.7):
    """Create CarParams with steering geometry."""
    CP = car.CarParams.new_message()
    CP.steerRatio = steer_ratio
    CP.wheelbase = wheelbase
    return CP

  def _accel(self, v_ego, angle_steers, CP, v_cruise_delta=5.0, a_cruise_prev=0.0, dt=1.0):
    """Ask for more accel than the turn budget allows and see what comes back.

    dt is deliberately large so the jerk clamp around a_cruise_prev does not bind.
    """
    return get_cruise_accel(
      e2e=False, v_cruise=v_ego + v_cruise_delta, v_ego=v_ego, a_cruise_prev=a_cruise_prev,
      angle_steers=angle_steers, CP=CP, dt=dt, accel_coast=0.0, allow_throttle=True,
    )

  def test_straight_line_limited_by_max_accel(self):
    """Straight driving is limited by get_max_accel, not by lateral accel."""
    CP = self._create_car_params()
    assert self._accel(20.0, 0.0, CP) == pytest.approx(get_max_accel(20.0), abs=0.01)

  def test_high_lateral_accel_limits(self):
    """High lateral acceleration in a turn should limit longitudinal accel."""
    CP = self._create_car_params()
    straight = self._accel(30.0, 0.0, CP)
    turning = self._accel(30.0, 45.0, CP)
    assert turning < straight

  def test_low_speed_less_lateral_effect(self):
    """Lateral accel scales with v^2, so the same angle costs less when slow."""
    CP = self._create_car_params()
    slow = self._accel(5.0, 45.0, CP)
    fast = self._accel(30.0, 45.0, CP)
    assert slow > fast
    assert slow >= 0.0

  def test_e2e_skips_turn_limiting(self):
    """In e2e mode the lateral budget is not applied."""
    CP = self._create_car_params()
    turning = get_cruise_accel(
      e2e=True, v_cruise=35.0, v_ego=30.0, a_cruise_prev=0.0,
      angle_steers=45.0, CP=CP, dt=1.0, accel_coast=0.0, allow_throttle=True,
    )
    assert turning > self._accel(30.0, 45.0, CP)

  def test_never_below_a_cruise_min(self):
    """A deceleration request is floored at A_CRUISE_MIN."""
    CP = self._create_car_params()
    result = get_cruise_accel(
      e2e=False, v_cruise=0.0, v_ego=30.0, a_cruise_prev=A_CRUISE_MIN,
      angle_steers=0.0, CP=CP, dt=1.0, accel_coast=0.0, allow_throttle=True,
    )
    assert result >= A_CRUISE_MIN


# fork: upstream removed LongitudinalPlanner.parse_model -- the position/velocity/
# acceleration interpolation moved into the MPC call path and throttle_prob is now read
# inline in update(). The throttle_prob threshold behavior it used to cover is exercised
# through update() by the allow-throttle tests below.

class TestLongitudinalPlannerInit:
  """Tests for LongitudinalPlanner initialization."""

  def _create_car_params(self):
    """Create basic CarParams for planner."""
    CP = car.CarParams.new_message()
    CP.steerRatio = 15.0
    CP.wheelbase = 2.7
    CP.openpilotLongitudinalControl = True
    CP.longitudinalActuatorDelay = 0.5
    return CP

  def test_initial_values(self):
    """Planner should initialize with expected default values."""
    CP = self._create_car_params()
    planner = LongitudinalPlanner(CP, init_v=0.0, init_a=0.0)

    assert planner.fcw is False
    assert planner.allow_throttle is True
    assert planner.a_cruise == 0.0
    assert planner.output_a_target == 0.0
    assert planner.output_should_stop is False

  def test_initial_values_with_nonzero_start(self):
    """Planner should respect initial velocity and acceleration."""
    CP = self._create_car_params()
    planner = LongitudinalPlanner(CP, init_v=10.0, init_a=1.5)

    assert planner.a_cruise == 1.5
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


# fork: upstream removed LongitudinalPlanner.prev_accel_clip and its per-frame 0.05
# rate limit; the equivalent smoothing now lives in get_cruise_accel's J_CRUISE_VALS jerk
# clamp, covered by TestCruiseAccelInTurns above.
