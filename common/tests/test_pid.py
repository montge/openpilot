import numpy as np
import pytest

from openpilot.common.pid import PIDController


class TestPIDController:
  def test_init_with_scalar_gains(self):
    """Test PID initialization with scalar gains (converted to lookup tables)."""
    pid = PIDController(k_p=1.0, k_i=0.1, k_d=0.01)

    assert pid._k_p == [[0], [1.0]]
    assert pid._k_i == [[0], [0.1]]
    assert pid._k_d == [[0], [0.01]]
    assert pid.pos_limit == 1e308
    assert pid.neg_limit == -1e308

  def test_init_with_lookup_tables(self):
    """Test PID initialization with speed-dependent lookup tables."""
    k_p = [[0, 10, 20], [1.0, 0.8, 0.6]]
    k_i = [[0, 10, 20], [0.1, 0.08, 0.06]]
    k_d = [[0, 10, 20], [0.01, 0.008, 0.006]]

    pid = PIDController(k_p=k_p, k_i=k_i, k_d=k_d)

    assert pid._k_p == k_p
    assert pid._k_i == k_i
    assert pid._k_d == k_d

  def test_init_with_limits(self):
    """Test PID initialization with custom limits."""
    pid = PIDController(k_p=1.0, k_i=0.1, pos_limit=100, neg_limit=-50)

    assert pid.pos_limit == 100
    assert pid.neg_limit == -50

  def test_init_with_rate(self):
    """Test PID initialization with custom rate."""
    pid = PIDController(k_p=1.0, k_i=0.1, rate=50)
    assert pid.i_dt == 1.0 / 50

  def test_reset(self):
    """Test that reset clears all internal state."""
    pid = PIDController(k_p=1.0, k_i=0.1, k_d=0.01)

    # Run a few updates to build up state
    pid.update(error=5.0, error_rate=0.5, speed=10.0, feedforward=0.1)
    pid.update(error=3.0, error_rate=0.3, speed=10.0, feedforward=0.1)

    # Reset and verify state is cleared
    pid.reset()

    assert pid.p == 0.0
    assert pid.i == 0.0
    assert pid.d == 0.0
    assert pid.f == 0.0
    assert pid.control == 0

  def test_set_limits(self):
    """Test setting limits after initialization."""
    pid = PIDController(k_p=1.0, k_i=0.1)
    pid.set_limits(pos_limit=50, neg_limit=-25)

    assert pid.pos_limit == 50
    assert pid.neg_limit == -25

  def test_proportional_term(self):
    """Test that proportional term is calculated correctly."""
    pid = PIDController(k_p=2.0, k_i=0.0, k_d=0.0)

    # P term should be k_p * error
    control = pid.update(error=5.0)

    assert pid.p == 10.0  # 2.0 * 5.0
    assert control == 10.0

  def test_derivative_term(self):
    """Test that derivative term is calculated correctly."""
    pid = PIDController(k_p=0.0, k_i=0.0, k_d=1.5)

    # D term should be k_d * error_rate
    control = pid.update(error=0.0, error_rate=4.0)

    assert pid.d == 6.0  # 1.5 * 4.0
    assert control == 6.0

  def test_integral_term(self):
    """Test that integral term accumulates correctly."""
    pid = PIDController(k_p=0.0, k_i=1.0, k_d=0.0, rate=100)

    # First update: i = 0 + k_i * i_dt * error = 0 + 1.0 * 0.01 * 10 = 0.1
    control1 = pid.update(error=10.0)
    assert pid.i == pytest.approx(0.1)
    assert control1 == pytest.approx(0.1)

    # Second update: i = 0.1 + 1.0 * 0.01 * 10 = 0.2
    control2 = pid.update(error=10.0)
    assert pid.i == pytest.approx(0.2)
    assert control2 == pytest.approx(0.2)

  def test_feedforward_term(self):
    """Test that feedforward term is passed through correctly."""
    pid = PIDController(k_p=0.0, k_i=0.0, k_d=0.0)

    control = pid.update(error=0.0, feedforward=7.5)

    assert pid.f == 7.5
    assert control == 7.5

  def test_combined_terms(self):
    """Test that all terms are combined correctly."""
    pid = PIDController(k_p=1.0, k_i=1.0, k_d=1.0, rate=100)

    # P = 1.0 * 2.0 = 2.0
    # I = 0 + 1.0 * 0.01 * 2.0 = 0.02
    # D = 1.0 * 0.5 = 0.5
    # F = 1.0
    # Total = 2.0 + 0.02 + 0.5 + 1.0 = 3.52
    control = pid.update(error=2.0, error_rate=0.5, feedforward=1.0)

    assert pid.p == pytest.approx(2.0)
    assert pid.i == pytest.approx(0.02)
    assert pid.d == pytest.approx(0.5)
    assert pid.f == pytest.approx(1.0)
    assert control == pytest.approx(3.52)

  def test_positive_limit_clipping(self):
    """Test that control output is clipped at positive limit."""
    pid = PIDController(k_p=10.0, k_i=0.0, pos_limit=5.0, neg_limit=-5.0)

    control = pid.update(error=10.0)  # Would be 100 without clipping

    assert control == 5.0

  def test_negative_limit_clipping(self):
    """Test that control output is clipped at negative limit."""
    pid = PIDController(k_p=10.0, k_i=0.0, pos_limit=5.0, neg_limit=-5.0)

    control = pid.update(error=-10.0)  # Would be -100 without clipping

    assert control == -5.0

  def test_integrator_anti_windup_positive(self):
    """Test that integrator doesn't wind up when clipping at positive limit."""
    pid = PIDController(k_p=1.0, k_i=10.0, pos_limit=10.0, neg_limit=-10.0, rate=100)

    # Large error that would cause integrator windup
    for _ in range(100):
      pid.update(error=5.0)

    # Control should be clipped at limit
    assert pid.control == 10.0

    # Now apply negative error - integrator should respond immediately
    # without having to "unwind" a large accumulated value
    pid.update(error=-5.0)
    assert pid.control < 10.0

  def test_integrator_anti_windup_negative(self):
    """Test that integrator doesn't wind up when clipping at negative limit."""
    pid = PIDController(k_p=1.0, k_i=10.0, pos_limit=10.0, neg_limit=-10.0, rate=100)

    # Large negative error that would cause integrator windup
    for _ in range(100):
      pid.update(error=-5.0)

    # Control should be clipped at limit
    assert pid.control == -10.0

    # Now apply positive error - integrator should respond immediately
    pid.update(error=5.0)
    assert pid.control > -10.0

  def test_freeze_integrator(self):
    """Test that freeze_integrator prevents integrator accumulation."""
    pid = PIDController(k_p=0.0, k_i=1.0, k_d=0.0, rate=100)

    # First update without freeze - integrator should accumulate
    pid.update(error=10.0, freeze_integrator=False)
    i_after_first = pid.i

    # Second update with freeze - integrator should not change
    pid.update(error=10.0, freeze_integrator=True)
    assert pid.i == i_after_first

    # Third update without freeze - integrator should accumulate again
    pid.update(error=10.0, freeze_integrator=False)
    assert pid.i > i_after_first

  def test_speed_dependent_gains(self):
    """Test that gains vary with speed when using lookup tables."""
    k_p = [[0, 10, 20], [2.0, 1.0, 0.5]]
    pid = PIDController(k_p=k_p, k_i=0.0, k_d=0.0)

    # At speed 0, k_p should be 2.0
    pid.update(error=1.0, speed=0.0)
    assert pid.p == pytest.approx(2.0)

    # At speed 10, k_p should be 1.0
    pid.update(error=1.0, speed=10.0)
    assert pid.p == pytest.approx(1.0)

    # At speed 20, k_p should be 0.5
    pid.update(error=1.0, speed=20.0)
    assert pid.p == pytest.approx(0.5)

    # At speed 5, k_p should be interpolated to 1.5
    pid.update(error=1.0, speed=5.0)
    assert pid.p == pytest.approx(1.5)

  def test_speed_updates_all_gains(self):
    """Test that speed affects all gain properties."""
    k_p = [[0, 10], [2.0, 1.0]]
    k_i = [[0, 10], [0.2, 0.1]]
    k_d = [[0, 10], [0.02, 0.01]]

    pid = PIDController(k_p=k_p, k_i=k_i, k_d=k_d)

    pid.speed = 5.0
    assert pid.k_p == pytest.approx(1.5)
    assert pid.k_i == pytest.approx(0.15)
    assert pid.k_d == pytest.approx(0.015)

  def test_negative_error(self):
    """Test PID response to negative error."""
    pid = PIDController(k_p=2.0, k_i=0.5, k_d=0.1, rate=100)

    control = pid.update(error=-3.0, error_rate=-0.5)

    assert pid.p == pytest.approx(-6.0)
    assert pid.d == pytest.approx(-0.05)
    assert pid.i == pytest.approx(-0.015)
    # Total = -6.0 + -0.015 + -0.05 = -6.065
    assert control == pytest.approx(-6.065)

  def test_zero_error(self):
    """Test PID response to zero error."""
    pid = PIDController(k_p=2.0, k_i=0.5, k_d=0.1, rate=100)

    control = pid.update(error=0.0, error_rate=0.0)

    assert pid.p == 0.0
    assert pid.d == 0.0
    assert pid.i == 0.0
    assert control == 0.0

  def test_control_is_numpy_type(self):
    """Test that control output can be a numpy type due to np.clip."""
    pid = PIDController(k_p=1.0, k_i=0.0)

    control = pid.update(error=5.0)

    # The control value should be numeric (either float or numpy scalar)
    assert isinstance(control, (float, np.floating, int, np.integer))
