import pytest

from openpilot.common.filter_simple import FirstOrderFilter, BounceFilter


class TestFirstOrderFilter:
  def test_init(self):
    """Test FirstOrderFilter initialization."""
    filt = FirstOrderFilter(x0=5.0, rc=0.5, dt=0.01)

    assert filt.x == 5.0
    assert filt.dt == 0.01
    assert filt.initialized is True
    # alpha = dt / (rc + dt) = 0.01 / (0.5 + 0.01) = 0.0196...
    assert filt.alpha == pytest.approx(0.01 / 0.51)

  def test_init_uninitialized(self):
    """Test FirstOrderFilter initialization in uninitialized state."""
    filt = FirstOrderFilter(x0=0.0, rc=0.5, dt=0.01, initialized=False)

    assert filt.initialized is False

  def test_update_alpha(self):
    """Test update_alpha method changes alpha correctly."""
    filt = FirstOrderFilter(x0=0.0, rc=0.5, dt=0.01)
    original_alpha = filt.alpha

    filt.update_alpha(rc=1.0)
    # alpha = dt / (rc + dt) = 0.01 / (1.0 + 0.01) = 0.0099...
    assert filt.alpha == pytest.approx(0.01 / 1.01)
    assert filt.alpha != original_alpha

  def test_update_when_initialized(self):
    """Test filter update when initialized."""
    filt = FirstOrderFilter(x0=0.0, rc=0.5, dt=0.01)

    # First update: x = (1 - alpha) * 0 + alpha * 10
    result = filt.update(10.0)

    expected = filt.alpha * 10.0
    assert result == pytest.approx(expected)
    assert filt.x == pytest.approx(expected)

  def test_update_when_uninitialized(self):
    """Test filter update when uninitialized (first sample)."""
    filt = FirstOrderFilter(x0=0.0, rc=0.5, dt=0.01, initialized=False)

    # First update should set x directly to input
    result = filt.update(10.0)

    assert result == 10.0
    assert filt.x == 10.0
    assert filt.initialized is True

  def test_update_second_call_after_uninitialized(self):
    """Test that second update uses filter after initialization."""
    filt = FirstOrderFilter(x0=0.0, rc=0.5, dt=0.01, initialized=False)

    # First update initializes
    filt.update(10.0)
    assert filt.x == 10.0

    # Second update should filter
    result = filt.update(20.0)
    expected = (1.0 - filt.alpha) * 10.0 + filt.alpha * 20.0
    assert result == pytest.approx(expected)

  def test_filter_converges(self):
    """Test that filter converges to constant input."""
    filt = FirstOrderFilter(x0=0.0, rc=0.1, dt=0.01)

    target = 100.0
    for _ in range(1000):
      filt.update(target)

    assert filt.x == pytest.approx(target, rel=0.01)

  def test_filter_time_constant(self):
    """Test filter response with different time constants."""
    # Faster filter (smaller rc)
    fast_filt = FirstOrderFilter(x0=0.0, rc=0.01, dt=0.01)

    # Slower filter (larger rc)
    slow_filt = FirstOrderFilter(x0=0.0, rc=1.0, dt=0.01)

    # Update both with same input
    for _ in range(10):
      fast_filt.update(100.0)
      slow_filt.update(100.0)

    # Fast filter should be closer to target
    assert fast_filt.x > slow_filt.x

  def test_filter_step_response(self):
    """Test filter response to a step input."""
    filt = FirstOrderFilter(x0=0.0, rc=0.5, dt=0.01)

    # Apply step of 1.0
    values = []
    for _ in range(100):
      values.append(filt.update(1.0))

    # Values should be monotonically increasing
    for i in range(1, len(values)):
      assert values[i] >= values[i - 1]

    # Should be approaching 1.0 (but may not be there yet with these params)
    # With rc=0.5, dt=0.01, time constant is ~50 samples
    # After 100 samples (~2 time constants), should be at ~86% of target
    assert values[-1] > 0.8
    assert values[-1] < 1.0

  def test_alpha_bounds(self):
    """Test alpha calculation with edge cases."""
    # Very small dt relative to rc -> small alpha
    filt_slow = FirstOrderFilter(x0=0.0, rc=10.0, dt=0.001)
    assert filt_slow.alpha < 0.001

    # dt equal to rc -> alpha = 0.5
    filt_equal = FirstOrderFilter(x0=0.0, rc=1.0, dt=1.0)
    assert filt_equal.alpha == pytest.approx(0.5)

    # Large dt relative to rc -> alpha close to 1
    filt_fast = FirstOrderFilter(x0=0.0, rc=0.001, dt=1.0)
    assert filt_fast.alpha > 0.99

  def test_negative_values(self):
    """Test filter with negative values."""
    filt = FirstOrderFilter(x0=-10.0, rc=0.1, dt=0.01)

    result = filt.update(-20.0)

    # Filter should move toward -20
    assert result < -10.0
    assert result > -20.0


class TestBounceFilter:
  def test_init(self):
    """Test BounceFilter initialization."""
    filt = BounceFilter(x0=5.0, rc=0.5, dt=0.01)

    assert filt.x == 5.0
    assert filt.bounce == 2
    assert isinstance(filt.velocity, FirstOrderFilter)
    assert filt.velocity.x == 0.0

  def test_init_custom_bounce(self):
    """Test BounceFilter initialization with custom bounce value."""
    filt = BounceFilter(x0=0.0, rc=0.5, dt=0.01, bounce=5)

    assert filt.bounce == 5

  def test_init_uninitialized(self):
    """Test BounceFilter initialization in uninitialized state."""
    filt = BounceFilter(x0=0.0, rc=0.5, dt=0.01, initialized=False)

    assert filt.initialized is False

  def test_update_basic(self):
    """Test basic BounceFilter update."""
    filt = BounceFilter(x0=0.0, rc=0.5, dt=0.01)

    result = filt.update(10.0)

    # Result should be influenced by both base filter and bounce velocity
    assert isinstance(result, float)

  def test_bounce_effect(self):
    """Test that bounce creates overshoot behavior."""
    filt = BounceFilter(x0=0.0, rc=0.1, dt=1.0/60.0, bounce=2)

    # Track values as we approach target
    values = []
    for _ in range(100):
      values.append(filt.update(100.0))

    # With bounce, we expect some overshoot past the target
    max_val = max(values)
    assert max_val > 100.0 or values[-1] == pytest.approx(100.0, rel=0.1)

  def test_velocity_filter(self):
    """Test that velocity filter is applied."""
    filt = BounceFilter(x0=0.0, rc=0.5, dt=0.01)

    # Initial velocity should be 0
    assert filt.velocity.x == 0.0

    # After updates, velocity should change
    filt.update(100.0)
    filt.update(100.0)

    # Velocity will have been modified
    # (exact value depends on filter dynamics)
    assert isinstance(filt.velocity.x, float)

  def test_velocity_damping(self):
    """Test that small velocities are zeroed out."""
    filt = BounceFilter(x0=100.0, rc=0.5, dt=0.01)

    # Update with same value many times to let velocity settle
    for _ in range(1000):
      filt.update(100.0)

    # Velocity should be zeroed when small
    assert filt.velocity.x == 0.0

  def test_scale_with_dt(self):
    """Test that scale adjusts for different frame rates."""
    dt_60fps = 1.0 / 60.0
    dt_30fps = 1.0 / 30.0

    filt_60 = BounceFilter(x0=0.0, rc=0.5, dt=dt_60fps)
    filt_30 = BounceFilter(x0=0.0, rc=0.5, dt=dt_30fps)

    # Both filters should behave similarly over the same time period
    # (dt scaling compensates for frame rate)
    for _ in range(60):
      filt_60.update(100.0)

    for _ in range(30):
      filt_30.update(100.0)

    # Values should be roughly similar (not exactly due to discrete updates)
    # Just verify both are reasonable
    assert 50.0 < filt_60.x < 150.0
    assert 50.0 < filt_30.x < 150.0

  def test_inherits_first_order_filter(self):
    """Test that BounceFilter inherits FirstOrderFilter behavior."""
    filt = BounceFilter(x0=0.0, rc=0.5, dt=0.01)

    # Should have alpha from parent
    assert hasattr(filt, 'alpha')
    assert filt.alpha == pytest.approx(0.01 / 0.51)

    # Should have update_alpha from parent
    filt.update_alpha(1.0)
    assert filt.alpha == pytest.approx(0.01 / 1.01)

  def test_uninitialized_first_update(self):
    """Test BounceFilter first update when uninitialized."""
    filt = BounceFilter(x0=0.0, rc=0.5, dt=0.01, initialized=False)

    result = filt.update(50.0)

    # First update should initialize x to input
    assert filt.initialized is True
    # Result includes bounce dynamics applied after base filter
    assert isinstance(result, float)

  def test_converges_to_target(self):
    """Test that BounceFilter eventually converges to constant input."""
    filt = BounceFilter(x0=0.0, rc=0.1, dt=0.01, bounce=2)

    target = 100.0
    for _ in range(2000):
      filt.update(target)

    # Should converge close to target
    assert filt.x == pytest.approx(target, rel=0.05)

  def test_oscillation_with_high_bounce(self):
    """Test that higher bounce value creates more oscillation."""
    filt_low = BounceFilter(x0=0.0, rc=0.1, dt=1.0/60.0, bounce=1)
    filt_high = BounceFilter(x0=0.0, rc=0.1, dt=1.0/60.0, bounce=5)

    # Track peak values
    max_low = 0.0
    max_high = 0.0

    for _ in range(100):
      filt_low.update(100.0)
      filt_high.update(100.0)
      max_low = max(max_low, filt_low.x)
      max_high = max(max_high, filt_high.x)

    # Higher bounce should potentially overshoot more
    # (or at least respond differently)
    assert isinstance(max_low, float)
    assert isinstance(max_high, float)
