"""
Tests for vehicle dynamics module.

Tests cover:
- VehicleDynamicsConfig validation and properties
- BicycleModel kinematics and dynamics
- Vehicle presets (sedan, SUV, truck, compact, sports)
- Edge cases and error handling
"""

import math
import pytest

from openpilot.selfdrive.controls.lib.tests.algorithm_harness.vehicle_dynamics import (
  VehicleDynamicsConfig,
  VehicleType,
  BicycleModel,
  BicycleModelState,
  get_sedan_config,
  get_suv_config,
  get_truck_config,
  get_compact_config,
  get_sports_config,
  get_vehicle_config,
  VEHICLE_PRESETS,
)


class TestVehicleDynamicsConfig:
  """Tests for VehicleDynamicsConfig dataclass."""

  def test_default_values(self):
    """Test default configuration values."""
    config = VehicleDynamicsConfig()
    assert config.wheelbase == 2.7
    assert config.steer_ratio == 15.0
    assert config.mass == 1500.0
    assert config.vehicle_type == VehicleType.SEDAN

  def test_custom_values(self):
    """Test creating config with custom values."""
    config = VehicleDynamicsConfig(
      name="Test Vehicle",
      wheelbase=3.0,
      steer_ratio=12.0,
      mass=2000.0,
    )
    assert config.name == "Test Vehicle"
    assert config.wheelbase == 3.0
    assert config.steer_ratio == 12.0
    assert config.mass == 2000.0

  def test_center_to_rear_property(self):
    """Test center_to_rear computed property."""
    config = VehicleDynamicsConfig(wheelbase=3.0, center_to_front=1.2)
    assert config.center_to_rear == 1.8

  def test_max_road_wheel_angle_deg(self):
    """Test max_road_wheel_angle_deg computed property."""
    config = VehicleDynamicsConfig(max_steer_angle_deg=540.0, steer_ratio=15.0)
    assert config.max_road_wheel_angle_deg == 36.0

  def test_understeer_gradient(self):
    """Test understeer gradient calculation."""
    config = VehicleDynamicsConfig()
    # Should be positive for typical understeering vehicle
    assert config.understeer_gradient > 0

  def test_invalid_wheelbase_raises(self):
    """Test that negative wheelbase raises ValueError."""
    with pytest.raises(ValueError, match="Wheelbase must be positive"):
      VehicleDynamicsConfig(wheelbase=-1.0)

  def test_zero_wheelbase_raises(self):
    """Test that zero wheelbase raises ValueError."""
    with pytest.raises(ValueError, match="Wheelbase must be positive"):
      VehicleDynamicsConfig(wheelbase=0.0)

  def test_invalid_steer_ratio_raises(self):
    """Test that non-positive steer ratio raises ValueError."""
    with pytest.raises(ValueError, match="Steer ratio must be positive"):
      VehicleDynamicsConfig(steer_ratio=-5.0)

  def test_invalid_mass_raises(self):
    """Test that non-positive mass raises ValueError."""
    with pytest.raises(ValueError, match="Mass must be positive"):
      VehicleDynamicsConfig(mass=0.0)

  def test_invalid_center_to_front_raises(self):
    """Test that center_to_front outside valid range raises."""
    with pytest.raises(ValueError, match="Center to front must be between"):
      VehicleDynamicsConfig(wheelbase=2.7, center_to_front=3.0)

  def test_negative_center_to_front_raises(self):
    """Test that negative center_to_front raises."""
    with pytest.raises(ValueError, match="Center to front must be between"):
      VehicleDynamicsConfig(center_to_front=-1.0)


class TestBicycleModelState:
  """Tests for BicycleModelState dataclass."""

  def test_default_state(self):
    """Test default state values."""
    state = BicycleModelState()
    assert state.x == 0.0
    assert state.y == 0.0
    assert state.yaw == 0.0
    assert state.v == 0.0
    assert state.yaw_rate == 0.0

  def test_custom_state(self):
    """Test creating state with custom values."""
    state = BicycleModelState(x=10.0, y=5.0, yaw=math.pi/4, v=20.0)
    assert state.x == 10.0
    assert state.y == 5.0
    assert state.yaw == math.pi / 4
    assert state.v == 20.0


class TestBicycleModel:
  """Tests for BicycleModel kinematics."""

  @pytest.fixture
  def model(self):
    """Create a bicycle model with default config."""
    config = VehicleDynamicsConfig()
    return BicycleModel(config)

  def test_initialization(self, model):
    """Test model initialization."""
    assert model.state.x == 0.0
    assert model.state.v == 0.0

  def test_reset(self, model):
    """Test model reset."""
    model.state.x = 100.0
    model.state.v = 20.0
    model.reset()
    assert model.state.x == 0.0
    assert model.state.v == 0.0

  def test_reset_with_initial_state(self, model):
    """Test reset with custom initial state."""
    initial = BicycleModelState(x=50.0, y=25.0, v=15.0)
    model.reset(initial)
    assert model.state.x == 50.0
    assert model.state.y == 25.0
    assert model.state.v == 15.0

  def test_curvature_to_steer_angle(self, model):
    """Test curvature to steering angle conversion."""
    # Zero curvature = zero steer
    assert model.curvature_to_steer_angle(0.0) == 0.0

    # Positive curvature = positive steer (left turn)
    steer = model.curvature_to_steer_angle(0.01)
    assert steer > 0

  def test_steer_angle_to_curvature(self, model):
    """Test steering angle to curvature conversion."""
    # Zero steer = zero curvature
    assert model.steer_angle_to_curvature(0.0) == 0.0

    # Round-trip should be close to original
    original_curvature = 0.01
    steer = model.curvature_to_steer_angle(original_curvature)
    recovered = model.steer_angle_to_curvature(steer)
    assert abs(recovered - original_curvature) < 1e-6

  def test_yaw_rate_from_curvature(self, model):
    """Test yaw rate calculation from curvature."""
    # yaw_rate = curvature * v
    curvature = 0.01  # 1/100m radius
    v = 20.0  # m/s
    yaw_rate = model.yaw_rate_from_curvature(curvature, v)
    assert abs(yaw_rate - 0.2) < 1e-6

  def test_lateral_accel_from_curvature(self, model):
    """Test lateral acceleration calculation."""
    # a_lat = curvature * v^2
    curvature = 0.01
    v = 20.0
    a_lat = model.lateral_accel_from_curvature(curvature, v)
    assert abs(a_lat - 4.0) < 1e-6

  def test_max_curvature_at_low_speed(self, model):
    """Test max curvature at low speed is limited by steering."""
    max_curv = model.max_curvature_at_speed(0.05)
    # Should be limited by max steering angle
    max_road_angle = math.radians(model.config.max_road_wheel_angle_deg)
    expected = math.tan(max_road_angle) / model.config.wheelbase
    assert abs(max_curv - expected) < 1e-6

  def test_max_curvature_at_high_speed(self, model):
    """Test max curvature at high speed is limited by lateral accel."""
    max_curv = model.max_curvature_at_speed(30.0)
    # Should be limited by lateral acceleration
    expected = model.config.max_lateral_accel / (30.0 * 30.0)
    assert max_curv <= expected + 1e-6

  def test_step_straight_line(self, model):
    """Test stepping model in a straight line."""
    model.state.v = 10.0  # Set initial velocity
    state = model.step(dt=0.1, steer_cmd=0.0, accel_cmd=0.0)

    # Should move forward
    assert state.x > 0
    assert abs(state.y) < 1e-6  # No lateral movement
    assert abs(state.yaw) < 1e-6  # No heading change

  def test_step_acceleration(self, model):
    """Test model acceleration."""
    initial_v = model.state.v
    model.step(dt=0.1, steer_cmd=0.0, accel_cmd=2.0)

    # Velocity should increase
    assert model.state.v > initial_v

  def test_step_deceleration(self, model):
    """Test model deceleration."""
    model.state.v = 20.0
    model.step(dt=0.1, steer_cmd=0.0, accel_cmd=-2.0)

    # Velocity should decrease
    assert model.state.v < 20.0

  def test_step_velocity_clamps_to_zero(self, model):
    """Test that velocity doesn't go negative."""
    model.state.v = 1.0
    model.step(dt=1.0, steer_cmd=0.0, accel_cmd=-10.0)

    # Velocity should be clamped to 0
    assert model.state.v >= 0

  def test_step_turning(self, model):
    """Test model turning behavior."""
    model.state.v = 10.0
    initial_yaw = model.state.yaw

    # Turn left
    for _ in range(10):
      model.step(dt=0.1, steer_cmd=0.5, accel_cmd=0.0)

    # Heading should have changed (left turn = positive yaw rate in our convention)
    assert model.state.yaw != initial_yaw
    assert model.state.y != 0  # Lateral position changed

  def test_step_at_standstill(self, model):
    """Test stepping at standstill doesn't move."""
    model.state.v = 0.0
    model.step(dt=0.1, steer_cmd=0.5, accel_cmd=0.0)

    assert model.state.x == 0.0
    assert model.state.y == 0.0
    assert model.state.yaw_rate == 0.0


class TestVehiclePresets:
  """Tests for vehicle preset configurations."""

  def test_sedan_config(self):
    """Test sedan preset configuration."""
    config = get_sedan_config()
    assert config.vehicle_type == VehicleType.SEDAN
    assert 2.5 < config.wheelbase < 3.0
    assert 1400 < config.mass < 1700

  def test_suv_config(self):
    """Test SUV preset configuration."""
    config = get_suv_config()
    assert config.vehicle_type == VehicleType.SUV
    assert config.mass > get_sedan_config().mass  # SUVs are heavier
    assert config.wheelbase > get_sedan_config().wheelbase  # SUVs are longer

  def test_truck_config(self):
    """Test truck preset configuration."""
    config = get_truck_config()
    assert config.vehicle_type == VehicleType.TRUCK
    assert config.mass > get_suv_config().mass  # Trucks are heaviest
    assert config.wheelbase > get_suv_config().wheelbase  # Trucks are longest

  def test_compact_config(self):
    """Test compact car preset configuration."""
    config = get_compact_config()
    assert config.vehicle_type == VehicleType.COMPACT
    assert config.mass < get_sedan_config().mass  # Compacts are lighter
    assert config.wheelbase < get_sedan_config().wheelbase  # Compacts are shorter

  def test_sports_config(self):
    """Test sports car preset configuration."""
    config = get_sports_config()
    assert config.vehicle_type == VehicleType.SPORTS
    assert config.steer_ratio < get_sedan_config().steer_ratio  # More direct steering
    assert config.max_lateral_accel > get_sedan_config().max_lateral_accel  # Better handling

  def test_get_vehicle_config_by_type(self):
    """Test getting config by VehicleType enum."""
    config = get_vehicle_config(VehicleType.SUV)
    assert config.vehicle_type == VehicleType.SUV

  def test_get_vehicle_config_with_custom_name(self):
    """Test getting config with custom name."""
    config = get_vehicle_config(VehicleType.SEDAN, name="My Custom Sedan")
    assert config.name == "My Custom Sedan"

  def test_get_vehicle_config_unknown_type(self):
    """Test that unknown type returns sedan as default."""
    config = get_vehicle_config(VehicleType.CUSTOM)
    assert config.vehicle_type == VehicleType.SEDAN

  def test_vehicle_presets_dict(self):
    """Test VEHICLE_PRESETS dictionary."""
    assert "sedan" in VEHICLE_PRESETS
    assert "suv" in VEHICLE_PRESETS
    assert "truck" in VEHICLE_PRESETS
    assert "compact" in VEHICLE_PRESETS
    assert "sports" in VEHICLE_PRESETS

  def test_all_presets_valid(self):
    """Test all presets have valid configurations."""
    for name, config in VEHICLE_PRESETS.items():
      assert config.wheelbase > 0
      assert config.steer_ratio > 0
      assert config.mass > 0
      # Verify center_to_front is within wheelbase
      assert 0 < config.center_to_front < config.wheelbase


class TestBicycleModelIntegration:
  """Integration tests for bicycle model simulation."""

  def test_circle_trajectory(self):
    """Test driving in a circle."""
    config = get_sedan_config()
    model = BicycleModel(config)
    model.state.v = 10.0  # Constant speed

    initial_x = model.state.x
    initial_y = model.state.y

    # Drive in a circle for many steps
    for _ in range(1000):
      model.step(dt=0.01, steer_cmd=0.3, accel_cmd=0.0)

    # Should be roughly back near start (circular motion)
    # This is approximate - depends on exact steering
    final_distance = math.sqrt(model.state.x**2 + model.state.y**2)
    # Just verify we've moved and are on some curved path
    assert final_distance > 0

  def test_s_curve_trajectory(self):
    """Test S-curve maneuver."""
    config = get_sedan_config()
    model = BicycleModel(config)
    model.state.v = 15.0

    # First half: turn right
    for _ in range(200):
      model.step(dt=0.01, steer_cmd=-0.2, accel_cmd=0.0)

    mid_y = model.state.y

    # Second half: turn left
    for _ in range(200):
      model.step(dt=0.01, steer_cmd=0.2, accel_cmd=0.0)

    # Should have turned back towards center
    final_y = model.state.y
    assert final_y > mid_y  # Turned back

  def test_acceleration_from_stop(self):
    """Test accelerating from standstill."""
    config = get_sedan_config()
    model = BicycleModel(config)

    # Accelerate from stop
    for _ in range(100):
      model.step(dt=0.1, steer_cmd=0.0, accel_cmd=2.0)

    # Should have moved forward significantly
    assert model.state.x > 50
    assert model.state.v > 10

  def test_emergency_stop(self):
    """Test emergency braking."""
    config = get_sedan_config()
    model = BicycleModel(config)
    model.state.v = 25.0
    model.state.x = 0.0

    # Emergency brake
    steps = 0
    while model.state.v > 0.1 and steps < 1000:
      model.step(dt=0.01, steer_cmd=0.0, accel_cmd=-config.max_longitudinal_decel)
      steps += 1

    # Should have stopped
    assert model.state.v < 1.0
    # Calculate stopping distance
    stopping_distance = model.state.x
    # Theoretical: d = v^2 / (2*a) = 625 / 8 ~= 78m
    assert 50 < stopping_distance < 120  # Reasonable stopping distance


class TestVehicleDynamicsEdgeCases:
  """Edge case tests for vehicle dynamics."""

  def test_very_small_steer_angle(self):
    """Test behavior with very small steering angle."""
    config = get_sedan_config()
    model = BicycleModel(config)
    model.state.v = 20.0

    # Very small steer - should be nearly straight
    for _ in range(100):
      model.step(dt=0.01, steer_cmd=0.001, accel_cmd=0.0)

    # Should have moved mostly straight
    assert abs(model.state.y) < 1.0

  def test_maximum_steer_angle(self):
    """Test behavior at maximum steering angle."""
    config = get_sedan_config()
    model = BicycleModel(config)
    model.state.v = 5.0  # Low speed for tight turn

    # Full steer
    for _ in range(100):
      model.step(dt=0.01, steer_cmd=1.0, accel_cmd=0.0)

    # Should have turned significantly
    assert abs(model.state.yaw) > math.pi / 4

  def test_very_high_speed(self):
    """Test behavior at very high speed."""
    config = get_sports_config()
    model = BicycleModel(config)
    model.state.v = 50.0  # ~180 km/h

    # Try to turn at high speed
    for _ in range(100):
      model.step(dt=0.01, steer_cmd=0.1, accel_cmd=0.0)

    # Should still work (no numerical issues)
    assert not math.isnan(model.state.x)
    assert not math.isnan(model.state.y)
    assert not math.isnan(model.state.yaw)

  def test_accel_clamping_max(self):
    """Test acceleration is clamped to max."""
    config = get_sedan_config()
    model = BicycleModel(config)

    model.step(dt=1.0, steer_cmd=0.0, accel_cmd=100.0)  # Unrealistic accel

    # Should be clamped to max
    expected_v = config.max_longitudinal_accel * 1.0
    assert abs(model.state.v - expected_v) < 0.1

  def test_decel_clamping_max(self):
    """Test deceleration is clamped to max."""
    config = get_sedan_config()
    model = BicycleModel(config)
    model.state.v = 30.0

    model.step(dt=1.0, steer_cmd=0.0, accel_cmd=-100.0)  # Unrealistic decel

    # Should be clamped to max decel
    expected_v = 30.0 - config.max_longitudinal_decel * 1.0
    assert abs(model.state.v - expected_v) < 0.1


@pytest.mark.algorithm_benchmark
class TestVehicleDynamicsBenchmark:
  """Benchmark tests for vehicle dynamics simulation."""

  def test_long_simulation_stability(self):
    """Test model stability over long simulation."""
    config = get_sedan_config()
    model = BicycleModel(config)
    model.state.v = 20.0

    # Run for 10 simulated minutes at 100Hz
    for _ in range(60000):
      steer = 0.1 * math.sin(model.state.x / 100)  # Gentle S-curve
      model.step(dt=0.01, steer_cmd=steer, accel_cmd=0.0)

    # Should still have reasonable values
    assert not math.isnan(model.state.x)
    assert not math.isinf(model.state.x)
    assert abs(model.state.v - 20.0) < 1.0  # Speed maintained
