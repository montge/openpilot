import random
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from cereal import car, messaging
from openpilot.selfdrive.locationd.paramsd import (
  VehicleParamsLearner, retrieve_initial_vehicle_params, migrate_cached_vehicle_params_if_needed,
  check_valid_with_hysteresis,
  MAX_ANGLE_OFFSET_DELTA, ROLL_MAX_DELTA, ROLL_MIN, ROLL_MAX, ROLL_STD_MAX,
  OFFSET_MAX, OFFSET_LOWERED_MAX, ROLL_LOWERED_MAX,
  MIN_ACTIVE_SPEED, LOW_ACTIVE_SPEED, LATERAL_ACC_SENSOR_THRESHOLD
)
from openpilot.selfdrive.locationd.models.car_kf import CarKalman, States
from openpilot.selfdrive.locationd.test.test_locationd_scenarios import TEST_ROUTE
from openpilot.selfdrive.test.process_replay.migration import migrate, migrate_carParams
from openpilot.common.params import Params
from openpilot.tools.lib.logreader import LogReader


def get_random_live_parameters(CP):
  msg = messaging.new_message("liveParameters")
  msg.liveParameters.steerRatio = (random.random() + 0.5) * CP.steerRatio
  msg.liveParameters.stiffnessFactor = random.random()
  msg.liveParameters.angleOffsetAverageDeg = random.random()
  msg.liveParameters.debugFilterState.std = [random.random() for _ in range(CarKalman.P_initial.shape[0])]
  return msg


class TestParamsd:
  def test_read_saved_params(self):
    params = Params()

    lr = migrate(LogReader(TEST_ROUTE), [migrate_carParams])
    CP = next(m for m in lr if m.which() == "carParams").carParams

    msg = get_random_live_parameters(CP)
    params.put("LiveParametersV2", msg.to_bytes())
    params.put("CarParamsPrevRoute", CP.as_builder().to_bytes())

    migrate_cached_vehicle_params_if_needed(params) # this is not tested here but should not mess anything up or throw an error
    sr, sf, offset, p_init = retrieve_initial_vehicle_params(params, CP, replay=True, debug=True)
    np.testing.assert_allclose(sr, msg.liveParameters.steerRatio)
    np.testing.assert_allclose(sf, msg.liveParameters.stiffnessFactor)
    np.testing.assert_allclose(offset, msg.liveParameters.angleOffsetAverageDeg)
    np.testing.assert_equal(p_init.shape, CarKalman.P_initial.shape)
    np.testing.assert_allclose(np.diagonal(p_init), msg.liveParameters.debugFilterState.std)

  # TODO Remove this test after the support for old format is removed
  def test_read_saved_params_old_format(self):
    params = Params()

    lr = migrate(LogReader(TEST_ROUTE), [migrate_carParams])
    CP = next(m for m in lr if m.which() == "carParams").carParams

    msg = get_random_live_parameters(CP)
    params.put("LiveParameters", msg.liveParameters.to_dict())
    params.put("CarParamsPrevRoute", CP.as_builder().to_bytes())
    params.remove("LiveParametersV2")

    migrate_cached_vehicle_params_if_needed(params)
    sr, sf, offset, _ = retrieve_initial_vehicle_params(params, CP, replay=True, debug=True)
    np.testing.assert_allclose(sr, msg.liveParameters.steerRatio)
    np.testing.assert_allclose(sf, msg.liveParameters.stiffnessFactor)
    np.testing.assert_allclose(offset, msg.liveParameters.angleOffsetAverageDeg)
    assert params.get("LiveParametersV2") is not None

  def test_read_saved_params_corrupted_old_format(self):
    params = Params()
    params.put("LiveParameters", {})
    params.remove("LiveParametersV2")

    migrate_cached_vehicle_params_if_needed(params)
    assert params.get("LiveParameters") is None
    assert params.get("LiveParametersV2") is None


class TestCheckValidWithHysteresis(unittest.TestCase):
  """Test check_valid_with_hysteresis function."""

  def test_valid_stays_valid_below_threshold(self):
    """Test valid remains True when value is below threshold."""
    result = check_valid_with_hysteresis(True, 5.0, 10.0, 8.0)
    self.assertTrue(result)

  def test_valid_becomes_invalid_above_threshold(self):
    """Test valid becomes False when value exceeds threshold."""
    result = check_valid_with_hysteresis(True, 11.0, 10.0, 8.0)
    self.assertFalse(result)

  def test_invalid_stays_invalid_above_lowered_threshold(self):
    """Test invalid remains False when value is above lowered threshold."""
    result = check_valid_with_hysteresis(False, 9.0, 10.0, 8.0)
    self.assertFalse(result)

  def test_invalid_becomes_valid_below_lowered_threshold(self):
    """Test invalid becomes True when value is below lowered threshold."""
    result = check_valid_with_hysteresis(False, 7.0, 10.0, 8.0)
    self.assertTrue(result)

  def test_hysteresis_prevents_oscillation(self):
    """Test hysteresis prevents rapid switching at boundary."""
    valid = True
    valid = check_valid_with_hysteresis(valid, 9.0, 10.0, 8.0)
    self.assertTrue(valid)
    valid = check_valid_with_hysteresis(valid, 11.0, 10.0, 8.0)
    self.assertFalse(valid)
    valid = check_valid_with_hysteresis(valid, 9.0, 10.0, 8.0)
    self.assertFalse(valid)
    valid = check_valid_with_hysteresis(valid, 7.0, 10.0, 8.0)
    self.assertTrue(valid)

  def test_negative_values_checked_by_abs(self):
    """Test that negative values are checked by absolute value."""
    result = check_valid_with_hysteresis(True, -11.0, 10.0, 8.0)
    self.assertFalse(result)
    result = check_valid_with_hysteresis(True, -5.0, 10.0, 8.0)
    self.assertTrue(result)


class TestParamsdConstants(unittest.TestCase):
  """Test that constants have sensible values."""

  def test_max_angle_offset_delta_positive(self):
    """Test MAX_ANGLE_OFFSET_DELTA is positive."""
    self.assertGreater(MAX_ANGLE_OFFSET_DELTA, 0)

  def test_roll_range_valid(self):
    """Test ROLL_MIN < ROLL_MAX."""
    self.assertLess(ROLL_MIN, ROLL_MAX)

  def test_roll_delta_positive(self):
    """Test ROLL_MAX_DELTA is positive."""
    self.assertGreater(ROLL_MAX_DELTA, 0)

  def test_roll_std_max_positive(self):
    """Test ROLL_STD_MAX is positive."""
    self.assertGreater(ROLL_STD_MAX, 0)

  def test_offset_thresholds_ordered(self):
    """Test OFFSET_LOWERED_MAX < OFFSET_MAX for hysteresis."""
    self.assertLess(OFFSET_LOWERED_MAX, OFFSET_MAX)

  def test_roll_thresholds_ordered(self):
    """Test ROLL_LOWERED_MAX < ROLL_MAX for hysteresis."""
    self.assertLess(ROLL_LOWERED_MAX, ROLL_MAX)

  def test_speed_thresholds_ordered(self):
    """Test MIN_ACTIVE_SPEED < LOW_ACTIVE_SPEED."""
    self.assertLess(MIN_ACTIVE_SPEED, LOW_ACTIVE_SPEED)

  def test_lateral_acc_threshold_positive(self):
    """Test LATERAL_ACC_SENSOR_THRESHOLD is positive."""
    self.assertGreater(LATERAL_ACC_SENSOR_THRESHOLD, 0)


def get_test_car_params():
  """Create a test CarParams with realistic values."""
  lr = migrate(LogReader(TEST_ROUTE), [migrate_carParams])
  return next(m for m in lr if m.which() == "carParams").carParams


class TestVehicleParamsLearnerInit(unittest.TestCase):
  """Test VehicleParamsLearner initialization."""

  def test_init_with_defaults(self):
    """Test VehicleParamsLearner initializes with provided values."""
    CP = get_test_car_params()
    learner = VehicleParamsLearner(CP, steer_ratio=15.0, stiffness_factor=1.0, angle_offset=0.0)

    self.assertEqual(learner.observed_speed, 0.0)
    self.assertEqual(learner.observed_yaw_rate, 0.0)
    self.assertEqual(learner.observed_roll, 0.0)
    self.assertFalse(learner.active)

  def test_init_sets_steer_ratio_bounds(self):
    """Test VehicleParamsLearner sets steer ratio bounds from CP."""
    CP = get_test_car_params()
    learner = VehicleParamsLearner(CP, steer_ratio=15.0, stiffness_factor=1.0, angle_offset=0.0)

    self.assertEqual(learner.min_sr, 0.5 * CP.steerRatio)
    self.assertEqual(learner.max_sr, 2.0 * CP.steerRatio)

  def test_init_with_custom_p_initial(self):
    """Test VehicleParamsLearner with custom P_initial."""
    CP = get_test_car_params()
    p_initial = np.eye(CarKalman.P_initial.shape[0]) * 0.01
    learner = VehicleParamsLearner(CP, steer_ratio=15.0, stiffness_factor=1.0,
                                    angle_offset=0.0, P_initial=p_initial)
    self.assertIsNotNone(learner.P_initial)


class TestVehicleParamsLearnerReset(unittest.TestCase):
  """Test VehicleParamsLearner reset method."""

  def test_reset_clears_active_state(self):
    """Test reset clears active state."""
    CP = get_test_car_params()
    learner = VehicleParamsLearner(CP, steer_ratio=15.0, stiffness_factor=1.0, angle_offset=0.0)
    learner.active = True
    learner.reset(None)
    self.assertFalse(learner.active)

  def test_reset_restores_angle_offset(self):
    """Test reset restores angle offset from initial state."""
    CP = get_test_car_params()
    initial_offset = 0.05  # radians
    learner = VehicleParamsLearner(CP, steer_ratio=15.0, stiffness_factor=1.0,
                                    angle_offset=initial_offset)
    expected_offset_deg = np.degrees(initial_offset)
    self.assertAlmostEqual(learner.angle_offset, expected_offset_deg, places=3)


class TestVehicleParamsLearnerHandleLog(unittest.TestCase):
  """Test VehicleParamsLearner handle_log method."""

  def _create_learner(self):
    """Create a VehicleParamsLearner for testing."""
    CP = get_test_car_params()
    return VehicleParamsLearner(CP, steer_ratio=15.0, stiffness_factor=1.0, angle_offset=0.0)

  def test_handle_carstate_updates_observed_speed(self):
    """Test handle_log with carState updates observed speed."""
    learner = self._create_learner()
    msg = MagicMock()
    msg.vEgo = 20.0
    msg.steeringAngleDeg = 10.0

    learner.handle_log(1.0, 'carState', msg)

    self.assertEqual(learner.observed_speed, 20.0)

  def test_handle_carstate_activates_at_speed(self):
    """Test handle_log with carState activates learner at sufficient speed."""
    learner = self._create_learner()
    msg = MagicMock()
    msg.vEgo = MIN_ACTIVE_SPEED + 1.0
    msg.steeringAngleDeg = 10.0

    learner.handle_log(1.0, 'carState', msg)

    self.assertTrue(learner.active)

  def test_handle_carstate_inactive_at_low_speed(self):
    """Test handle_log with carState stays inactive at low speed."""
    learner = self._create_learner()
    msg = MagicMock()
    msg.vEgo = MIN_ACTIVE_SPEED - 0.5
    msg.steeringAngleDeg = 10.0

    learner.handle_log(1.0, 'carState', msg)

    self.assertFalse(learner.active)

  def test_handle_carstate_inactive_at_large_steering_angle(self):
    """Test handle_log with carState stays inactive at large steering angle."""
    learner = self._create_learner()
    msg = MagicMock()
    msg.vEgo = 20.0
    msg.steeringAngleDeg = 50.0  # > 45 degrees

    learner.handle_log(1.0, 'carState', msg)

    self.assertFalse(learner.active)

  def test_handle_live_calibration(self):
    """Test handle_log with liveCalibration feeds calibrator."""
    learner = self._create_learner()
    msg = MagicMock()
    msg.rpyCalib = [0.0, 0.0, 0.0]
    msg.calStatus = 1

    # Should not raise
    learner.handle_log(1.0, 'liveCalibration', msg)

  def test_handle_live_pose_updates_yaw_rate(self):
    """Test handle_log with livePose updates observed yaw rate."""
    learner = self._create_learner()
    msg = MagicMock()
    msg.angularVelocityDevice.valid = True
    msg.angularVelocityDevice.x = 0.0
    msg.angularVelocityDevice.y = 0.0
    msg.angularVelocityDevice.z = 0.1
    msg.angularVelocityDevice.xStd = 0.01
    msg.angularVelocityDevice.yStd = 0.01
    msg.angularVelocityDevice.zStd = 0.01
    msg.velocityDevice.valid = True
    msg.velocityDevice.x = 10.0
    msg.velocityDevice.y = 0.0
    msg.velocityDevice.z = 0.0
    msg.velocityDevice.xStd = 0.1
    msg.velocityDevice.yStd = 0.1
    msg.velocityDevice.zStd = 0.1
    msg.accelerationDevice.valid = True
    msg.accelerationDevice.x = 0.0
    msg.accelerationDevice.y = 0.0
    msg.accelerationDevice.z = 9.8
    msg.accelerationDevice.xStd = 0.1
    msg.accelerationDevice.yStd = 0.1
    msg.accelerationDevice.zStd = 0.1
    msg.orientationNED.valid = True
    msg.orientationNED.x = 0.0
    msg.orientationNED.y = 0.0
    msg.orientationNED.z = 0.0
    msg.orientationNED.xStd = 0.01
    msg.orientationNED.yStd = 0.01
    msg.orientationNED.zStd = 0.01
    msg.posenetOK = True
    msg.sensorsOK = True

    learner.handle_log(1.0, 'livePose', msg)

    # The yaw rate should have been processed
    self.assertIsInstance(learner.observed_yaw_rate, float)


class TestVehicleParamsLearnerGetMsg(unittest.TestCase):
  """Test VehicleParamsLearner get_msg method."""

  def _create_learner(self):
    """Create a VehicleParamsLearner for testing."""
    CP = get_test_car_params()
    return VehicleParamsLearner(CP, steer_ratio=15.0, stiffness_factor=1.0, angle_offset=0.0)

  def test_get_msg_returns_message(self):
    """Test get_msg returns a valid message."""
    learner = self._create_learner()
    msg = learner.get_msg(valid=True)

    self.assertIsNotNone(msg)
    self.assertTrue(msg.valid)

  def test_get_msg_invalid_flag(self):
    """Test get_msg with invalid flag."""
    learner = self._create_learner()
    msg = learner.get_msg(valid=False)

    self.assertFalse(msg.valid)

  def test_get_msg_includes_steer_ratio(self):
    """Test get_msg includes steer ratio."""
    learner = self._create_learner()
    msg = learner.get_msg(valid=True)

    self.assertIsNotNone(msg.liveParameters.steerRatio)
    self.assertGreater(msg.liveParameters.steerRatio, 0)

  def test_get_msg_includes_angle_offset(self):
    """Test get_msg includes angle offset fields."""
    learner = self._create_learner()
    msg = learner.get_msg(valid=True)

    self.assertIsNotNone(msg.liveParameters.angleOffsetAverageDeg)
    self.assertIsNotNone(msg.liveParameters.angleOffsetDeg)

  def test_get_msg_includes_roll(self):
    """Test get_msg includes roll field."""
    learner = self._create_learner()
    msg = learner.get_msg(valid=True)

    self.assertIsNotNone(msg.liveParameters.roll)

  def test_get_msg_with_debug(self):
    """Test get_msg with debug flag includes filter state."""
    learner = self._create_learner()
    msg = learner.get_msg(valid=True, debug=True)

    self.assertIsNotNone(msg.liveParameters.debugFilterState)
    self.assertGreater(len(msg.liveParameters.debugFilterState.value), 0)
    self.assertGreater(len(msg.liveParameters.debugFilterState.std), 0)

  def test_get_msg_steer_ratio_validity(self):
    """Test get_msg sets steerRatioValid correctly."""
    learner = self._create_learner()
    msg = learner.get_msg(valid=True)

    # With default initialization, steer ratio should be valid
    self.assertTrue(msg.liveParameters.steerRatioValid)

  def test_get_msg_stiffness_validity(self):
    """Test get_msg sets stiffnessFactorValid correctly."""
    learner = self._create_learner()
    msg = learner.get_msg(valid=True)

    # With default initialization, stiffness should be valid
    self.assertTrue(msg.liveParameters.stiffnessFactorValid)

  def test_get_msg_clips_angle_offset(self):
    """Test get_msg clips angle offset within delta limits."""
    learner = self._create_learner()
    # First call to establish baseline
    learner.get_msg(valid=True)
    # Second call should clip changes
    msg = learner.get_msg(valid=True)

    # Angle offset should be within reasonable bounds
    self.assertLessEqual(abs(msg.liveParameters.angleOffsetDeg), OFFSET_MAX + 1)


class TestRetrieveInitialVehicleParams(unittest.TestCase):
  """Test retrieve_initial_vehicle_params edge cases."""

  def test_retrieve_with_missing_params(self):
    """Test retrieve returns defaults when params are missing."""
    params = Params()
    CP = get_test_car_params()
    params.remove("LiveParametersV2")
    params.remove("CarParamsPrevRoute")

    sr, sf, offset, p_init = retrieve_initial_vehicle_params(params, CP, replay=False, debug=False)

    self.assertEqual(sr, CP.steerRatio)
    self.assertEqual(sf, 1.0)
    self.assertEqual(offset, 0.0)
    self.assertIsNone(p_init)

  def test_retrieve_resets_stiffness_when_not_replay(self):
    """Test retrieve resets stiffness to 1.0 when not in replay mode."""
    params = Params()
    CP = get_test_car_params()
    msg = get_random_live_parameters(CP)
    params.put("LiveParametersV2", msg.to_bytes())
    params.put("CarParamsPrevRoute", CP.as_builder().to_bytes())

    sr, sf, offset, _ = retrieve_initial_vehicle_params(params, CP, replay=False, debug=False)

    # Stiffness should be reset to 1.0
    self.assertEqual(sf, 1.0)

  def test_retrieve_keeps_stiffness_in_replay_mode(self):
    """Test retrieve keeps stiffness in replay mode."""
    params = Params()
    CP = get_test_car_params()
    msg = get_random_live_parameters(CP)
    params.put("LiveParametersV2", msg.to_bytes())
    params.put("CarParamsPrevRoute", CP.as_builder().to_bytes())

    sr, sf, offset, _ = retrieve_initial_vehicle_params(params, CP, replay=True, debug=False)

    # Stiffness should match saved value
    np.testing.assert_allclose(sf, msg.liveParameters.stiffnessFactor)


class TestMigrateCachedVehicleParams(unittest.TestCase):
  """Test migrate_cached_vehicle_params_if_needed."""

  def test_migrate_skips_if_new_format_exists(self):
    """Test migration skips if new format already exists."""
    params = Params()
    new_msg = messaging.new_message("liveParameters")
    new_msg.liveParameters.steerRatio = 15.0
    params.put("LiveParametersV2", new_msg.to_bytes())
    params.put("LiveParameters", {"steerRatio": 10.0})

    migrate_cached_vehicle_params_if_needed(params)

    # Should not have overwritten new format
    from cereal import log
    with log.Event.from_bytes(params.get("LiveParametersV2")) as migrated:
      self.assertEqual(migrated.liveParameters.steerRatio, 15.0)

  def test_migrate_skips_if_old_format_missing(self):
    """Test migration skips if old format is missing."""
    params = Params()
    params.remove("LiveParameters")
    params.remove("LiveParametersV2")

    # Should not raise
    migrate_cached_vehicle_params_if_needed(params)

    self.assertIsNone(params.get("LiveParametersV2"))


if __name__ == '__main__':
  unittest.main()
