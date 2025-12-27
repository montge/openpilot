import random
import unittest
from unittest.mock import MagicMock
import numpy as np

from cereal import messaging
from openpilot.selfdrive.locationd.paramsd import (
  retrieve_initial_vehicle_params, migrate_cached_vehicle_params_if_needed,
  check_valid_with_hysteresis,
  MAX_ANGLE_OFFSET_DELTA, ROLL_MAX_DELTA, ROLL_MIN, ROLL_MAX, ROLL_STD_MAX,
  OFFSET_MAX, OFFSET_LOWERED_MAX, ROLL_LOWERED_MAX,
  MIN_ACTIVE_SPEED, LOW_ACTIVE_SPEED, LATERAL_ACC_SENSOR_THRESHOLD
)
from openpilot.selfdrive.locationd.models.car_kf import CarKalman
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


if __name__ == '__main__':
  unittest.main()
