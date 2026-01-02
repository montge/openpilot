import random

import numpy as np

import cereal.messaging as messaging
from cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.locationd.calibrationd import (
  Calibrator,
  INPUTS_NEEDED,
  INPUTS_WANTED,
  BLOCK_SIZE,
  MIN_SPEED_FILTER,
  MAX_YAW_RATE_FILTER,
  SMOOTH_CYCLES,
  HEIGHT_INIT,
  MAX_ALLOWED_PITCH_SPREAD,
  MAX_ALLOWED_YAW_SPREAD,
)


def process_messages(
  c,
  cam_odo_calib,
  cycles,
  cam_odo_speed=MIN_SPEED_FILTER + 1,
  carstate_speed=MIN_SPEED_FILTER + 1,
  cam_odo_yr=0.0,
  cam_odo_speed_std=1e-3,
  cam_odo_height_std=1e-3,
):
  old_rpy_weight_prev = 0.0
  for _ in range(cycles):
    assert old_rpy_weight_prev - c.old_rpy_weight < 1 / SMOOTH_CYCLES + 1e-3
    old_rpy_weight_prev = c.old_rpy_weight
    c.handle_v_ego(carstate_speed)
    c.handle_cam_odom(
      [cam_odo_speed, np.sin(cam_odo_calib[2]) * cam_odo_speed, -np.sin(cam_odo_calib[1]) * cam_odo_speed],
      [0.0, 0.0, cam_odo_yr],
      [0.0, 0.0, 0.0],
      [cam_odo_speed_std, cam_odo_speed_std, cam_odo_speed_std],
      [0.0, 0.0, HEIGHT_INIT.item()],
      [cam_odo_height_std, cam_odo_height_std, cam_odo_height_std],
    )


class TestCalibrationd:
  def test_read_saved_params(self):
    msg = messaging.new_message('liveCalibration')
    msg.liveCalibration.validBlocks = random.randint(1, 10)
    msg.liveCalibration.rpyCalib = [random.random() for _ in range(3)]
    msg.liveCalibration.height = [random.random() for _ in range(1)]
    Params().put("CalibrationParams", msg.to_bytes())
    c = Calibrator(param_put=True)

    np.testing.assert_allclose(msg.liveCalibration.rpyCalib, c.rpy)
    np.testing.assert_allclose(msg.liveCalibration.height, c.height)
    assert msg.liveCalibration.validBlocks == c.valid_blocks

  def test_calibration_basics(self):
    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_WANTED)
    assert c.valid_blocks == INPUTS_WANTED
    np.testing.assert_allclose(c.rpy, np.zeros(3))
    np.testing.assert_allclose(c.height, HEIGHT_INIT)
    c.reset()

  def test_calibration_low_speed_reject(self):
    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_WANTED, cam_odo_speed=MIN_SPEED_FILTER - 1)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_WANTED, carstate_speed=MIN_SPEED_FILTER - 1)
    assert c.valid_blocks == 0
    np.testing.assert_allclose(c.rpy, np.zeros(3))
    np.testing.assert_allclose(c.height, HEIGHT_INIT)

  def test_calibration_yaw_rate_reject(self):
    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_WANTED, cam_odo_yr=MAX_YAW_RATE_FILTER)
    assert c.valid_blocks == 0
    np.testing.assert_allclose(c.rpy, np.zeros(3))
    np.testing.assert_allclose(c.height, HEIGHT_INIT)

  def test_calibration_speed_std_reject(self):
    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_WANTED, cam_odo_speed_std=1e3)
    assert c.valid_blocks == INPUTS_NEEDED
    np.testing.assert_allclose(c.rpy, np.zeros(3))

  def test_calibration_speed_std_height_reject(self):
    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_WANTED, cam_odo_height_std=1e3)
    assert c.valid_blocks == INPUTS_NEEDED
    np.testing.assert_allclose(c.rpy, np.zeros(3))

  def test_calibration_auto_reset(self):
    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_NEEDED)
    assert c.valid_blocks == INPUTS_NEEDED
    np.testing.assert_allclose(c.rpy, [0.0, 0.0, 0.0], atol=1e-3)
    process_messages(c, [0.0, MAX_ALLOWED_PITCH_SPREAD * 0.9, MAX_ALLOWED_YAW_SPREAD * 0.9], BLOCK_SIZE + 10)
    assert c.valid_blocks == INPUTS_NEEDED + 1
    assert c.cal_status == log.LiveCalibrationData.Status.calibrated

    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_NEEDED)
    assert c.valid_blocks == INPUTS_NEEDED
    np.testing.assert_allclose(c.rpy, [0.0, 0.0, 0.0])
    process_messages(c, [0.0, MAX_ALLOWED_PITCH_SPREAD * 1.1, 0.0], BLOCK_SIZE + 10)
    assert c.valid_blocks == 1
    assert c.cal_status == log.LiveCalibrationData.Status.recalibrating
    np.testing.assert_allclose(c.rpy, [0.0, MAX_ALLOWED_PITCH_SPREAD * 1.1, 0.0], atol=1e-2)

    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_NEEDED)
    assert c.valid_blocks == INPUTS_NEEDED
    np.testing.assert_allclose(c.rpy, [0.0, 0.0, 0.0])
    process_messages(c, [0.0, 0.0, MAX_ALLOWED_YAW_SPREAD * 1.1], BLOCK_SIZE + 10)
    assert c.valid_blocks == 1
    assert c.cal_status == log.LiveCalibrationData.Status.recalibrating
    np.testing.assert_allclose(c.rpy, [0.0, 0.0, MAX_ALLOWED_YAW_SPREAD * 1.1], atol=1e-2)


class TestCalibratorEdgeCases:
  """Test Calibrator edge cases."""

  def test_read_saved_params_invalid_bytes(self):
    """Test Calibrator handles invalid cached params gracefully."""
    # Write invalid bytes to CalibrationParams
    Params().put("CalibrationParams", b"invalid bytes")

    # Should not raise, should use defaults
    c = Calibrator(param_put=True)

    np.testing.assert_allclose(c.rpy, np.zeros(3))
    np.testing.assert_allclose(c.height, HEIGHT_INIT)
    assert c.valid_blocks == 0


class TestCalibrationHelpers:
  """Test helper functions in calibrationd."""

  def test_is_calibration_valid_within_limits(self):
    """Test is_calibration_valid returns True for valid calibration."""
    from openpilot.selfdrive.locationd.calibrationd import is_calibration_valid, PITCH_LIMITS, YAW_LIMITS

    # Center of valid range
    rpy = np.array([0.0, (PITCH_LIMITS[0] + PITCH_LIMITS[1]) / 2, (YAW_LIMITS[0] + YAW_LIMITS[1]) / 2])
    assert is_calibration_valid(rpy)

  def test_is_calibration_valid_outside_pitch(self):
    """Test is_calibration_valid returns False for invalid pitch."""
    from openpilot.selfdrive.locationd.calibrationd import is_calibration_valid, PITCH_LIMITS

    # Pitch outside limits
    rpy = np.array([0.0, PITCH_LIMITS[1] + 0.1, 0.0])
    assert not is_calibration_valid(rpy)

  def test_is_calibration_valid_outside_yaw(self):
    """Test is_calibration_valid returns False for invalid yaw."""
    from openpilot.selfdrive.locationd.calibrationd import is_calibration_valid, YAW_LIMITS

    # Yaw outside limits
    rpy = np.array([0.0, 0.0, YAW_LIMITS[1] + 0.1])
    assert not is_calibration_valid(rpy)

  def test_sanity_clip_nan_values(self):
    """Test sanity_clip replaces NaN with RPY_INIT."""
    from openpilot.selfdrive.locationd.calibrationd import sanity_clip, RPY_INIT

    rpy = np.array([np.nan, 0.0, 0.0])
    result = sanity_clip(rpy)
    np.testing.assert_allclose(result, RPY_INIT)

  def test_sanity_clip_clips_pitch(self):
    """Test sanity_clip clips pitch to limits."""
    from openpilot.selfdrive.locationd.calibrationd import sanity_clip, PITCH_LIMITS

    rpy = np.array([0.0, 1.0, 0.0])  # Pitch way out of bounds
    result = sanity_clip(rpy)
    assert result[1] <= PITCH_LIMITS[1] + 0.005

  def test_sanity_clip_clips_yaw(self):
    """Test sanity_clip clips yaw to limits."""
    from openpilot.selfdrive.locationd.calibrationd import sanity_clip, YAW_LIMITS

    rpy = np.array([0.0, 0.0, 1.0])  # Yaw way out of bounds
    result = sanity_clip(rpy)
    assert result[2] <= YAW_LIMITS[1] + 0.005

  def test_moving_avg_with_linear_decay(self):
    """Test moving_avg_with_linear_decay calculation."""
    from openpilot.selfdrive.locationd.calibrationd import moving_avg_with_linear_decay

    prev_mean = np.array([1.0, 2.0, 3.0])
    new_val = np.array([10.0, 20.0, 30.0])
    # At idx=0, result should be weighted towards new_val
    result = moving_avg_with_linear_decay(prev_mean, new_val, idx=0, block_size=10.0)
    np.testing.assert_allclose(result, new_val)

  def test_moving_avg_midpoint(self):
    """Test moving_avg_with_linear_decay at midpoint."""
    from openpilot.selfdrive.locationd.calibrationd import moving_avg_with_linear_decay

    prev_mean = np.array([0.0])
    new_val = np.array([10.0])
    # At idx=5, block_size=10, weight is 50/50
    result = moving_avg_with_linear_decay(prev_mean, new_val, idx=5, block_size=10.0)
    np.testing.assert_allclose(result, np.array([5.0]))


class TestCalibratorMethods:
  """Test Calibrator class methods."""

  def test_calibrator_init_default(self):
    """Test Calibrator initializes with defaults."""
    c = Calibrator(param_put=False)
    assert c.valid_blocks == 0
    np.testing.assert_allclose(c.rpy, np.zeros(3))
    np.testing.assert_allclose(c.height, HEIGHT_INIT)

  def test_calibrator_reset(self):
    """Test Calibrator reset method."""
    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * 2)
    assert c.valid_blocks > 0

    c.reset()
    assert c.valid_blocks == 0
    assert c.idx == 0
    assert c.block_idx == 0

  def test_calibrator_reset_with_priors(self):
    """Test Calibrator reset with initial values."""
    c = Calibrator(param_put=False)
    rpy_init = np.array([0.0, 0.05, 0.02])
    c.reset(rpy_init=rpy_init, valid_blocks=3)
    np.testing.assert_allclose(c.rpy, rpy_init)
    assert c.valid_blocks == 3

  def test_calibrator_reset_with_invalid_values(self):
    """Test Calibrator reset handles invalid inputs."""
    c = Calibrator(param_put=False)
    # NaN values should be replaced with defaults
    c.reset(rpy_init=np.array([np.nan, 0.0, 0.0]))
    np.testing.assert_allclose(c.rpy, np.zeros(3))

    # Invalid valid_blocks should be set to 0
    c.reset(valid_blocks=-5)
    assert c.valid_blocks == 0

  def test_calibrator_handle_v_ego(self):
    """Test handle_v_ego stores velocity."""
    c = Calibrator(param_put=False)
    c.handle_v_ego(25.0)
    assert c.v_ego == 25.0

  def test_calibrator_get_smooth_rpy_no_weight(self):
    """Test get_smooth_rpy with no old_rpy_weight."""
    c = Calibrator(param_put=False)
    c.old_rpy_weight = 0.0
    c.rpy = np.array([0.1, 0.2, 0.3])
    result = c.get_smooth_rpy()
    np.testing.assert_allclose(result, c.rpy)

  def test_calibrator_get_smooth_rpy_with_weight(self):
    """Test get_smooth_rpy blends old and new rpy."""
    c = Calibrator(param_put=False)
    c.old_rpy_weight = 0.5
    c.old_rpy = np.array([0.0, 0.0, 0.0])
    c.rpy = np.array([0.1, 0.2, 0.3])
    result = c.get_smooth_rpy()
    expected = 0.5 * c.old_rpy + 0.5 * c.rpy
    np.testing.assert_allclose(result, expected)

  def test_calibrator_get_msg(self):
    """Test get_msg returns valid message."""
    c = Calibrator(param_put=False)
    msg = c.get_msg(valid=True)
    assert msg.valid
    assert msg.liveCalibration.validBlocks == c.valid_blocks

  def test_calibrator_get_msg_not_car(self):
    """Test get_msg with not_car flag."""
    c = Calibrator(param_put=False)
    c.not_car = True
    msg = c.get_msg(valid=True)
    assert msg.liveCalibration.validBlocks == INPUTS_NEEDED
    assert msg.liveCalibration.calStatus == log.LiveCalibrationData.Status.calibrated
    assert msg.liveCalibration.calPerc == 100.0

  def test_calibrator_get_valid_idxs(self):
    """Test get_valid_idxs returns correct indices."""
    c = Calibrator(param_put=False)
    c.valid_blocks = 5
    c.block_idx = 2
    valid_idxs = c.get_valid_idxs()
    # Should include 0, 1 (before current) and 3, 4 (after current)
    assert 0 in valid_idxs
    assert 1 in valid_idxs
    assert 2 not in valid_idxs  # Current block excluded
    assert 3 in valid_idxs
    assert 4 in valid_idxs

  def test_calibrator_update_status_uncalibrated(self):
    """Test update_status with insufficient blocks."""
    c = Calibrator(param_put=False)
    c.valid_blocks = INPUTS_NEEDED - 1
    c.update_status()
    assert c.cal_status == log.LiveCalibrationData.Status.uncalibrated

  def test_calibrator_update_status_calibrated(self):
    """Test update_status with valid calibration."""
    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_NEEDED)
    assert c.cal_status == log.LiveCalibrationData.Status.calibrated


class TestCalibrationConstants:
  """Test calibration constants are reasonable."""

  def test_min_speed_filter_positive(self):
    """Test MIN_SPEED_FILTER is positive."""
    assert MIN_SPEED_FILTER > 0

  def test_inputs_needed_less_than_wanted(self):
    """Test INPUTS_NEEDED < INPUTS_WANTED."""
    assert INPUTS_NEEDED < INPUTS_WANTED

  def test_block_size_positive(self):
    """Test BLOCK_SIZE is positive."""
    assert BLOCK_SIZE > 0

  def test_smooth_cycles_positive(self):
    """Test SMOOTH_CYCLES is positive."""
    assert SMOOTH_CYCLES > 0

  def test_spread_limits_positive(self):
    """Test spread limits are positive."""
    assert MAX_ALLOWED_PITCH_SPREAD > 0
    assert MAX_ALLOWED_YAW_SPREAD > 0
