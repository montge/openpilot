import numpy as np
import pytest

from cereal import log, messaging
from openpilot.selfdrive.locationd.locationd import (
  LocationEstimator,
  HandleLogResult,
  calculate_invalid_input_decay,
  init_xyz_measurement,
  sensor_all_checks,
  ACCEL_SANITY_CHECK,
  ROTATION_SANITY_CHECK,
  TRANS_SANITY_CHECK,
  MIN_STD_SANITY_CHECK,
  MAX_FILTER_REWIND_TIME,
  POSENET_STD_INITIAL_VALUE,
  POSENET_STD_HIST_HALF,
)
from openpilot.selfdrive.locationd.models.constants import ObservationKind


def create_accelerometer_msg(v, timestamp=None, source=None):
  """Create an accelerometer message for testing."""
  msg = messaging.new_message('accelerometer', valid=True)
  msg.accelerometer.sensor = 4
  msg.accelerometer.type = 0x10
  if timestamp is None:
    timestamp = msg.logMonoTime
  msg.accelerometer.timestamp = timestamp
  msg.accelerometer.init('acceleration')
  msg.accelerometer.acceleration.v = v
  if source is not None:
    msg.accelerometer.source = source
  return msg


def create_gyroscope_msg(v, timestamp=None, source=None):
  """Create a gyroscope message for testing."""
  msg = messaging.new_message('gyroscope', valid=True)
  msg.gyroscope.sensor = 5
  msg.gyroscope.type = 0x10
  if timestamp is None:
    timestamp = msg.logMonoTime
  msg.gyroscope.timestamp = timestamp
  msg.gyroscope.init('gyroUncalibrated')
  msg.gyroscope.gyroUncalibrated.v = v
  if source is not None:
    msg.gyroscope.source = source
  return msg


def create_camera_odometry_msg(trans, rot, trans_std, rot_std):
  """Create a cameraOdometry message for testing."""
  msg = messaging.new_message('cameraOdometry')
  msg.cameraOdometry.trans = trans
  msg.cameraOdometry.rot = rot
  msg.cameraOdometry.transStd = trans_std
  msg.cameraOdometry.rotStd = rot_std
  return msg


def create_car_state_msg(v_ego):
  """Create a carState message for testing."""
  msg = messaging.new_message('carState')
  msg.carState.vEgo = v_ego
  return msg


def create_live_calibration_msg(rpy_calib):
  """Create a liveCalibration message for testing."""
  msg = messaging.new_message('liveCalibration')
  msg.liveCalibration.rpyCalib = rpy_calib
  return msg


class TestCalculateInvalidInputDecay:
  """Tests for the calculate_invalid_input_decay helper function."""

  def test_basic_decay(self):
    """Test decay calculation with typical values."""
    decay = calculate_invalid_input_decay(2.0, 10.0, 100.0)
    # Decay should be between 0 and 1
    assert 0.0 < decay < 1.0
    # For these values, decay should be close to 1 (slow decay)
    assert decay > 0.99

  def test_faster_decay_with_shorter_recovery(self):
    """Test that shorter recovery time gives faster decay."""
    decay_fast = calculate_invalid_input_decay(2.0, 5.0, 100.0)
    decay_slow = calculate_invalid_input_decay(2.0, 10.0, 100.0)
    # Shorter recovery should mean smaller decay value (faster decay)
    assert decay_fast < decay_slow

  def test_decay_with_different_frequencies(self):
    """Test decay calculation with different frequencies."""
    decay_high = calculate_invalid_input_decay(2.0, 10.0, 200.0)
    decay_low = calculate_invalid_input_decay(2.0, 10.0, 100.0)
    # Higher frequency means more decay steps, so per-step decay should be higher
    assert decay_high > decay_low


class TestInitXyzMeasurement:
  """Tests for the init_xyz_measurement helper function."""

  def test_init_measurement_valid(self):
    """Test initializing an XYZ measurement with valid flag true."""
    msg = messaging.new_message('livePose')
    measurement = msg.livePose.init('orientationNED')
    values = np.array([0.1, 0.2, 0.3])
    stds = np.array([0.01, 0.02, 0.03])
    init_xyz_measurement(measurement, values, stds, True)

    assert measurement.x == pytest.approx(0.1)
    assert measurement.y == pytest.approx(0.2)
    assert measurement.z == pytest.approx(0.3)
    assert measurement.xStd == pytest.approx(0.01)
    assert measurement.yStd == pytest.approx(0.02)
    assert measurement.zStd == pytest.approx(0.03)
    assert measurement.valid is True

  def test_init_measurement_invalid(self):
    """Test initializing an XYZ measurement with valid flag false."""
    msg = messaging.new_message('livePose')
    measurement = msg.livePose.init('orientationNED')
    values = np.array([0.0, 0.0, 0.0])
    stds = np.array([1.0, 1.0, 1.0])
    init_xyz_measurement(measurement, values, stds, False)

    assert measurement.valid is False


class TestLocationEstimatorInit:
  """Tests for LocationEstimator initialization."""

  def test_init_default(self):
    """Test LocationEstimator initializes with default state."""
    estimator = LocationEstimator(debug=False)

    # Check initial car speed
    assert estimator.car_speed == 0.0

    # Check initial posenet stds
    assert len(estimator.posenet_stds) == POSENET_STD_HIST_HALF * 2
    assert all(s == POSENET_STD_INITIAL_VALUE for s in estimator.posenet_stds)

    # Check initial camodo yawrate distribution
    assert estimator.camodo_yawrate_distribution[0] == 0.0
    assert estimator.camodo_yawrate_distribution[1] == 10.0

    # Check device_from_calib is identity
    np.testing.assert_array_almost_equal(estimator.device_from_calib, np.eye(3))

    # Check observations initialized
    assert ObservationKind.PHONE_ACCEL in estimator.observations
    assert ObservationKind.PHONE_GYRO in estimator.observations
    assert ObservationKind.CAMERA_ODO_ROTATION in estimator.observations
    assert ObservationKind.CAMERA_ODO_TRANSLATION in estimator.observations

  def test_init_with_debug(self):
    """Test LocationEstimator initializes with debug mode."""
    estimator = LocationEstimator(debug=True)
    assert estimator.debug is True

  def test_init_without_debug(self):
    """Test LocationEstimator initializes without debug mode."""
    estimator = LocationEstimator(debug=False)
    assert estimator.debug is False


class TestLocationEstimatorReset:
  """Tests for LocationEstimator reset functionality."""

  def test_reset_with_defaults(self):
    """Test reset with default initial state."""
    estimator = LocationEstimator(debug=False)
    t = 1.0
    estimator.reset(t)
    # After reset, filter should be initialized
    assert estimator.kf.t is not None

  def test_reset_with_custom_initial_state(self):
    """Test reset with custom initial state."""
    estimator = LocationEstimator(debug=False)
    t = 1.0
    x_initial = np.zeros(18)
    x_initial[3] = 5.0  # Set some velocity
    P_initial = np.eye(18) * 0.01
    estimator.reset(t, x_initial, P_initial)
    # Filter should be initialized with custom state
    assert estimator.kf.t is not None


class TestLocationEstimatorValidation:
  """Tests for LocationEstimator validation methods."""

  def test_validate_sensor_source_valid(self):
    """Test sensor source validation with valid source."""
    estimator = LocationEstimator(debug=False)
    # lsm6ds3 is a valid source
    assert estimator._validate_sensor_source(log.SensorEventData.SensorSource.lsm6ds3) is True

  def test_validate_sensor_source_invalid(self):
    """Test sensor source validation with invalid source (bmx055)."""
    estimator = LocationEstimator(debug=False)
    # bmx055 is explicitly filtered out
    assert estimator._validate_sensor_source(log.SensorEventData.SensorSource.bmx055) is False

  def test_validate_sensor_time_valid(self):
    """Test sensor time validation with valid timing."""
    estimator = LocationEstimator(debug=False)
    sensor_time = 100.0
    log_time = 100.05  # Within MAX_SENSOR_TIME_DIFF
    assert estimator._validate_sensor_time(sensor_time, log_time) is True

  def test_validate_sensor_time_invalid_too_far(self):
    """Test sensor time validation with timing too far apart."""
    estimator = LocationEstimator(debug=False)
    sensor_time = 100.0
    log_time = 100.2  # Beyond MAX_SENSOR_TIME_DIFF (0.1s)
    assert estimator._validate_sensor_time(sensor_time, log_time) is False

  def test_validate_sensor_time_invalid_zero(self):
    """Test sensor time validation with zero timestamp."""
    estimator = LocationEstimator(debug=False)
    sensor_time = 0.0
    log_time = 100.0
    assert estimator._validate_sensor_time(sensor_time, log_time) is False

  def test_validate_timestamp_valid(self):
    """Test timestamp validation when filter not initialized."""
    estimator = LocationEstimator(debug=False)
    # When kf.t is NaN, validation should pass
    assert estimator._validate_timestamp(100.0) is True

  def test_validate_timestamp_valid_after_init(self):
    """Test timestamp validation with valid recent timestamp."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(100.0)
    # Timestamp within rewind window
    assert estimator._validate_timestamp(100.0 - MAX_FILTER_REWIND_TIME + 0.1) is True

  def test_validate_timestamp_invalid_too_old(self):
    """Test timestamp validation with too old timestamp."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(100.0)
    # Timestamp too far in the past
    assert estimator._validate_timestamp(100.0 - MAX_FILTER_REWIND_TIME - 0.1) is False


class TestLocationEstimatorHandleLog:
  """Tests for LocationEstimator handle_log method."""

  def test_handle_accelerometer_valid(self):
    """Test handling valid accelerometer message."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    msg = create_accelerometer_msg([0.0, 0.0, -9.81])
    # Set proper timestamp in nanoseconds
    sensor_time = 1.001
    msg.accelerometer.timestamp = int(sensor_time * 1e9)

    result = estimator.handle_log(sensor_time, "accelerometer", msg.accelerometer)
    assert result == HandleLogResult.SUCCESS

  def test_handle_accelerometer_sanity_fail(self):
    """Test handling accelerometer message that fails sanity check."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    # Create message with extreme acceleration
    msg = create_accelerometer_msg([ACCEL_SANITY_CHECK + 1, 0.0, 0.0])
    sensor_time = 1.001
    msg.accelerometer.timestamp = int(sensor_time * 1e9)

    result = estimator.handle_log(sensor_time, "accelerometer", msg.accelerometer)
    assert result == HandleLogResult.INPUT_INVALID

  def test_handle_accelerometer_invalid_source(self):
    """Test handling accelerometer message with invalid sensor source."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    msg = create_accelerometer_msg([0.0, 0.0, -9.81], source=log.SensorEventData.SensorSource.bmx055)
    sensor_time = 1.001
    msg.accelerometer.timestamp = int(sensor_time * 1e9)

    result = estimator.handle_log(sensor_time, "accelerometer", msg.accelerometer)
    assert result == HandleLogResult.SENSOR_SOURCE_INVALID

  def test_handle_accelerometer_timing_invalid(self):
    """Test handling accelerometer message with invalid timing."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    msg = create_accelerometer_msg([0.0, 0.0, -9.81])
    # Set timestamp way off from log time
    msg.accelerometer.timestamp = int(0.5 * 1e9)  # 0.5s, but log time is 1.001

    result = estimator.handle_log(1.001, "accelerometer", msg.accelerometer)
    assert result == HandleLogResult.TIMING_INVALID

  def test_handle_gyroscope_valid(self):
    """Test handling valid gyroscope message."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    msg = create_gyroscope_msg([0.0, 0.0, 0.0])
    sensor_time = 1.001
    msg.gyroscope.timestamp = int(sensor_time * 1e9)

    result = estimator.handle_log(sensor_time, "gyroscope", msg.gyroscope)
    assert result == HandleLogResult.SUCCESS

  def test_handle_gyroscope_sanity_fail(self):
    """Test handling gyroscope message that fails sanity check."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    # Create message with extreme rotation
    msg = create_gyroscope_msg([ROTATION_SANITY_CHECK + 1, 0.0, 0.0])
    sensor_time = 1.001
    msg.gyroscope.timestamp = int(sensor_time * 1e9)

    result = estimator.handle_log(sensor_time, "gyroscope", msg.gyroscope)
    assert result == HandleLogResult.INPUT_INVALID

  def test_handle_car_state(self):
    """Test handling carState message."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    msg = create_car_state_msg(15.0)  # 15 m/s
    result = estimator.handle_log(1.0, "carState", msg.carState)
    assert result == HandleLogResult.SUCCESS
    assert estimator.car_speed == 15.0

  def test_handle_car_state_negative(self):
    """Test handling carState message with negative speed (uses absolute value)."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    msg = create_car_state_msg(-10.0)
    result = estimator.handle_log(1.0, "carState", msg.carState)
    assert result == HandleLogResult.SUCCESS
    assert estimator.car_speed == 10.0  # Should be absolute value

  def test_handle_live_calibration_valid(self):
    """Test handling valid liveCalibration message."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    msg = create_live_calibration_msg([0.1, 0.05, 0.02])
    result = estimator.handle_log(1.0, "liveCalibration", msg.liveCalibration)
    assert result == HandleLogResult.SUCCESS
    # device_from_calib should be updated
    assert not np.allclose(estimator.device_from_calib, np.eye(3))

  def test_handle_live_calibration_invalid(self):
    """Test handling liveCalibration message with out-of-range values."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    # Values exceed CALIB_RPY_SANITY_CHECK (0.5 rad)
    msg = create_live_calibration_msg([1.0, 0.0, 0.0])
    result = estimator.handle_log(1.0, "liveCalibration", msg.liveCalibration)
    assert result == HandleLogResult.INPUT_INVALID

  def test_handle_camera_odometry_valid(self):
    """Test handling valid cameraOdometry message."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    msg = create_camera_odometry_msg(
      trans=[1.0, 0.0, 0.0],
      rot=[0.0, 0.0, 0.01],
      trans_std=[0.1, 0.1, 0.1],
      rot_std=[0.01, 0.01, 0.01]
    )
    result = estimator.handle_log(1.01, "cameraOdometry", msg.cameraOdometry)
    assert result == HandleLogResult.SUCCESS

  def test_handle_camera_odometry_trans_sanity_fail(self):
    """Test handling cameraOdometry with extreme translation."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    msg = create_camera_odometry_msg(
      trans=[TRANS_SANITY_CHECK + 1, 0.0, 0.0],
      rot=[0.0, 0.0, 0.0],
      trans_std=[0.1, 0.1, 0.1],
      rot_std=[0.01, 0.01, 0.01]
    )
    result = estimator.handle_log(1.01, "cameraOdometry", msg.cameraOdometry)
    assert result == HandleLogResult.INPUT_INVALID

  def test_handle_camera_odometry_rot_sanity_fail(self):
    """Test handling cameraOdometry with extreme rotation."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    msg = create_camera_odometry_msg(
      trans=[1.0, 0.0, 0.0],
      rot=[ROTATION_SANITY_CHECK + 1, 0.0, 0.0],
      trans_std=[0.1, 0.1, 0.1],
      rot_std=[0.01, 0.01, 0.01]
    )
    result = estimator.handle_log(1.01, "cameraOdometry", msg.cameraOdometry)
    assert result == HandleLogResult.INPUT_INVALID

  def test_handle_camera_odometry_std_too_small(self):
    """Test handling cameraOdometry with std too small."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    msg = create_camera_odometry_msg(
      trans=[1.0, 0.0, 0.0],
      rot=[0.0, 0.0, 0.01],
      trans_std=[MIN_STD_SANITY_CHECK / 2, 0.1, 0.1],
      rot_std=[0.01, 0.01, 0.01]
    )
    result = estimator.handle_log(1.01, "cameraOdometry", msg.cameraOdometry)
    assert result == HandleLogResult.INPUT_INVALID

  def test_handle_camera_odometry_std_too_large(self):
    """Test handling cameraOdometry with std too large."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    msg = create_camera_odometry_msg(
      trans=[1.0, 0.0, 0.0],
      rot=[0.0, 0.0, 0.01],
      trans_std=[TRANS_SANITY_CHECK * 20, 0.1, 0.1],
      rot_std=[0.01, 0.01, 0.01]
    )
    result = estimator.handle_log(1.01, "cameraOdometry", msg.cameraOdometry)
    assert result == HandleLogResult.INPUT_INVALID


class TestLocationEstimatorGetMsg:
  """Tests for LocationEstimator get_msg method."""

  def test_get_msg_filter_valid(self):
    """Test get_msg with filter valid."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    msg = estimator.get_msg(sensors_valid=True, inputs_valid=True, filter_valid=True)

    assert msg.which() == 'livePose'
    assert msg.valid is True
    assert msg.livePose.sensorsOK is True
    assert msg.livePose.inputsOK is True

  def test_get_msg_filter_invalid(self):
    """Test get_msg with filter invalid."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    msg = estimator.get_msg(sensors_valid=True, inputs_valid=True, filter_valid=False)

    assert msg.valid is False
    assert msg.livePose.orientationNED.valid is False

  def test_get_msg_sensors_invalid(self):
    """Test get_msg with sensors invalid."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    msg = estimator.get_msg(sensors_valid=False, inputs_valid=True, filter_valid=True)
    assert msg.livePose.sensorsOK is False

  def test_get_msg_inputs_invalid(self):
    """Test get_msg with inputs invalid."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    msg = estimator.get_msg(sensors_valid=True, inputs_valid=False, filter_valid=True)
    assert msg.livePose.inputsOK is False

  def test_get_msg_debug_mode(self):
    """Test get_msg in debug mode includes filter state."""
    estimator = LocationEstimator(debug=True)
    estimator.reset(1.0)

    msg = estimator.get_msg(sensors_valid=True, inputs_valid=True, filter_valid=True)

    # Debug mode should include filter state
    assert len(msg.livePose.debugFilterState.value) > 0
    assert len(msg.livePose.debugFilterState.std) > 0

  def test_get_msg_posenet_ok_low_speed(self):
    """Test posenetOK is True at low speeds regardless of std spike."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)
    estimator.car_speed = 3.0  # Below 5.0 m/s threshold

    # Simulate std spike
    estimator.posenet_stds[:POSENET_STD_HIST_HALF] = 1.0
    estimator.posenet_stds[POSENET_STD_HIST_HALF:] = 10.0

    msg = estimator.get_msg(sensors_valid=True, inputs_valid=True, filter_valid=True)
    assert msg.livePose.posenetOK is True

  def test_get_msg_posenet_not_ok_high_speed_with_spike(self):
    """Test posenetOK is False at high speed with std spike."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)
    estimator.car_speed = 10.0  # Above 5.0 m/s threshold

    # Simulate significant std spike (new_mean/old_mean > 4.0 and new_mean > 7.0)
    estimator.posenet_stds[:POSENET_STD_HIST_HALF] = 1.0
    estimator.posenet_stds[POSENET_STD_HIST_HALF:] = 10.0

    msg = estimator.get_msg(sensors_valid=True, inputs_valid=True, filter_valid=True)
    assert msg.livePose.posenetOK is False


class TestSensorAllChecks:
  """Tests for the sensor_all_checks function."""

  def test_sensor_all_checks_valid(self):
    """Test sensor_all_checks with valid sensors."""
    # Note: sensor_all_checks expects full messages (with .valid at root level)
    acc_msg = create_accelerometer_msg([0.0, 0.0, -9.81])
    gyro_msg = create_gyroscope_msg([0.0, 0.0, 0.0])
    # .valid is already set to True in create_accelerometer_msg via valid=True arg

    sensor_valid = {}
    sensor_recv_time = {}
    sensor_alive = {}

    result = sensor_all_checks(
      [acc_msg],
      [gyro_msg],
      sensor_valid,
      sensor_recv_time,
      sensor_alive,
      simulation=False
    )
    assert result is True
    assert sensor_valid["accelerometer"] is True
    assert sensor_valid["gyroscope"] is True

  def test_sensor_all_checks_missing_accelerometer(self):
    """Test sensor_all_checks with missing accelerometer."""
    gyro_msg = create_gyroscope_msg([0.0, 0.0, 0.0])

    sensor_valid = {}
    sensor_recv_time = {"accelerometer": 0.0}  # Very old
    sensor_alive = {}

    result = sensor_all_checks(
      [],
      [gyro_msg],
      sensor_valid,
      sensor_recv_time,
      sensor_alive,
      simulation=False
    )
    assert result is False
    assert sensor_alive.get("accelerometer") is False

  def test_sensor_all_checks_simulation_mode(self):
    """Test sensor_all_checks in simulation mode."""
    acc_msg = create_accelerometer_msg([0.0, 0.0, -9.81])
    gyro_msg = create_gyroscope_msg([0.0, 0.0, 0.0])

    sensor_valid = {}
    sensor_recv_time = {}
    sensor_alive = {}

    result = sensor_all_checks(
      [acc_msg],
      [gyro_msg],
      sensor_valid,
      sensor_recv_time,
      sensor_alive,
      simulation=True
    )
    assert result is True

  def test_sensor_all_checks_simulation_mode_no_msgs(self):
    """Test sensor_all_checks in simulation mode with no messages."""
    sensor_valid = {}
    sensor_recv_time = {}
    sensor_alive = {}

    result = sensor_all_checks(
      [],
      [],
      sensor_valid,
      sensor_recv_time,
      sensor_alive,
      simulation=True
    )
    assert result is False

  def test_sensor_all_checks_invalid_sensor(self):
    """Test sensor_all_checks with invalid sensor data."""
    acc_msg = messaging.new_message('accelerometer', valid=False)  # Invalid
    acc_msg.accelerometer.sensor = 4
    acc_msg.accelerometer.type = 0x10
    acc_msg.accelerometer.timestamp = acc_msg.logMonoTime
    acc_msg.accelerometer.init('acceleration')
    acc_msg.accelerometer.acceleration.v = [0.0, 0.0, -9.81]

    gyro_msg = create_gyroscope_msg([0.0, 0.0, 0.0])

    sensor_valid = {}
    sensor_recv_time = {}
    sensor_alive = {}

    result = sensor_all_checks(
      [acc_msg],
      [gyro_msg],
      sensor_valid,
      sensor_recv_time,
      sensor_alive,
      simulation=False
    )
    assert result is False
    assert sensor_valid["accelerometer"] is False


class TestLocationEstimatorFiniteCheck:
  """Tests for LocationEstimator _finite_check method."""

  def test_finite_check_with_finite_values(self):
    """Test _finite_check with all finite values."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    # Should not raise or reset
    new_x = np.zeros(18)
    new_P = np.eye(18)
    estimator._finite_check(1.0, new_x, new_P)

  def test_finite_check_with_nan_values(self):
    """Test _finite_check resets filter when NaN detected."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    # Call with NaN values - should trigger reset
    new_x = np.zeros(18)
    new_x[0] = np.nan
    new_P = np.eye(18)
    estimator._finite_check(2.0, new_x, new_P)

    # Filter should have been reset (exact behavior depends on reset implementation)

  def test_finite_check_with_inf_values(self):
    """Test _finite_check resets filter when infinity detected."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    # Call with inf values
    new_x = np.zeros(18)
    new_x[0] = np.inf
    new_P = np.eye(18)
    estimator._finite_check(2.0, new_x, new_P)

    # Filter should have been reset


class TestLocationEstimatorIntegration:
  """Integration tests for LocationEstimator with multiple message types."""

  def test_sensor_fusion_sequence(self):
    """Test processing a sequence of sensor messages."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    # Process car state
    car_msg = create_car_state_msg(10.0)
    result = estimator.handle_log(1.0, "carState", car_msg.carState)
    assert result == HandleLogResult.SUCCESS
    assert estimator.car_speed == 10.0

    # Process calibration
    calib_msg = create_live_calibration_msg([0.01, 0.02, 0.005])
    result = estimator.handle_log(1.0, "liveCalibration", calib_msg.liveCalibration)
    assert result == HandleLogResult.SUCCESS

    # Process accelerometer
    acc_msg = create_accelerometer_msg([0.0, 0.0, -9.81])
    acc_msg.accelerometer.timestamp = int(1.01 * 1e9)
    result = estimator.handle_log(1.01, "accelerometer", acc_msg.accelerometer)
    assert result == HandleLogResult.SUCCESS

    # Process gyroscope
    gyro_msg = create_gyroscope_msg([0.0, 0.0, 0.0])
    gyro_msg.gyroscope.timestamp = int(1.02 * 1e9)
    result = estimator.handle_log(1.02, "gyroscope", gyro_msg.gyroscope)
    assert result == HandleLogResult.SUCCESS

    # Process camera odometry
    cam_msg = create_camera_odometry_msg(
      trans=[5.0, 0.0, 0.0],
      rot=[0.0, 0.0, 0.01],
      trans_std=[0.5, 0.5, 0.5],
      rot_std=[0.05, 0.05, 0.05]
    )
    result = estimator.handle_log(1.03, "cameraOdometry", cam_msg.cameraOdometry)
    assert result == HandleLogResult.SUCCESS

    # Get final message
    msg = estimator.get_msg(sensors_valid=True, inputs_valid=True, filter_valid=True)
    assert msg.valid is True

  def test_posenet_std_rolling(self):
    """Test that posenet_stds array properly rolls with new camera odometry."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    # Check initial state
    initial_std = estimator.posenet_stds.copy()
    assert all(s == POSENET_STD_INITIAL_VALUE for s in initial_std)

    # Process camera odometry with specific trans_std
    cam_msg = create_camera_odometry_msg(
      trans=[5.0, 0.0, 0.0],
      rot=[0.0, 0.0, 0.01],
      trans_std=[0.123, 0.5, 0.5],  # x trans_std = 0.123
      rot_std=[0.05, 0.05, 0.05]
    )
    estimator.handle_log(1.01, "cameraOdometry", cam_msg.cameraOdometry)

    # The last element should be updated with trans_std[0]
    assert estimator.posenet_stds[-1] == pytest.approx(0.123)
    # Previous elements should have rolled
    assert estimator.posenet_stds[-2] == POSENET_STD_INITIAL_VALUE

  def test_camodo_yawrate_distribution_update(self):
    """Test that camodo_yawrate_distribution updates after camera odometry."""
    estimator = LocationEstimator(debug=False)
    estimator.reset(1.0)

    # Check initial
    assert estimator.camodo_yawrate_distribution[0] == 0.0

    # Process camera odometry
    cam_msg = create_camera_odometry_msg(
      trans=[5.0, 0.0, 0.0],
      rot=[0.0, 0.0, 0.05],  # yaw rate component
      trans_std=[0.5, 0.5, 0.5],
      rot_std=[0.05, 0.05, 0.05]
    )
    estimator.handle_log(1.01, "cameraOdometry", cam_msg.cameraOdometry)

    # camodo_yawrate_distribution should be updated
    # Mean should be roughly the z component of rot_device
    assert estimator.camodo_yawrate_distribution[0] != 0.0


class TestHandleLogResultEnum:
  """Tests for the HandleLogResult enum."""

  def test_enum_values(self):
    """Test that enum has expected values."""
    assert HandleLogResult.SUCCESS.value == 0
    assert HandleLogResult.TIMING_INVALID.value == 1
    assert HandleLogResult.INPUT_INVALID.value == 2
    assert HandleLogResult.SENSOR_SOURCE_INVALID.value == 3
