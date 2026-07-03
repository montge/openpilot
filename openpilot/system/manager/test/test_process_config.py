"""Tests for system/manager/process_config.py - process configuration helpers."""

from openpilot.system.manager.process_config import (
  always_run,
  and_,
  driverview,
  iscar,
  joystick,
  logging,
  long_maneuver,
  notcar,
  not_joystick,
  not_long_maneuver,
  only_offroad,
  only_onroad,
  or_,
  qcomgps,
  ublox,
  ublox_available,
)


class TestDriverview:
  """Test driverview function."""

  def test_driverview_started_true(self, mocker):
    """Test driverview returns True when started."""
    params = mocker.MagicMock()
    params.get_bool.return_value = False
    CP = mocker.MagicMock()

    result = driverview(True, params, CP)

    assert result is True

  def test_driverview_driver_view_enabled(self, mocker):
    """Test driverview returns True when driver view enabled."""
    params = mocker.MagicMock()
    params.get_bool.return_value = True
    CP = mocker.MagicMock()

    result = driverview(False, params, CP)

    assert result is True

  def test_driverview_false(self, mocker):
    """Test driverview returns False when not started and not enabled."""
    params = mocker.MagicMock()
    params.get_bool.return_value = False
    CP = mocker.MagicMock()

    result = driverview(False, params, CP)

    assert result is False


class TestNotcar:
  """Test notcar function."""

  def test_notcar_started_and_notcar(self, mocker):
    """Test notcar returns True when started and CP.notCar."""
    params = mocker.MagicMock()
    CP = mocker.MagicMock()
    CP.notCar = True

    result = notcar(True, params, CP)

    assert result is True

  def test_notcar_not_started(self, mocker):
    """Test notcar returns False when not started."""
    params = mocker.MagicMock()
    CP = mocker.MagicMock()
    CP.notCar = True

    result = notcar(False, params, CP)

    assert result is False

  def test_notcar_is_car(self, mocker):
    """Test notcar returns False when is a car."""
    params = mocker.MagicMock()
    CP = mocker.MagicMock()
    CP.notCar = False

    result = notcar(True, params, CP)

    assert result is False


class TestIscar:
  """Test iscar function."""

  def test_iscar_started_and_is_car(self, mocker):
    """Test iscar returns True when started and is a car."""
    params = mocker.MagicMock()
    CP = mocker.MagicMock()
    CP.notCar = False

    result = iscar(True, params, CP)

    assert result is True

  def test_iscar_not_started(self, mocker):
    """Test iscar returns False when not started."""
    params = mocker.MagicMock()
    CP = mocker.MagicMock()
    CP.notCar = False

    result = iscar(False, params, CP)

    assert result is False


class TestLogging:
  """Test logging function."""

  def test_logging_started_is_car(self, mocker):
    """Test logging returns True when started and is a car."""
    params = mocker.MagicMock()
    params.get_bool.return_value = False
    CP = mocker.MagicMock()
    CP.notCar = False

    result = logging(True, params, CP)

    assert result is True

  def test_logging_not_started(self, mocker):
    """Test logging returns False when not started."""
    params = mocker.MagicMock()
    CP = mocker.MagicMock()
    CP.notCar = False

    result = logging(False, params, CP)

    assert result is False

  def test_logging_notcar_logging_disabled(self, mocker):
    """Test logging returns False when notCar and logging disabled."""
    params = mocker.MagicMock()
    params.get_bool.return_value = True  # DisableLogging = True
    CP = mocker.MagicMock()
    CP.notCar = True

    result = logging(True, params, CP)

    assert result is False

  def test_logging_notcar_logging_enabled(self, mocker):
    """Test logging returns True when notCar but logging not disabled."""
    params = mocker.MagicMock()
    params.get_bool.return_value = False  # DisableLogging = False
    CP = mocker.MagicMock()
    CP.notCar = True

    result = logging(True, params, CP)

    assert result is True


class TestUbloxAvailable:
  """Test ublox_available function."""

  def test_ublox_available_true(self, mocker):
    """Test ublox_available returns True when device exists."""
    mocker.patch('os.path.exists', side_effect=lambda p: p == '/dev/ttyHS0')

    result = ublox_available()

    assert result is True

  def test_ublox_available_no_device(self, mocker):
    """Test ublox_available returns False when no device."""
    mocker.patch('os.path.exists', return_value=False)

    result = ublox_available()

    assert result is False

  def test_ublox_available_quectel_override(self, mocker):
    """Test ublox_available returns False when quectel override exists."""
    mocker.patch('os.path.exists', return_value=True)

    result = ublox_available()

    assert result is False


class TestUblox:
  """Test ublox function."""

  def test_ublox_available_and_started(self, mocker):
    """Test ublox returns True when available and started."""
    mocker.patch('openpilot.system.manager.process_config.ublox_available', return_value=True)
    params = mocker.MagicMock()
    params.get_bool.return_value = True
    CP = mocker.MagicMock()

    result = ublox(True, params, CP)

    assert result is True

  def test_ublox_not_available(self, mocker):
    """Test ublox returns False when not available."""
    mocker.patch('openpilot.system.manager.process_config.ublox_available', return_value=False)
    params = mocker.MagicMock()
    params.get_bool.return_value = False
    CP = mocker.MagicMock()

    result = ublox(True, params, CP)

    assert result is False

  def test_ublox_updates_param(self, mocker):
    """Test ublox updates UbloxAvailable param when changed."""
    mocker.patch('openpilot.system.manager.process_config.ublox_available', return_value=True)
    params = mocker.MagicMock()
    params.get_bool.return_value = False  # Different from ublox_available
    CP = mocker.MagicMock()

    ublox(True, params, CP)

    params.put_bool.assert_called_once_with("UbloxAvailable", True)


class TestJoystick:
  """Test joystick function."""

  def test_joystick_enabled(self, mocker):
    """Test joystick returns True when started and enabled."""
    params = mocker.MagicMock()
    params.get_bool.return_value = True
    CP = mocker.MagicMock()

    result = joystick(True, params, CP)

    assert result is True

  def test_joystick_disabled(self, mocker):
    """Test joystick returns False when not enabled."""
    params = mocker.MagicMock()
    params.get_bool.return_value = False
    CP = mocker.MagicMock()

    result = joystick(True, params, CP)

    assert result is False


class TestNotJoystick:
  """Test not_joystick function."""

  def test_not_joystick_disabled(self, mocker):
    """Test not_joystick returns True when joystick disabled."""
    params = mocker.MagicMock()
    params.get_bool.return_value = False
    CP = mocker.MagicMock()

    result = not_joystick(True, params, CP)

    assert result is True

  def test_not_joystick_enabled(self, mocker):
    """Test not_joystick returns False when joystick enabled."""
    params = mocker.MagicMock()
    params.get_bool.return_value = True
    CP = mocker.MagicMock()

    result = not_joystick(True, params, CP)

    assert result is False


class TestLongManeuver:
  """Test long_maneuver function."""

  def test_long_maneuver_enabled(self, mocker):
    """Test long_maneuver returns True when enabled."""
    params = mocker.MagicMock()
    params.get_bool.return_value = True
    CP = mocker.MagicMock()

    result = long_maneuver(True, params, CP)

    assert result is True

  def test_long_maneuver_disabled(self, mocker):
    """Test long_maneuver returns False when disabled."""
    params = mocker.MagicMock()
    params.get_bool.return_value = False
    CP = mocker.MagicMock()

    result = long_maneuver(True, params, CP)

    assert result is False


class TestNotLongManeuver:
  """Test not_long_maneuver function."""

  def test_not_long_maneuver_disabled(self, mocker):
    """Test not_long_maneuver returns True when maneuver disabled."""
    params = mocker.MagicMock()
    params.get_bool.return_value = False
    CP = mocker.MagicMock()

    result = not_long_maneuver(True, params, CP)

    assert result is True


class TestQcomgps:
  """Test qcomgps function."""

  def test_qcomgps_when_ublox_not_available(self, mocker):
    """Test qcomgps returns True when ublox not available."""
    mocker.patch('openpilot.system.manager.process_config.ublox_available', return_value=False)
    params = mocker.MagicMock()
    CP = mocker.MagicMock()

    result = qcomgps(True, params, CP)

    assert result is True

  def test_qcomgps_when_ublox_available(self, mocker):
    """Test qcomgps returns False when ublox available."""
    mocker.patch('openpilot.system.manager.process_config.ublox_available', return_value=True)
    params = mocker.MagicMock()
    CP = mocker.MagicMock()

    result = qcomgps(True, params, CP)

    assert result is False


class TestAlwaysRun:
  """Test always_run function."""

  def test_always_run(self, mocker):
    """Test always_run always returns True."""
    params = mocker.MagicMock()
    CP = mocker.MagicMock()

    assert always_run(True, params, CP) is True
    assert always_run(False, params, CP) is True


class TestOnlyOnroad:
  """Test only_onroad function."""

  def test_only_onroad_started(self, mocker):
    """Test only_onroad returns True when started."""
    params = mocker.MagicMock()
    CP = mocker.MagicMock()

    result = only_onroad(True, params, CP)

    assert result is True

  def test_only_onroad_not_started(self, mocker):
    """Test only_onroad returns False when not started."""
    params = mocker.MagicMock()
    CP = mocker.MagicMock()

    result = only_onroad(False, params, CP)

    assert result is False


class TestOnlyOffroad:
  """Test only_offroad function."""

  def test_only_offroad_not_started(self, mocker):
    """Test only_offroad returns True when not started."""
    params = mocker.MagicMock()
    CP = mocker.MagicMock()

    result = only_offroad(False, params, CP)

    assert result is True

  def test_only_offroad_started(self, mocker):
    """Test only_offroad returns False when started."""
    params = mocker.MagicMock()
    CP = mocker.MagicMock()

    result = only_offroad(True, params, CP)

    assert result is False


def _always_false(*args):
  return False


def _always_true(*args):
  return True


class TestOrCombinator:
  """Test or_ combinator function."""

  def test_or_any_true(self, mocker):
    """Test or_ returns True when any function returns True."""
    combined = or_(_always_false, _always_true)

    assert combined(True, None, None) == 1  # or returns int

  def test_or_all_false(self, mocker):
    """Test or_ returns False when all functions return False."""
    combined = or_(_always_false, _always_false)

    assert combined(True, None, None) == 0


class TestAndCombinator:
  """Test and_ combinator function."""

  def test_and_all_true(self, mocker):
    """Test and_ returns True when all functions return True."""
    combined = and_(_always_true, _always_true)

    assert combined(True, None, None) == 1

  def test_and_any_false(self, mocker):
    """Test and_ returns False when any function returns False."""
    combined = and_(_always_true, _always_false)

    assert combined(True, None, None) == 0
