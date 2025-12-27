"""Tests for system/hardware/base.py - hardware abstraction base classes."""
import os
import tempfile
import unittest
from unittest.mock import patch, mock_open, MagicMock
from dataclasses import fields

from cereal import log

from openpilot.system.hardware.base import (
  Profile, ThermalZone, ThermalConfig, LPABase, HardwareBase,
  LPAError, LPAProfileNotFoundError, NetworkType, NetworkStrength,
)


class TestProfile(unittest.TestCase):
  """Test Profile dataclass."""

  def test_profile_creation(self):
    """Test Profile can be created with all fields."""
    profile = Profile(
      iccid="12345678901234567890",
      nickname="Test SIM",
      enabled=True,
      provider="Test Provider",
    )
    self.assertEqual(profile.iccid, "12345678901234567890")
    self.assertEqual(profile.nickname, "Test SIM")
    self.assertTrue(profile.enabled)
    self.assertEqual(profile.provider, "Test Provider")

  def test_profile_disabled(self):
    """Test Profile with enabled=False."""
    profile = Profile(
      iccid="123",
      nickname="Disabled",
      enabled=False,
      provider="Provider",
    )
    self.assertFalse(profile.enabled)


class TestThermalZone(unittest.TestCase):
  """Test ThermalZone dataclass."""

  def test_thermal_zone_creation(self):
    """Test ThermalZone can be created."""
    zone = ThermalZone(name="cpu-0-0", scale=1000.0)
    self.assertEqual(zone.name, "cpu-0-0")
    self.assertEqual(zone.scale, 1000.0)
    self.assertEqual(zone.zone_number, -1)

  def test_thermal_zone_default_scale(self):
    """Test ThermalZone has default scale of 1000."""
    zone = ThermalZone(name="test")
    self.assertEqual(zone.scale, 1000.0)

  @patch('os.listdir')
  @patch('builtins.open', mock_open(read_data="45000"))
  def test_thermal_zone_read_finds_zone(self, mock_listdir):
    """Test ThermalZone.read finds and reads correct zone."""
    mock_listdir.return_value = ["thermal_zone0", "thermal_zone1"]

    zone = ThermalZone(name="cpu-0-0")
    zone.zone_number = 0  # Pre-set to avoid file lookup

    with patch('builtins.open', mock_open(read_data="45000")):
      temp = zone.read()
      self.assertEqual(temp, 45.0)  # 45000 / 1000

  def test_thermal_zone_read_not_found(self):
    """Test ThermalZone.read returns 0 when zone not found."""
    zone = ThermalZone(name="nonexistent")
    zone.zone_number = 999

    # Should return 0 when file not found
    temp = zone.read()
    self.assertEqual(temp, 0)


class TestThermalConfig(unittest.TestCase):
  """Test ThermalConfig dataclass."""

  def test_thermal_config_defaults(self):
    """Test ThermalConfig has None defaults."""
    config = ThermalConfig()
    self.assertIsNone(config.cpu)
    self.assertIsNone(config.gpu)
    self.assertIsNone(config.dsp)
    self.assertIsNone(config.memory)

  def test_thermal_config_with_zones(self):
    """Test ThermalConfig with thermal zones."""
    cpu_zone = ThermalZone(name="cpu-0-0")
    config = ThermalConfig(cpu=[cpu_zone])
    self.assertIsNotNone(config.cpu)
    self.assertEqual(len(config.cpu), 1)

  def test_thermal_config_get_msg_empty(self):
    """Test get_msg returns empty dict for empty config."""
    config = ThermalConfig()
    msg = config.get_msg()
    self.assertEqual(msg, {})

  @patch.object(ThermalZone, 'read', return_value=50.0)
  def test_thermal_config_get_msg_single_zone(self, mock_read):
    """Test get_msg with single zone."""
    zone = ThermalZone(name="memory")
    config = ThermalConfig(memory=zone)
    msg = config.get_msg()
    self.assertIn('memoryTempC', msg)
    self.assertEqual(msg['memoryTempC'], 50.0)

  @patch.object(ThermalZone, 'read', return_value=45.0)
  def test_thermal_config_get_msg_list_zones(self, mock_read):
    """Test get_msg with list of zones."""
    zones = [ThermalZone(name="cpu-0"), ThermalZone(name="cpu-1")]
    config = ThermalConfig(cpu=zones)
    msg = config.get_msg()
    self.assertIn('cpuTempC', msg)
    self.assertEqual(msg['cpuTempC'], [45.0, 45.0])


class TestLPABase(unittest.TestCase):
  """Test LPABase abstract class."""

  def test_is_comma_profile_true(self):
    """Test is_comma_profile returns True for comma ICCIDs."""

    class ConcreteLPA(LPABase):
      def list_profiles(self): return []
      def get_active_profile(self): return None
      def delete_profile(self, iccid): pass
      def download_profile(self, qr, nickname=None): pass
      def nickname_profile(self, iccid, nickname): pass
      def switch_profile(self, iccid): pass

    lpa = ConcreteLPA()
    self.assertTrue(lpa.is_comma_profile("8985235123456789"))

  def test_is_comma_profile_false(self):
    """Test is_comma_profile returns False for non-comma ICCIDs."""

    class ConcreteLPA(LPABase):
      def list_profiles(self): return []
      def get_active_profile(self): return None
      def delete_profile(self, iccid): pass
      def download_profile(self, qr, nickname=None): pass
      def nickname_profile(self, iccid, nickname): pass
      def switch_profile(self, iccid): pass

    lpa = ConcreteLPA()
    self.assertFalse(lpa.is_comma_profile("1234567890123456"))


class TestLPAErrors(unittest.TestCase):
  """Test LPA error classes."""

  def test_lpa_error_is_runtime_error(self):
    """Test LPAError is a RuntimeError."""
    self.assertTrue(issubclass(LPAError, RuntimeError))

  def test_lpa_profile_not_found_is_lpa_error(self):
    """Test LPAProfileNotFoundError is an LPAError."""
    self.assertTrue(issubclass(LPAProfileNotFoundError, LPAError))

  def test_can_raise_lpa_error(self):
    """Test LPAError can be raised."""
    with self.assertRaises(LPAError):
      raise LPAError("Test error")

  def test_can_raise_profile_not_found(self):
    """Test LPAProfileNotFoundError can be raised."""
    with self.assertRaises(LPAProfileNotFoundError):
      raise LPAProfileNotFoundError("Profile not found")


class TestHardwareBase(unittest.TestCase):
  """Test HardwareBase class."""

  def _create_hardware(self):
    """Create concrete HardwareBase for testing."""

    class ConcreteHardware(HardwareBase):
      def get_device_type(self):
        return "test"

    return ConcreteHardware()

  def test_booted_returns_true(self):
    """Test booted() returns True by default."""
    hw = self._create_hardware()
    self.assertTrue(hw.booted())

  def test_get_os_version_returns_none(self):
    """Test get_os_version() returns None by default."""
    hw = self._create_hardware()
    self.assertIsNone(hw.get_os_version())

  def test_get_imei_returns_empty_string(self):
    """Test get_imei() returns empty string by default."""
    hw = self._create_hardware()
    self.assertEqual(hw.get_imei(0), "")

  def test_get_serial_returns_empty_string(self):
    """Test get_serial() returns empty string by default."""
    hw = self._create_hardware()
    self.assertEqual(hw.get_serial(), "")

  def test_get_network_info_returns_none(self):
    """Test get_network_info() returns None by default."""
    hw = self._create_hardware()
    self.assertIsNone(hw.get_network_info())

  def test_get_network_type_returns_none_type(self):
    """Test get_network_type() returns NetworkType.none by default."""
    hw = self._create_hardware()
    self.assertEqual(hw.get_network_type(), NetworkType.none)

  def test_get_sim_info_returns_default_dict(self):
    """Test get_sim_info() returns expected default structure."""
    hw = self._create_hardware()
    info = hw.get_sim_info()
    self.assertEqual(info['sim_id'], '')
    self.assertIsNone(info['mcc_mnc'])
    self.assertEqual(info['network_type'], ["Unknown"])
    self.assertEqual(info['sim_state'], ["ABSENT"])
    self.assertFalse(info['data_connected'])

  def test_get_sim_lpa_raises_not_implemented(self):
    """Test get_sim_lpa() raises NotImplementedError by default."""
    hw = self._create_hardware()
    with self.assertRaises(NotImplementedError):
      hw.get_sim_lpa()

  def test_get_network_strength_returns_unknown(self):
    """Test get_network_strength() returns unknown by default."""
    hw = self._create_hardware()
    self.assertEqual(hw.get_network_strength(NetworkType.wifi), NetworkStrength.unknown)

  def test_get_network_metered_wifi_false(self):
    """Test get_network_metered() returns False for wifi."""
    hw = self._create_hardware()
    self.assertFalse(hw.get_network_metered(NetworkType.wifi))

  def test_get_network_metered_ethernet_false(self):
    """Test get_network_metered() returns False for ethernet."""
    hw = self._create_hardware()
    self.assertFalse(hw.get_network_metered(NetworkType.ethernet))

  def test_get_network_metered_none_false(self):
    """Test get_network_metered() returns False for none type."""
    hw = self._create_hardware()
    self.assertFalse(hw.get_network_metered(NetworkType.none))

  def test_get_network_metered_cell_true(self):
    """Test get_network_metered() returns True for cell."""
    hw = self._create_hardware()
    self.assertTrue(hw.get_network_metered(NetworkType.cell4G))

  def test_get_current_power_draw_returns_zero(self):
    """Test get_current_power_draw() returns 0 by default."""
    hw = self._create_hardware()
    self.assertEqual(hw.get_current_power_draw(), 0)

  def test_get_som_power_draw_returns_zero(self):
    """Test get_som_power_draw() returns 0 by default."""
    hw = self._create_hardware()
    self.assertEqual(hw.get_som_power_draw(), 0)

  def test_get_thermal_config_returns_empty(self):
    """Test get_thermal_config() returns empty ThermalConfig."""
    hw = self._create_hardware()
    config = hw.get_thermal_config()
    self.assertIsInstance(config, ThermalConfig)

  def test_get_screen_brightness_returns_zero(self):
    """Test get_screen_brightness() returns 0 by default."""
    hw = self._create_hardware()
    self.assertEqual(hw.get_screen_brightness(), 0)

  def test_get_gpu_usage_percent_returns_zero(self):
    """Test get_gpu_usage_percent() returns 0 by default."""
    hw = self._create_hardware()
    self.assertEqual(hw.get_gpu_usage_percent(), 0)

  def test_get_modem_version_returns_none(self):
    """Test get_modem_version() returns None by default."""
    hw = self._create_hardware()
    self.assertIsNone(hw.get_modem_version())

  def test_get_modem_temperatures_returns_empty_list(self):
    """Test get_modem_temperatures() returns empty list by default."""
    hw = self._create_hardware()
    self.assertEqual(hw.get_modem_temperatures(), [])

  def test_get_networks_returns_none(self):
    """Test get_networks() returns None by default."""
    hw = self._create_hardware()
    self.assertIsNone(hw.get_networks())

  def test_has_internal_panda_returns_false(self):
    """Test has_internal_panda() returns False by default."""
    hw = self._create_hardware()
    self.assertFalse(hw.has_internal_panda())

  def test_get_modem_data_usage_returns_negative(self):
    """Test get_modem_data_usage() returns (-1, -1) by default."""
    hw = self._create_hardware()
    self.assertEqual(hw.get_modem_data_usage(), (-1, -1))

  def test_get_voltage_returns_zero(self):
    """Test get_voltage() returns 0.0 by default."""
    hw = self._create_hardware()
    self.assertEqual(hw.get_voltage(), 0.0)

  def test_get_current_returns_zero(self):
    """Test get_current() returns 0.0 by default."""
    hw = self._create_hardware()
    self.assertEqual(hw.get_current(), 0.0)


class TestHardwareBaseStaticMethods(unittest.TestCase):
  """Test HardwareBase static methods."""

  def test_read_param_file_success(self):
    """Test read_param_file reads and parses file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
      f.write("12345")
      f.flush()

      result = HardwareBase.read_param_file(f.name, int, default=0)
      self.assertEqual(result, 12345)

      os.unlink(f.name)

  def test_read_param_file_returns_default_on_error(self):
    """Test read_param_file returns default when file not found."""
    result = HardwareBase.read_param_file("/nonexistent/path", int, default=42)
    self.assertEqual(result, 42)

  def test_read_param_file_with_float_parser(self):
    """Test read_param_file with float parser."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
      f.write("3.14159")
      f.flush()

      result = HardwareBase.read_param_file(f.name, float, default=0.0)
      self.assertAlmostEqual(result, 3.14159, places=4)

      os.unlink(f.name)

  @patch('builtins.open', mock_open(read_data="key1=value1 key2=value2 standalone"))
  def test_get_cmdline_parses_key_value(self):
    """Test get_cmdline parses key=value pairs."""
    result = HardwareBase.get_cmdline()
    self.assertEqual(result.get('key1'), 'value1')
    self.assertEqual(result.get('key2'), 'value2')
    self.assertNotIn('standalone', result)


if __name__ == '__main__':
  unittest.main()
