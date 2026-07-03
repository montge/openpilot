"""Tests for system/hardware/hw.py - hardware paths."""

import os
from pathlib import Path

from openpilot.system.hardware.hw import Paths, DEFAULT_DOWNLOAD_CACHE_ROOT


class TestDefaultDownloadCacheRoot:
  """Test DEFAULT_DOWNLOAD_CACHE_ROOT constant."""

  def test_default_value(self):
    """Test default download cache root path."""
    assert DEFAULT_DOWNLOAD_CACHE_ROOT == "/tmp/comma_download_cache"


class TestPathsCommaHome:
  """Test Paths.comma_home method."""

  def test_comma_home_without_prefix(self, mocker):
    """Test comma_home returns ~/.comma without prefix."""
    mocker.patch.dict(os.environ, {}, clear=True)
    # Remove OPENPILOT_PREFIX if present
    os.environ.pop('OPENPILOT_PREFIX', None)
    result = Paths.comma_home()

    expected = os.path.join(str(Path.home()), ".comma")
    assert result == expected

  def test_comma_home_with_prefix(self, mocker):
    """Test comma_home includes prefix."""
    mocker.patch.dict(os.environ, {'OPENPILOT_PREFIX': '_test123'})
    result = Paths.comma_home()

    expected = os.path.join(str(Path.home()), ".comma_test123")
    assert result == expected


class TestPathsLogRoot:
  """Test Paths.log_root method."""

  def test_log_root_from_env(self, mocker):
    """Test log_root uses LOG_ROOT env var if set."""
    mocker.patch.dict(os.environ, {'LOG_ROOT': '/custom/log/root'})
    result = Paths.log_root()

    assert result == '/custom/log/root'

  def test_log_root_pc(self, mocker):
    """Test log_root on PC platform."""
    mocker.patch('openpilot.system.hardware.hw.PC', True)
    mocker.patch.dict(os.environ, {}, clear=True)
    os.environ.pop('LOG_ROOT', None)
    os.environ.pop('OPENPILOT_PREFIX', None)
    result = Paths.log_root()

    expected = str(Path(Paths.comma_home()) / "media" / "0" / "realdata")
    assert result == expected

  def test_log_root_device(self, mocker):
    """Test log_root on device (non-PC)."""
    mocker.patch('openpilot.system.hardware.hw.PC', False)
    mocker.patch.dict(os.environ, {}, clear=True)
    os.environ.pop('LOG_ROOT', None)
    result = Paths.log_root()

    assert result == '/data/media/0/realdata/'


class TestPathsSwaglogRoot:
  """Test Paths.swaglog_root method."""

  def test_swaglog_root_pc(self, mocker):
    """Test swaglog_root on PC platform."""
    mocker.patch('openpilot.system.hardware.hw.PC', True)
    result = Paths.swaglog_root()

    expected = os.path.join(Paths.comma_home(), "log")
    assert result == expected

  def test_swaglog_root_device(self, mocker):
    """Test swaglog_root on device."""
    mocker.patch('openpilot.system.hardware.hw.PC', False)
    result = Paths.swaglog_root()

    assert result == "/data/log/"


class TestPathsSwaglogIpc:
  """Test Paths.swaglog_ipc method."""

  def test_swaglog_ipc_without_prefix(self, mocker):
    """Test swaglog_ipc without prefix."""
    mocker.patch.dict(os.environ, {}, clear=True)
    os.environ.pop('OPENPILOT_PREFIX', None)
    result = Paths.swaglog_ipc()

    assert result == "ipc:///tmp/logmessage"

  def test_swaglog_ipc_with_prefix(self, mocker):
    """Test swaglog_ipc includes prefix."""
    mocker.patch.dict(os.environ, {'OPENPILOT_PREFIX': '_test'})
    result = Paths.swaglog_ipc()

    assert result == "ipc:///tmp/logmessage_test"


class TestPathsDownloadCacheRoot:
  """Test Paths.download_cache_root method."""

  def test_download_cache_from_env(self, mocker):
    """Test download_cache_root uses COMMA_CACHE env var if set."""
    mocker.patch.dict(os.environ, {'COMMA_CACHE': '/custom/cache'})
    result = Paths.download_cache_root()

    assert result == '/custom/cache/'

  def test_download_cache_default_without_prefix(self, mocker):
    """Test download_cache_root uses default without prefix."""
    mocker.patch.dict(os.environ, {}, clear=True)
    os.environ.pop('COMMA_CACHE', None)
    os.environ.pop('OPENPILOT_PREFIX', None)
    result = Paths.download_cache_root()

    assert result == DEFAULT_DOWNLOAD_CACHE_ROOT + "/"

  def test_download_cache_default_with_prefix(self, mocker):
    """Test download_cache_root includes prefix."""
    mocker.patch.dict(os.environ, {'OPENPILOT_PREFIX': '_abc'}, clear=True)
    os.environ.pop('COMMA_CACHE', None)
    result = Paths.download_cache_root()

    assert result == DEFAULT_DOWNLOAD_CACHE_ROOT + "_abc/"


class TestPathsPersistRoot:
  """Test Paths.persist_root method."""

  def test_persist_root_pc(self, mocker):
    """Test persist_root on PC platform."""
    mocker.patch('openpilot.system.hardware.hw.PC', True)
    result = Paths.persist_root()

    expected = os.path.join(Paths.comma_home(), "persist")
    assert result == expected

  def test_persist_root_device(self, mocker):
    """Test persist_root on device."""
    mocker.patch('openpilot.system.hardware.hw.PC', False)
    result = Paths.persist_root()

    assert result == "/persist/"


class TestPathsStatsRoot:
  """Test Paths.stats_root method."""

  def test_stats_root_pc(self, mocker):
    """Test stats_root on PC platform."""
    mocker.patch('openpilot.system.hardware.hw.PC', True)
    result = Paths.stats_root()

    expected = str(Path(Paths.comma_home()) / "stats")
    assert result == expected

  def test_stats_root_device(self, mocker):
    """Test stats_root on device."""
    mocker.patch('openpilot.system.hardware.hw.PC', False)
    result = Paths.stats_root()

    assert result == "/data/stats/"


class TestPathsConfigRoot:
  """Test Paths.config_root method."""

  def test_config_root_pc(self, mocker):
    """Test config_root on PC platform."""
    mocker.patch('openpilot.system.hardware.hw.PC', True)
    result = Paths.config_root()

    assert result == Paths.comma_home()

  def test_config_root_device(self, mocker):
    """Test config_root on device."""
    mocker.patch('openpilot.system.hardware.hw.PC', False)
    result = Paths.config_root()

    assert result == "/tmp/.comma"


class TestPathsShmPath:
  """Test Paths.shm_path method."""

  def test_shm_path_macos(self, mocker):
    """Test shm_path on macOS returns /tmp."""
    mocker.patch('openpilot.system.hardware.hw.PC', True)
    mocker.patch('openpilot.system.hardware.hw.platform.system', return_value="Darwin")
    result = Paths.shm_path()

    assert result == "/tmp"

  def test_shm_path_linux_pc(self, mocker):
    """Test shm_path on Linux PC returns /dev/shm."""
    mocker.patch('openpilot.system.hardware.hw.PC', True)
    mocker.patch('openpilot.system.hardware.hw.platform.system', return_value="Linux")
    result = Paths.shm_path()

    assert result == "/dev/shm"

  def test_shm_path_device(self, mocker):
    """Test shm_path on device returns /dev/shm."""
    mocker.patch('openpilot.system.hardware.hw.PC', False)
    result = Paths.shm_path()

    assert result == "/dev/shm"


class TestPcHardware:
  """Test Pc hardware class."""

  def test_get_device_type(self):
    """Test Pc.get_device_type returns 'pc'."""
    from openpilot.system.hardware.pc.hardware import Pc

    pc = Pc()
    assert pc.get_device_type() == "pc"

  def test_get_network_type(self):
    """Test Pc.get_network_type returns wifi."""
    from cereal import log
    from openpilot.system.hardware.pc.hardware import Pc

    pc = Pc()
    result = pc.get_network_type()

    assert result == log.DeviceState.NetworkType.wifi
