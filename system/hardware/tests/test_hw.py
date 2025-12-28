"""Tests for system/hardware/hw.py - hardware paths."""
import os
import platform
import unittest
from pathlib import Path
from unittest.mock import patch

from openpilot.system.hardware.hw import Paths, DEFAULT_DOWNLOAD_CACHE_ROOT


class TestDefaultDownloadCacheRoot(unittest.TestCase):
  """Test DEFAULT_DOWNLOAD_CACHE_ROOT constant."""

  def test_default_value(self):
    """Test default download cache root path."""
    self.assertEqual(DEFAULT_DOWNLOAD_CACHE_ROOT, "/tmp/comma_download_cache")


class TestPathsCommaHome(unittest.TestCase):
  """Test Paths.comma_home method."""

  def test_comma_home_without_prefix(self):
    """Test comma_home returns ~/.comma without prefix."""
    with patch.dict(os.environ, {}, clear=True):
      # Remove OPENPILOT_PREFIX if present
      os.environ.pop('OPENPILOT_PREFIX', None)
      result = Paths.comma_home()

      expected = os.path.join(str(Path.home()), ".comma")
      self.assertEqual(result, expected)

  def test_comma_home_with_prefix(self):
    """Test comma_home includes prefix."""
    with patch.dict(os.environ, {'OPENPILOT_PREFIX': '_test123'}):
      result = Paths.comma_home()

      expected = os.path.join(str(Path.home()), ".comma_test123")
      self.assertEqual(result, expected)


class TestPathsLogRoot(unittest.TestCase):
  """Test Paths.log_root method."""

  def test_log_root_from_env(self):
    """Test log_root uses LOG_ROOT env var if set."""
    with patch.dict(os.environ, {'LOG_ROOT': '/custom/log/root'}):
      result = Paths.log_root()

      self.assertEqual(result, '/custom/log/root')

  @patch('openpilot.system.hardware.hw.PC', True)
  def test_log_root_pc(self):
    """Test log_root on PC platform."""
    with patch.dict(os.environ, {}, clear=True):
      os.environ.pop('LOG_ROOT', None)
      os.environ.pop('OPENPILOT_PREFIX', None)
      result = Paths.log_root()

      expected = str(Path(Paths.comma_home()) / "media" / "0" / "realdata")
      self.assertEqual(result, expected)

  @patch('openpilot.system.hardware.hw.PC', False)
  def test_log_root_device(self):
    """Test log_root on device (non-PC)."""
    with patch.dict(os.environ, {}, clear=True):
      os.environ.pop('LOG_ROOT', None)
      result = Paths.log_root()

      self.assertEqual(result, '/data/media/0/realdata/')


class TestPathsSwaglogRoot(unittest.TestCase):
  """Test Paths.swaglog_root method."""

  @patch('openpilot.system.hardware.hw.PC', True)
  def test_swaglog_root_pc(self):
    """Test swaglog_root on PC platform."""
    result = Paths.swaglog_root()

    expected = os.path.join(Paths.comma_home(), "log")
    self.assertEqual(result, expected)

  @patch('openpilot.system.hardware.hw.PC', False)
  def test_swaglog_root_device(self):
    """Test swaglog_root on device."""
    result = Paths.swaglog_root()

    self.assertEqual(result, "/data/log/")


class TestPathsSwaglogIpc(unittest.TestCase):
  """Test Paths.swaglog_ipc method."""

  def test_swaglog_ipc_without_prefix(self):
    """Test swaglog_ipc without prefix."""
    with patch.dict(os.environ, {}, clear=True):
      os.environ.pop('OPENPILOT_PREFIX', None)
      result = Paths.swaglog_ipc()

      self.assertEqual(result, "ipc:///tmp/logmessage")

  def test_swaglog_ipc_with_prefix(self):
    """Test swaglog_ipc includes prefix."""
    with patch.dict(os.environ, {'OPENPILOT_PREFIX': '_test'}):
      result = Paths.swaglog_ipc()

      self.assertEqual(result, "ipc:///tmp/logmessage_test")


class TestPathsDownloadCacheRoot(unittest.TestCase):
  """Test Paths.download_cache_root method."""

  def test_download_cache_from_env(self):
    """Test download_cache_root uses COMMA_CACHE env var if set."""
    with patch.dict(os.environ, {'COMMA_CACHE': '/custom/cache'}):
      result = Paths.download_cache_root()

      self.assertEqual(result, '/custom/cache/')

  def test_download_cache_default_without_prefix(self):
    """Test download_cache_root uses default without prefix."""
    with patch.dict(os.environ, {}, clear=True):
      os.environ.pop('COMMA_CACHE', None)
      os.environ.pop('OPENPILOT_PREFIX', None)
      result = Paths.download_cache_root()

      self.assertEqual(result, DEFAULT_DOWNLOAD_CACHE_ROOT + "/")

  def test_download_cache_default_with_prefix(self):
    """Test download_cache_root includes prefix."""
    with patch.dict(os.environ, {'OPENPILOT_PREFIX': '_abc'}, clear=True):
      os.environ.pop('COMMA_CACHE', None)
      result = Paths.download_cache_root()

      self.assertEqual(result, DEFAULT_DOWNLOAD_CACHE_ROOT + "_abc/")


class TestPathsPersistRoot(unittest.TestCase):
  """Test Paths.persist_root method."""

  @patch('openpilot.system.hardware.hw.PC', True)
  def test_persist_root_pc(self):
    """Test persist_root on PC platform."""
    result = Paths.persist_root()

    expected = os.path.join(Paths.comma_home(), "persist")
    self.assertEqual(result, expected)

  @patch('openpilot.system.hardware.hw.PC', False)
  def test_persist_root_device(self):
    """Test persist_root on device."""
    result = Paths.persist_root()

    self.assertEqual(result, "/persist/")


class TestPathsStatsRoot(unittest.TestCase):
  """Test Paths.stats_root method."""

  @patch('openpilot.system.hardware.hw.PC', True)
  def test_stats_root_pc(self):
    """Test stats_root on PC platform."""
    result = Paths.stats_root()

    expected = str(Path(Paths.comma_home()) / "stats")
    self.assertEqual(result, expected)

  @patch('openpilot.system.hardware.hw.PC', False)
  def test_stats_root_device(self):
    """Test stats_root on device."""
    result = Paths.stats_root()

    self.assertEqual(result, "/data/stats/")


class TestPathsConfigRoot(unittest.TestCase):
  """Test Paths.config_root method."""

  @patch('openpilot.system.hardware.hw.PC', True)
  def test_config_root_pc(self):
    """Test config_root on PC platform."""
    result = Paths.config_root()

    self.assertEqual(result, Paths.comma_home())

  @patch('openpilot.system.hardware.hw.PC', False)
  def test_config_root_device(self):
    """Test config_root on device."""
    result = Paths.config_root()

    self.assertEqual(result, "/tmp/.comma")


class TestPathsShmPath(unittest.TestCase):
  """Test Paths.shm_path method."""

  @patch('openpilot.system.hardware.hw.PC', True)
  @patch('openpilot.system.hardware.hw.platform.system')
  def test_shm_path_macos(self, mock_system):
    """Test shm_path on macOS returns /tmp."""
    mock_system.return_value = "Darwin"
    result = Paths.shm_path()

    self.assertEqual(result, "/tmp")

  @patch('openpilot.system.hardware.hw.PC', True)
  @patch('openpilot.system.hardware.hw.platform.system')
  def test_shm_path_linux_pc(self, mock_system):
    """Test shm_path on Linux PC returns /dev/shm."""
    mock_system.return_value = "Linux"
    result = Paths.shm_path()

    self.assertEqual(result, "/dev/shm")

  @patch('openpilot.system.hardware.hw.PC', False)
  def test_shm_path_device(self):
    """Test shm_path on device returns /dev/shm."""
    result = Paths.shm_path()

    self.assertEqual(result, "/dev/shm")


if __name__ == '__main__':
  unittest.main()
