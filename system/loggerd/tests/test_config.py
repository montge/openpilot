"""Tests for system/loggerd/config.py - logging configuration."""
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from openpilot.system.loggerd.config import (
  CAMERA_FPS, SEGMENT_LENGTH, STATS_DIR_FILE_LIMIT,
  STATS_SOCKET, STATS_FLUSH_TIME_S,
  get_available_percent, get_available_bytes,
)


class TestConstants(unittest.TestCase):
  """Test module constants."""

  def test_camera_fps(self):
    """Test CAMERA_FPS is 20."""
    self.assertEqual(CAMERA_FPS, 20)

  def test_segment_length(self):
    """Test SEGMENT_LENGTH is 60 seconds."""
    self.assertEqual(SEGMENT_LENGTH, 60)

  def test_stats_dir_file_limit(self):
    """Test STATS_DIR_FILE_LIMIT value."""
    self.assertEqual(STATS_DIR_FILE_LIMIT, 10000)

  def test_stats_socket(self):
    """Test STATS_SOCKET is an IPC socket."""
    self.assertTrue(STATS_SOCKET.startswith("ipc://"))

  def test_stats_flush_time(self):
    """Test STATS_FLUSH_TIME_S is 60 seconds."""
    self.assertEqual(STATS_FLUSH_TIME_S, 60)


class TestGetAvailablePercent(unittest.TestCase):
  """Test get_available_percent function."""

  @patch('openpilot.system.loggerd.config.os.statvfs')
  @patch('openpilot.system.loggerd.config.Paths.log_root')
  def test_returns_calculated_percent(self, mock_log_root, mock_statvfs):
    """Test get_available_percent calculates correctly."""
    mock_log_root.return_value = "/data/media"

    mock_stat = MagicMock()
    mock_stat.f_bavail = 500  # 500 available blocks
    mock_stat.f_blocks = 1000  # 1000 total blocks
    mock_statvfs.return_value = mock_stat

    result = get_available_percent(default=0.0)

    self.assertEqual(result, 50.0)

  @patch('openpilot.system.loggerd.config.os.statvfs')
  @patch('openpilot.system.loggerd.config.Paths.log_root')
  def test_returns_default_on_oserror(self, mock_log_root, mock_statvfs):
    """Test get_available_percent returns default on OSError."""
    mock_log_root.return_value = "/nonexistent"
    mock_statvfs.side_effect = OSError("No such file")

    result = get_available_percent(default=75.0)

    self.assertEqual(result, 75.0)

  @patch('openpilot.system.loggerd.config.os.statvfs')
  @patch('openpilot.system.loggerd.config.Paths.log_root')
  def test_full_disk_returns_zero(self, mock_log_root, mock_statvfs):
    """Test get_available_percent returns 0 when disk is full."""
    mock_log_root.return_value = "/data"

    mock_stat = MagicMock()
    mock_stat.f_bavail = 0
    mock_stat.f_blocks = 1000
    mock_statvfs.return_value = mock_stat

    result = get_available_percent(default=50.0)

    self.assertEqual(result, 0.0)

  @patch('openpilot.system.loggerd.config.os.statvfs')
  @patch('openpilot.system.loggerd.config.Paths.log_root')
  def test_empty_disk_returns_100(self, mock_log_root, mock_statvfs):
    """Test get_available_percent returns 100 when disk is empty."""
    mock_log_root.return_value = "/data"

    mock_stat = MagicMock()
    mock_stat.f_bavail = 1000
    mock_stat.f_blocks = 1000
    mock_statvfs.return_value = mock_stat

    result = get_available_percent(default=0.0)

    self.assertEqual(result, 100.0)


class TestGetAvailableBytes(unittest.TestCase):
  """Test get_available_bytes function."""

  @patch('openpilot.system.loggerd.config.os.statvfs')
  @patch('openpilot.system.loggerd.config.Paths.log_root')
  def test_returns_calculated_bytes(self, mock_log_root, mock_statvfs):
    """Test get_available_bytes calculates correctly."""
    mock_log_root.return_value = "/data/media"

    mock_stat = MagicMock()
    mock_stat.f_bavail = 1000  # 1000 available blocks
    mock_stat.f_frsize = 4096  # 4KB block size
    mock_statvfs.return_value = mock_stat

    result = get_available_bytes(default=0)

    self.assertEqual(result, 1000 * 4096)

  @patch('openpilot.system.loggerd.config.os.statvfs')
  @patch('openpilot.system.loggerd.config.Paths.log_root')
  def test_returns_default_on_oserror(self, mock_log_root, mock_statvfs):
    """Test get_available_bytes returns default on OSError."""
    mock_log_root.return_value = "/nonexistent"
    mock_statvfs.side_effect = OSError("No such file")

    result = get_available_bytes(default=1000000)

    self.assertEqual(result, 1000000)

  @patch('openpilot.system.loggerd.config.os.statvfs')
  @patch('openpilot.system.loggerd.config.Paths.log_root')
  def test_zero_available(self, mock_log_root, mock_statvfs):
    """Test get_available_bytes returns 0 when no space."""
    mock_log_root.return_value = "/data"

    mock_stat = MagicMock()
    mock_stat.f_bavail = 0
    mock_stat.f_frsize = 4096
    mock_statvfs.return_value = mock_stat

    result = get_available_bytes(default=100)

    self.assertEqual(result, 0)

  @patch('openpilot.system.loggerd.config.os.statvfs')
  @patch('openpilot.system.loggerd.config.Paths.log_root')
  def test_large_disk(self, mock_log_root, mock_statvfs):
    """Test get_available_bytes with large disk values."""
    mock_log_root.return_value = "/data"

    mock_stat = MagicMock()
    mock_stat.f_bavail = 1000000000  # 1 billion blocks
    mock_stat.f_frsize = 4096
    mock_statvfs.return_value = mock_stat

    result = get_available_bytes(default=0)

    # Should be approximately 4TB
    self.assertEqual(result, 1000000000 * 4096)


if __name__ == '__main__':
  unittest.main()
