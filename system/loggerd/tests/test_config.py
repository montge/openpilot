"""Tests for system/loggerd/config.py - logging configuration."""

from openpilot.system.loggerd.config import (
  CAMERA_FPS,
  SEGMENT_LENGTH,
  STATS_DIR_FILE_LIMIT,
  STATS_SOCKET,
  STATS_FLUSH_TIME_S,
  get_available_percent,
  get_available_bytes,
)


class TestConstants:
  """Test module constants."""

  def test_camera_fps(self):
    """Test CAMERA_FPS is 20."""
    assert CAMERA_FPS == 20

  def test_segment_length(self):
    """Test SEGMENT_LENGTH is 60 seconds."""
    assert SEGMENT_LENGTH == 60

  def test_stats_dir_file_limit(self):
    """Test STATS_DIR_FILE_LIMIT value."""
    assert STATS_DIR_FILE_LIMIT == 10000

  def test_stats_socket(self):
    """Test STATS_SOCKET is an IPC socket."""
    assert STATS_SOCKET.startswith("ipc://")

  def test_stats_flush_time(self):
    """Test STATS_FLUSH_TIME_S is 60 seconds."""
    assert STATS_FLUSH_TIME_S == 60


class TestGetAvailablePercent:
  """Test get_available_percent function."""

  def test_returns_calculated_percent(self, mocker):
    """Test get_available_percent calculates correctly."""
    mocker.patch('openpilot.system.loggerd.config.Paths.log_root', return_value="/data/media")

    mock_stat = mocker.MagicMock()
    mock_stat.f_bavail = 500  # 500 available blocks
    mock_stat.f_blocks = 1000  # 1000 total blocks
    mocker.patch('openpilot.system.loggerd.config.os.statvfs', return_value=mock_stat)

    result = get_available_percent(default=0.0)

    assert result == 50.0

  def test_returns_default_on_oserror(self, mocker):
    """Test get_available_percent returns default on OSError."""
    mocker.patch('openpilot.system.loggerd.config.Paths.log_root', return_value="/nonexistent")
    mocker.patch('openpilot.system.loggerd.config.os.statvfs', side_effect=OSError("No such file"))

    result = get_available_percent(default=75.0)

    assert result == 75.0

  def test_full_disk_returns_zero(self, mocker):
    """Test get_available_percent returns 0 when disk is full."""
    mocker.patch('openpilot.system.loggerd.config.Paths.log_root', return_value="/data")

    mock_stat = mocker.MagicMock()
    mock_stat.f_bavail = 0
    mock_stat.f_blocks = 1000
    mocker.patch('openpilot.system.loggerd.config.os.statvfs', return_value=mock_stat)

    result = get_available_percent(default=50.0)

    assert result == 0.0

  def test_empty_disk_returns_100(self, mocker):
    """Test get_available_percent returns 100 when disk is empty."""
    mocker.patch('openpilot.system.loggerd.config.Paths.log_root', return_value="/data")

    mock_stat = mocker.MagicMock()
    mock_stat.f_bavail = 1000
    mock_stat.f_blocks = 1000
    mocker.patch('openpilot.system.loggerd.config.os.statvfs', return_value=mock_stat)

    result = get_available_percent(default=0.0)

    assert result == 100.0


class TestGetAvailableBytes:
  """Test get_available_bytes function."""

  def test_returns_calculated_bytes(self, mocker):
    """Test get_available_bytes calculates correctly."""
    mocker.patch('openpilot.system.loggerd.config.Paths.log_root', return_value="/data/media")

    mock_stat = mocker.MagicMock()
    mock_stat.f_bavail = 1000  # 1000 available blocks
    mock_stat.f_frsize = 4096  # 4KB block size
    mocker.patch('openpilot.system.loggerd.config.os.statvfs', return_value=mock_stat)

    result = get_available_bytes(default=0)

    assert result == 1000 * 4096

  def test_returns_default_on_oserror(self, mocker):
    """Test get_available_bytes returns default on OSError."""
    mocker.patch('openpilot.system.loggerd.config.Paths.log_root', return_value="/nonexistent")
    mocker.patch('openpilot.system.loggerd.config.os.statvfs', side_effect=OSError("No such file"))

    result = get_available_bytes(default=1000000)

    assert result == 1000000

  def test_zero_available(self, mocker):
    """Test get_available_bytes returns 0 when no space."""
    mocker.patch('openpilot.system.loggerd.config.Paths.log_root', return_value="/data")

    mock_stat = mocker.MagicMock()
    mock_stat.f_bavail = 0
    mock_stat.f_frsize = 4096
    mocker.patch('openpilot.system.loggerd.config.os.statvfs', return_value=mock_stat)

    result = get_available_bytes(default=100)

    assert result == 0

  def test_large_disk(self, mocker):
    """Test get_available_bytes with large disk values."""
    mocker.patch('openpilot.system.loggerd.config.Paths.log_root', return_value="/data")

    mock_stat = mocker.MagicMock()
    mock_stat.f_bavail = 1000000000  # 1 billion blocks
    mock_stat.f_frsize = 4096
    mocker.patch('openpilot.system.loggerd.config.os.statvfs', return_value=mock_stat)

    result = get_available_bytes(default=0)

    # Should be approximately 4TB
    assert result == 1000000000 * 4096
