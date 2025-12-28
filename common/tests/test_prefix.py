"""Tests for common/prefix.py - OpenpilotPrefix context manager."""
import os
import unittest
from unittest.mock import patch

from openpilot.common.prefix import OpenpilotPrefix


class TestOpenpilotPrefixInit(unittest.TestCase):
  """Test OpenpilotPrefix initialization."""

  def test_init_with_custom_prefix(self):
    """Test initialization with custom prefix."""
    prefix = OpenpilotPrefix(prefix="test_prefix", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    self.assertEqual(prefix.prefix, "test_prefix")

  def test_init_generates_uuid_prefix(self):
    """Test initialization generates UUID prefix when not specified."""
    prefix = OpenpilotPrefix(create_dirs_on_enter=False, clean_dirs_on_exit=False)
    self.assertEqual(len(prefix.prefix), 15)

  def test_init_uuid_is_hex(self):
    """Test generated prefix is valid hex."""
    prefix = OpenpilotPrefix(create_dirs_on_enter=False, clean_dirs_on_exit=False)
    # UUID hex is 0-9a-f
    for c in prefix.prefix:
      self.assertTrue(c in '0123456789abcdef')

  def test_init_default_create_dirs(self):
    """Test create_dirs_on_enter defaults to True."""
    prefix = OpenpilotPrefix.__new__(OpenpilotPrefix)
    prefix.__init__()
    self.assertTrue(prefix.create_dirs_on_enter)

  def test_init_default_clean_dirs(self):
    """Test clean_dirs_on_exit defaults to True."""
    prefix = OpenpilotPrefix.__new__(OpenpilotPrefix)
    prefix.__init__()
    self.assertTrue(prefix.clean_dirs_on_exit)

  def test_init_default_shared_cache(self):
    """Test shared_download_cache defaults to False."""
    prefix = OpenpilotPrefix.__new__(OpenpilotPrefix)
    prefix.__init__()
    self.assertFalse(prefix.shared_download_cache)

  def test_init_custom_flags(self):
    """Test initialization with custom flags."""
    prefix = OpenpilotPrefix(
      prefix="test",
      create_dirs_on_enter=False,
      clean_dirs_on_exit=False,
      shared_download_cache=True
    )
    self.assertFalse(prefix.create_dirs_on_enter)
    self.assertFalse(prefix.clean_dirs_on_exit)
    self.assertTrue(prefix.shared_download_cache)

  @patch('openpilot.common.prefix.Paths.shm_path')
  def test_msgq_path_set(self, mock_shm_path):
    """Test msgq_path is set correctly."""
    mock_shm_path.return_value = "/dev/shm"
    prefix = OpenpilotPrefix(prefix="myprefix", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    self.assertEqual(prefix.msgq_path, "/dev/shm/msgq_myprefix")

  def test_different_prefixes_generate_different_paths(self):
    """Test different prefixes generate different msgq paths."""
    p1 = OpenpilotPrefix(prefix="prefix1", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    p2 = OpenpilotPrefix(prefix="prefix2", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    self.assertNotEqual(p1.msgq_path, p2.msgq_path)

  def test_multiple_init_generate_unique_prefixes(self):
    """Test multiple initializations generate unique prefixes."""
    prefixes = set()
    for _ in range(10):
      p = OpenpilotPrefix(create_dirs_on_enter=False, clean_dirs_on_exit=False)
      prefixes.add(p.prefix)
    self.assertEqual(len(prefixes), 10)


class TestOpenpilotPrefixCreateDirs(unittest.TestCase):
  """Test OpenpilotPrefix.create_dirs method."""

  @patch('openpilot.common.prefix.os.makedirs')
  @patch('openpilot.common.prefix.os.mkdir')
  @patch('openpilot.common.prefix.Paths.log_root')
  @patch('openpilot.common.prefix.Paths.shm_path')
  def test_creates_msgq_dir(self, mock_shm, mock_log, mock_mkdir, mock_makedirs):
    """Test create_dirs creates msgq directory."""
    mock_shm.return_value = "/dev/shm"
    mock_log.return_value = "/data/realdata"

    prefix = OpenpilotPrefix(prefix="test", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    prefix.create_dirs()

    mock_mkdir.assert_called_once_with("/dev/shm/msgq_test")

  @patch('openpilot.common.prefix.os.makedirs')
  @patch('openpilot.common.prefix.os.mkdir')
  @patch('openpilot.common.prefix.Paths.log_root')
  @patch('openpilot.common.prefix.Paths.shm_path')
  def test_creates_log_root(self, mock_shm, mock_log, mock_mkdir, mock_makedirs):
    """Test create_dirs creates log root directory."""
    mock_shm.return_value = "/dev/shm"
    mock_log.return_value = "/data/realdata"

    prefix = OpenpilotPrefix(prefix="test", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    prefix.create_dirs()

    mock_makedirs.assert_called_once_with("/data/realdata", exist_ok=True)

  @patch('openpilot.common.prefix.os.makedirs')
  @patch('openpilot.common.prefix.os.mkdir')
  @patch('openpilot.common.prefix.Paths.log_root')
  @patch('openpilot.common.prefix.Paths.shm_path')
  def test_handles_existing_msgq_dir(self, mock_shm, mock_log, mock_mkdir, mock_makedirs):
    """Test create_dirs handles FileExistsError for msgq dir."""
    mock_shm.return_value = "/dev/shm"
    mock_log.return_value = "/data/realdata"
    mock_mkdir.side_effect = FileExistsError()

    prefix = OpenpilotPrefix(prefix="test", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    # Should not raise
    prefix.create_dirs()
    mock_makedirs.assert_called_once()


if __name__ == '__main__':
  unittest.main()
