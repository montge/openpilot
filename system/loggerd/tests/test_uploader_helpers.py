"""Tests for system/loggerd/uploader.py helper functions."""

import os
import tempfile


from openpilot.system.loggerd.uploader import (
  get_directory_sort,
  listdir_by_creation,
  clear_locks,
  FakeRequest,
  FakeResponse,
  MAX_UPLOAD_SIZES,
  UPLOAD_ATTR_NAME,
  UPLOAD_ATTR_VALUE,
)


class TestGetDirectorySort:
  """Test get_directory_sort function."""

  def test_old_format_sorted_first(self):
    """Test 2024- directories are sorted first."""
    old_dir = "2024-01-15--12-30-00"
    new_dir = "2025-01-15--12-30-00"

    old_result = get_directory_sort(old_dir)
    new_result = get_directory_sort(new_dir)

    assert old_result[0] == "0"
    assert new_result[0] == "1"
    assert old_result < new_result

  def test_segments_sorted_by_number(self):
    """Test segment directories are sorted by segment number."""
    seg0 = "2024-01-15--12-30-00--0"
    seg10 = "2024-01-15--12-30-00--10"

    result0 = get_directory_sort(seg0)
    result10 = get_directory_sort(seg10)

    # Segment 0 should come before segment 10
    assert result0 < result10

  def test_same_date_different_segments(self):
    """Test same date with different segment numbers."""
    seg1 = "2024-01-15--12-30-00--1"
    seg2 = "2024-01-15--12-30-00--2"

    assert get_directory_sort(seg1) < get_directory_sort(seg2)


class TestListdirByCreation:
  """Test listdir_by_creation function."""

  def test_nonexistent_directory_returns_empty(self):
    """Test returns empty list for non-existent directory."""
    result = listdir_by_creation("/nonexistent/path")
    assert result == []

  def test_empty_directory_returns_empty(self):
    """Test returns empty list for empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      result = listdir_by_creation(tmpdir)
      assert result == []

  def test_returns_sorted_directories(self):
    """Test returns directories sorted by creation order."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create some directories
      os.makedirs(os.path.join(tmpdir, "2024-01-15--12-30-00--10"))
      os.makedirs(os.path.join(tmpdir, "2024-01-15--12-30-00--1"))
      os.makedirs(os.path.join(tmpdir, "2024-01-15--12-30-00--2"))

      result = listdir_by_creation(tmpdir)

      # Should be sorted numerically
      assert result == [
        "2024-01-15--12-30-00--1",
        "2024-01-15--12-30-00--2",
        "2024-01-15--12-30-00--10",
      ]

  def test_ignores_files(self):
    """Test ignores files, only returns directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create a directory and a file
      os.makedirs(os.path.join(tmpdir, "2024-01-15--12-30-00--0"))
      with open(os.path.join(tmpdir, "some_file.txt"), "w") as f:
        f.write("test")

      result = listdir_by_creation(tmpdir)

      assert len(result) == 1
      assert result[0] == "2024-01-15--12-30-00--0"


class TestClearLocks:
  """Test clear_locks function."""

  def test_clear_locks_removes_lock_files(self):
    """Test clear_locks removes .lock files."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create a log directory with a lock file
      logdir = os.path.join(tmpdir, "2024-01-15--12-30-00--0")
      os.makedirs(logdir)

      lock_file = os.path.join(logdir, "qlog.lock")
      with open(lock_file, "w") as f:
        f.write("")

      regular_file = os.path.join(logdir, "qlog")
      with open(regular_file, "w") as f:
        f.write("log data")

      clear_locks(tmpdir)

      # Lock file should be removed
      assert not os.path.exists(lock_file)
      # Regular file should remain
      assert os.path.exists(regular_file)

  def test_clear_locks_handles_empty_directory(self):
    """Test clear_locks handles empty log directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      logdir = os.path.join(tmpdir, "2024-01-15--12-30-00--0")
      os.makedirs(logdir)

      # Should not raise
      clear_locks(tmpdir)


class TestFakeRequestResponse:
  """Test FakeRequest and FakeResponse classes."""

  def test_fake_request_has_headers(self):
    """Test FakeRequest has Content-Length header."""
    req = FakeRequest()

    assert req.headers["Content-Length"] == "0"

  def test_fake_response_has_status_code(self):
    """Test FakeResponse has status_code 200."""
    resp = FakeResponse()

    assert resp.status_code == 200
    assert resp.request is not None
    assert resp.request.headers["Content-Length"] == "0"


class TestConstants:
  """Test module constants."""

  def test_upload_attr_name(self):
    """Test UPLOAD_ATTR_NAME constant."""
    assert UPLOAD_ATTR_NAME == 'user.upload'

  def test_upload_attr_value(self):
    """Test UPLOAD_ATTR_VALUE constant."""
    assert UPLOAD_ATTR_VALUE == b'1'

  def test_max_upload_sizes(self):
    """Test MAX_UPLOAD_SIZES has expected keys."""
    assert "qlog" in MAX_UPLOAD_SIZES
    assert "qcam" in MAX_UPLOAD_SIZES
    assert MAX_UPLOAD_SIZES["qlog"] == 25 * 1e6
    assert MAX_UPLOAD_SIZES["qcam"] == 5 * 1e6


class TestListdirByCreationOSError:
  """Test listdir_by_creation OSError handling."""

  def test_listdir_oserror_returns_empty(self, mocker):
    """Test listdir_by_creation returns empty on OSError in listdir."""
    mocker.patch('os.path.isdir', return_value=True)
    mocker.patch('os.listdir', side_effect=OSError("Permission denied"))
    mocker.patch('openpilot.system.loggerd.uploader.cloudlog')

    result = listdir_by_creation("/some/path")

    assert result == []


class TestClearLocksOSError:
  """Test clear_locks OSError handling."""

  def test_clear_locks_handles_oserror(self, mocker):
    """Test clear_locks logs exception on OSError."""
    with tempfile.TemporaryDirectory() as tmpdir:
      logdir = os.path.join(tmpdir, "2024-01-15--12-30-00--0")
      os.makedirs(logdir)

      # Create a lock file
      lock_file = os.path.join(logdir, "qlog.lock")
      with open(lock_file, "w") as f:
        f.write("")

      # Mock os.listdir to succeed first time (for root), then fail
      original_listdir = os.listdir

      def mock_listdir(path):
        if path == tmpdir:
          return original_listdir(path)
        raise OSError("Permission denied")

      mocker.patch('os.listdir', side_effect=mock_listdir)
      mock_cloudlog = mocker.patch('openpilot.system.loggerd.uploader.cloudlog')

      # Should not raise, just log exception
      clear_locks(tmpdir)

      mock_cloudlog.exception.assert_called_once()
