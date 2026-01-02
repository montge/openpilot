"""Tests for tools/lib/filereader.py."""

import os
import tempfile

from openpilot.tools.lib.filereader import (
  resolve_name,
  file_exists,
  FileReader,
  DiskFile,
  internal_source_available,
  DATA_ENDPOINT,
)


class TestResolveName:
  """Test resolve_name function."""

  def test_resolve_cd_prefix(self):
    """Test cd:/ prefix is resolved to DATA_ENDPOINT."""
    result = resolve_name("cd:/path/to/file.txt")
    expected = DATA_ENDPOINT + "path/to/file.txt"
    assert result == expected

  def test_resolve_no_prefix(self):
    """Test paths without cd:/ are unchanged."""
    path = "/some/local/path/file.txt"
    result = resolve_name(path)
    assert result == path

  def test_resolve_http_unchanged(self):
    """Test http:// URLs are unchanged."""
    url = "http://example.com/file.txt"
    result = resolve_name(url)
    assert result == url

  def test_resolve_https_unchanged(self):
    """Test https:// URLs are unchanged."""
    url = "https://example.com/file.txt"
    result = resolve_name(url)
    assert result == url


class TestFileExists:
  """Test file_exists function."""

  def test_local_file_exists(self):
    """Test file_exists returns True for existing local files."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
      f.write(b"test content")
      temp_path = f.name

    try:
      # Clear the cache for this test
      file_exists.cache_clear()
      assert file_exists(temp_path) is True
    finally:
      os.unlink(temp_path)

  def test_local_file_not_exists(self):
    """Test file_exists returns False for non-existent local files."""
    file_exists.cache_clear()
    assert file_exists("/nonexistent/path/to/file.txt") is False


class TestDiskFile:
  """Test DiskFile class."""

  def test_get_multi_range_single(self):
    """Test get_multi_range with single range."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
      f.write(b"0123456789ABCDEF")
      temp_path = f.name

    try:
      disk_file = DiskFile(open(temp_path, "rb"))
      parts = disk_file.get_multi_range([(0, 5)])
      assert parts == [b"01234"]
      disk_file.close()
    finally:
      os.unlink(temp_path)

  def test_get_multi_range_multiple(self):
    """Test get_multi_range with multiple ranges."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
      f.write(b"0123456789ABCDEF")
      temp_path = f.name

    try:
      disk_file = DiskFile(open(temp_path, "rb"))
      parts = disk_file.get_multi_range([(0, 3), (5, 8), (10, 14)])
      assert parts == [b"012", b"567", b"ABCD"]
      disk_file.close()
    finally:
      os.unlink(temp_path)

  def test_get_multi_range_empty(self):
    """Test get_multi_range with empty ranges list."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
      f.write(b"test content")
      temp_path = f.name

    try:
      disk_file = DiskFile(open(temp_path, "rb"))
      parts = disk_file.get_multi_range([])
      assert parts == []
      disk_file.close()
    finally:
      os.unlink(temp_path)


class TestFileReader:
  """Test FileReader function."""

  def test_local_file_returns_disk_file(self):
    """Test FileReader returns DiskFile for local paths."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
      f.write(b"test content")
      temp_path = f.name

    try:
      reader = FileReader(temp_path)
      assert isinstance(reader, DiskFile)
      reader.close()
    finally:
      os.unlink(temp_path)

  def test_local_file_readable(self):
    """Test FileReader can read local files."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
      f.write(b"test content")
      temp_path = f.name

    try:
      reader = FileReader(temp_path)
      content = reader.read()
      assert content == b"test content"
      reader.close()
    finally:
      os.unlink(temp_path)


class TestInternalSourceAvailable:
  """Test internal_source_available function."""

  def test_directory_is_available(self):
    """Test directory path returns True."""
    with tempfile.TemporaryDirectory() as tmpdir:
      internal_source_available.cache_clear()
      assert internal_source_available(tmpdir) is True

  def test_connection_refused_returns_false(self, mocker):
    """Test ConnectionRefusedError returns False."""
    import socket as socket_module

    internal_source_available.cache_clear()

    mock_socket = mocker.MagicMock()
    mock_socket.__enter__ = mocker.MagicMock(return_value=mock_socket)
    mock_socket.__exit__ = mocker.MagicMock(return_value=False)
    mock_socket.connect.side_effect = ConnectionRefusedError()

    mocker.patch.object(socket_module, 'socket', return_value=mock_socket)

    result = internal_source_available("http://localhost:12345/test")
    assert result is False

  def test_successful_connection_returns_true(self, mocker):
    """Test successful socket connection returns True."""
    import socket as socket_module

    internal_source_available.cache_clear()

    mock_socket = mocker.MagicMock()
    mock_socket.__enter__ = mocker.MagicMock(return_value=mock_socket)
    mock_socket.__exit__ = mocker.MagicMock(return_value=False)
    mock_socket.connect.return_value = None  # Success

    mocker.patch.object(socket_module, 'socket', return_value=mock_socket)

    result = internal_source_available("http://localhost:12345/test")
    assert result is True
