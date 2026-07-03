"""Tests for tools/lib/url_file.py - URL file reader with caching."""

import os
import pytest

from openpilot.tools.lib.url_file import (
  hash_256,
  URLFile,
  URLFileException,
  CHUNK_SIZE,
  K,
)


class TestHash256:
  """Test hash_256 function."""

  def test_returns_hex_string(self):
    """Test hash_256 returns hex string."""
    result = hash_256("https://example.com/file.txt")
    assert all(c in '0123456789abcdef' for c in result)

  def test_consistent_hash(self):
    """Test same input gives same hash."""
    url = "https://example.com/file.txt"
    h1 = hash_256(url)
    h2 = hash_256(url)
    assert h1 == h2

  def test_different_urls_different_hashes(self):
    """Test different URLs give different hashes."""
    h1 = hash_256("https://example.com/file1.txt")
    h2 = hash_256("https://example.com/file2.txt")
    assert h1 != h2

  def test_ignores_query_params(self):
    """Test hash ignores query parameters."""
    h1 = hash_256("https://example.com/file.txt")
    h2 = hash_256("https://example.com/file.txt?token=abc")
    assert h1 == h2

  def test_hash_length(self):
    """Test hash is 64 characters (SHA-256)."""
    result = hash_256("https://example.com/file.txt")
    assert len(result) == 64


class TestURLFileException:
  """Test URLFileException."""

  def test_is_exception(self):
    """Test URLFileException is an Exception."""
    assert issubclass(URLFileException, Exception)

  def test_can_raise(self):
    """Test can raise URLFileException."""
    with pytest.raises(URLFileException):
      raise URLFileException("test error")

  def test_has_message(self):
    """Test URLFileException carries message."""
    try:
      raise URLFileException("test message")
    except URLFileException as e:
      assert str(e) == "test message"


class TestURLFileConstants:
  """Test module constants."""

  def test_k_value(self):
    """Test K is 1000."""
    assert K == 1000

  def test_chunk_size(self):
    """Test CHUNK_SIZE is 1MB."""
    assert CHUNK_SIZE == 1000 * K
    assert CHUNK_SIZE == 1000000


class TestURLFileInit:
  """Test URLFile initialization."""

  def test_init_with_url(self, mocker):
    """Test URLFile initializes with URL."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")
    assert uf._url == "https://example.com/file.txt"

  def test_init_default_timeout(self, mocker):
    """Test URLFile has default timeout."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")
    assert uf._timeout.connect_timeout == 10
    assert uf._timeout.read_timeout == 10

  def test_init_custom_timeout(self, mocker):
    """Test URLFile with custom timeout."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt", timeout=30)
    assert uf._timeout.connect_timeout == 30
    assert uf._timeout.read_timeout == 30

  def test_init_position_zero(self, mocker):
    """Test URLFile starts at position 0."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")
    assert uf._pos == 0

  def test_init_length_none(self, mocker):
    """Test URLFile length starts as None."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")
    assert uf._length is None

  def test_init_cache_env_enabled(self, mocker):
    """Test FILEREADER_CACHE=1 enables caching."""
    mocker.patch.dict(os.environ, {'FILEREADER_CACHE': '1'})
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    mocker.patch('openpilot.tools.lib.url_file.os.makedirs')
    uf = URLFile("https://example.com/file.txt")
    assert uf._force_download is False

  def test_init_cache_env_disabled(self, mocker):
    """Test FILEREADER_CACHE=0 disables caching."""
    mocker.patch.dict(os.environ, {'FILEREADER_CACHE': '0'})
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")
    assert uf._force_download is True

  def test_init_cache_override_true(self, mocker):
    """Test cache=True overrides env."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    mocker.patch('openpilot.tools.lib.url_file.os.makedirs')
    uf = URLFile("https://example.com/file.txt", cache=True)
    assert uf._force_download is False

  def test_init_cache_override_false(self, mocker):
    """Test cache=False overrides env."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt", cache=False)
    assert uf._force_download is True


class TestURLFileContextManager:
  """Test URLFile context manager."""

  def test_enter_returns_self(self, mocker):
    """Test __enter__ returns self."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")
    result = uf.__enter__()
    assert result is uf

  def test_exit_returns_none(self, mocker):
    """Test __exit__ returns None."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")
    result = uf.__exit__(None, None, None)
    assert result is None

  def test_context_manager_usage(self, mocker):
    """Test URLFile works as context manager."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    with URLFile("https://example.com/file.txt") as uf:
      assert isinstance(uf, URLFile)


class TestURLFileSeek:
  """Test URLFile.seek method."""

  def test_seek_updates_position(self, mocker):
    """Test seek updates position."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")
    uf.seek(100)
    assert uf._pos == 100

  def test_seek_to_zero(self, mocker):
    """Test seek to zero."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")
    uf._pos = 100
    uf.seek(0)
    assert uf._pos == 0


class TestURLFileName:
  """Test URLFile.name property."""

  def test_name_returns_url(self, mocker):
    """Test name property returns URL."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")
    assert uf.name == "https://example.com/file.txt"


class TestURLFilePoolManager:
  """Test URLFile.pool_manager method."""

  def test_pool_manager_singleton(self):
    """Test pool_manager returns same instance."""
    URLFile.reset()
    pm1 = URLFile.pool_manager()
    pm2 = URLFile.pool_manager()
    assert pm1 is pm2

  def test_reset_clears_pool_manager(self):
    """Test reset clears pool manager."""
    URLFile.reset()
    pm1 = URLFile.pool_manager()
    URLFile.reset()
    assert URLFile._pool_manager is None
    pm2 = URLFile.pool_manager()
    assert pm1 is not pm2


class TestURLFileGetLengthOnline:
  """Test URLFile.get_length_online method."""

  def test_returns_content_length(self, mocker):
    """Test get_length_online returns content-length."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")

    mock_response = mocker.MagicMock()
    mock_response.status = 200
    mock_response.headers = {'content-length': '12345'}
    mocker.patch.object(uf, '_request', return_value=mock_response)

    result = uf.get_length_online()

    assert result == 12345

  def test_returns_negative_on_error(self, mocker):
    """Test get_length_online returns -1 on error status."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")

    mock_response = mocker.MagicMock()
    mock_response.status = 404
    mocker.patch.object(uf, '_request', return_value=mock_response)

    result = uf.get_length_online()

    assert result == -1

  def test_returns_zero_when_no_header(self, mocker):
    """Test get_length_online returns 0 when no content-length."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")

    mock_response = mocker.MagicMock()
    mock_response.status = 200
    mock_response.headers = {}
    mocker.patch.object(uf, '_request', return_value=mock_response)

    result = uf.get_length_online()

    assert result == 0


class TestURLFileGetLength:
  """Test URLFile.get_length method."""

  def test_returns_cached_length(self, mocker):
    """Test get_length returns cached length."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")
    uf._length = 5000

    result = uf.get_length()

    assert result == 5000

  def test_caches_length(self, mocker):
    """Test get_length caches the result."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")

    mocker.patch.object(uf, 'get_length_online', return_value=10000)
    uf.get_length()

    assert uf._length == 10000


class TestURLFileRequest:
  """Test URLFile._request method."""

  def test_request_raises_urlfile_exception_on_max_retry(self, mocker):
    """Test _request raises URLFileException on MaxRetryError."""
    from urllib3.exceptions import MaxRetryError

    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")

    mock_pool = mocker.MagicMock()
    mock_pool.request.side_effect = MaxRetryError(mock_pool, "https://example.com/file.txt")
    mocker.patch.object(URLFile, 'pool_manager', return_value=mock_pool)

    with pytest.raises(URLFileException) as exc_info:
      uf._request('GET', 'https://example.com/file.txt')

    assert 'Failed to GET' in str(exc_info.value)


class TestURLFileGetMultiRange:
  """Test URLFile.get_multi_range method."""

  def test_single_range(self, mocker):
    """Test get_multi_range with single range."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")

    mock_response = mocker.MagicMock()
    mock_response.status = 206
    mock_response.headers = {'content-type': 'application/octet-stream'}
    mock_response.data = b"Hello World"
    mocker.patch.object(uf, '_request', return_value=mock_response)

    result = uf.get_multi_range([(0, 11)])

    assert result == [b"Hello World"]

  def test_raises_on_bad_status(self, mocker):
    """Test get_multi_range raises on bad status."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")

    mock_response = mocker.MagicMock()
    mock_response.status = 404
    mocker.patch.object(uf, '_request', return_value=mock_response)

    with pytest.raises(URLFileException):
      uf.get_multi_range([(0, 11)])

  def test_assert_end_greater_than_start(self, mocker):
    """Test get_multi_range asserts end > start."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")

    with pytest.raises(AssertionError):
      uf.get_multi_range([(10, 5)])  # end < start

  def test_multipart_response(self, mocker):
    """Test get_multi_range handles multipart response."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")

    # Simulate a multipart/byteranges response
    boundary = "abc123"
    body = b"--abc123\r\nContent-Range: bytes 0-4/100\r\n\r\nHello\r\n--abc123\r\nContent-Range: bytes 5-9/100\r\n\r\nWorld\r\n--abc123--"
    mock_response = mocker.MagicMock()
    mock_response.status = 206
    mock_response.headers = {'content-type': f'multipart/byteranges; boundary={boundary}'}
    mock_response.data = body
    mocker.patch.object(uf, '_request', return_value=mock_response)

    result = uf.get_multi_range([(0, 5), (5, 10)])

    assert result == [b"Hello", b"World"]

  def test_multipart_missing_boundary(self, mocker):
    """Test get_multi_range raises on missing boundary."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")

    mock_response = mocker.MagicMock()
    mock_response.status = 206
    mock_response.headers = {'content-type': 'multipart/byteranges'}  # No boundary
    mock_response.data = b""
    mocker.patch.object(uf, '_request', return_value=mock_response)

    with pytest.raises(URLFileException) as exc_info:
      uf.get_multi_range([(0, 5)])

    assert 'Missing multipart boundary' in str(exc_info.value)

  def test_multipart_part_count_mismatch(self, mocker):
    """Test get_multi_range raises on part count mismatch."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")

    # Only 1 part but requested 2 ranges
    boundary = "abc123"
    body = b"--abc123\r\nContent-Range: bytes 0-4/100\r\n\r\nHello\r\n--abc123--"
    mock_response = mocker.MagicMock()
    mock_response.status = 206
    mock_response.headers = {'content-type': f'multipart/byteranges; boundary={boundary}'}
    mock_response.data = body
    mocker.patch.object(uf, '_request', return_value=mock_response)

    with pytest.raises(URLFileException) as exc_info:
      uf.get_multi_range([(0, 5), (5, 10)])

    assert 'Expected 2 parts, got 1' in str(exc_info.value)


class TestURLFileReadAux:
  """Test URLFile.read_aux method."""

  def test_read_aux_with_length(self, mocker):
    """Test read_aux with specified length."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")
    uf._pos = 0

    mock_response = mocker.MagicMock()
    mock_response.status = 206
    mock_response.headers = {'content-type': 'application/octet-stream'}
    mock_response.data = b"Hello"
    mocker.patch.object(uf, '_request', return_value=mock_response)

    result = uf.read_aux(ll=5)

    assert result == b"Hello"
    assert uf._pos == 5

  def test_read_aux_without_length_error(self, mocker):
    """Test read_aux raises on empty remote file when length is None."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")

    mocker.patch.object(uf, 'get_length', return_value=-1)

    with pytest.raises(URLFileException) as exc_info:
      uf.read_aux(ll=None)

    assert 'Remote file is empty' in str(exc_info.value)

  def test_read_aux_without_length_success(self, mocker):
    """Test read_aux reads entire file when length is None."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt")
    uf._pos = 0

    mocker.patch.object(uf, 'get_length', return_value=10)
    mock_response = mocker.MagicMock()
    mock_response.status = 206
    mock_response.headers = {'content-type': 'application/octet-stream'}
    mock_response.data = b"0123456789"
    mocker.patch.object(uf, '_request', return_value=mock_response)

    result = uf.read_aux(ll=None)

    assert result == b"0123456789"


class TestURLFileGetLengthCaching:
  """Test URLFile.get_length caching behavior."""

  def test_get_length_reads_cached_file(self, mocker, tmp_path):
    """Test get_length reads from cached length file."""
    cache_dir = str(tmp_path)
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value=cache_dir)
    mocker.patch('openpilot.tools.lib.url_file.os.makedirs')

    uf = URLFile("https://example.com/file.txt", cache=True)

    # Create cached length file
    from openpilot.tools.lib.url_file import hash_256

    length_file = tmp_path / (hash_256("https://example.com/file.txt") + "_length")
    length_file.write_text("12345")

    result = uf.get_length()

    assert result == 12345
    assert uf._length == 12345

  def test_get_length_caches_to_file(self, mocker, tmp_path):
    """Test get_length caches length to file."""
    cache_dir = str(tmp_path)
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value=cache_dir)
    mocker.patch('openpilot.tools.lib.url_file.os.makedirs')

    uf = URLFile("https://example.com/newfile.txt", cache=True)

    mocker.patch.object(uf, 'get_length_online', return_value=9999)

    result = uf.get_length()

    assert result == 9999

    # Check file was created
    from openpilot.tools.lib.url_file import hash_256

    length_file = tmp_path / (hash_256("https://example.com/newfile.txt") + "_length")
    assert length_file.exists()
    assert length_file.read_text() == "9999"


class TestURLFileRead:
  """Test URLFile.read method with caching."""

  def test_read_with_force_download(self, mocker):
    """Test read uses read_aux when force_download is True."""
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value="/tmp/cache")
    uf = URLFile("https://example.com/file.txt", cache=False)

    mocker.patch.object(uf, 'read_aux', return_value=b"Hello World")

    result = uf.read(ll=11)

    assert result == b"Hello World"
    uf.read_aux.assert_called_once_with(ll=11)

  def test_read_caches_chunk(self, mocker, tmp_path):
    """Test read caches chunks to disk."""
    cache_dir = str(tmp_path)
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value=cache_dir)
    mocker.patch('openpilot.tools.lib.url_file.os.makedirs')

    uf = URLFile("https://example.com/file.txt", cache=True)
    uf._pos = 0

    # Mock read_aux to return data
    mocker.patch.object(uf, 'read_aux', return_value=b"A" * 100)
    mocker.patch.object(uf, 'get_length', return_value=100)

    result = uf.read(ll=50)

    assert len(result) == 50
    assert result == b"A" * 50

  def test_read_uses_cached_chunk(self, mocker, tmp_path):
    """Test read uses existing cached chunks."""
    from openpilot.tools.lib.url_file import hash_256

    cache_dir = str(tmp_path)
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value=cache_dir)
    mocker.patch('openpilot.tools.lib.url_file.os.makedirs')

    uf = URLFile("https://example.com/file.txt", cache=True)
    uf._pos = 0

    # Create cached chunk file
    chunk_file = tmp_path / (hash_256("https://example.com/file.txt") + "_0.0")
    chunk_file.write_bytes(b"Cached Data")

    mocker.patch.object(uf, 'get_length', return_value=11)

    # Mock read_aux but it shouldn't be called since chunk exists
    mock_read_aux = mocker.patch.object(uf, 'read_aux')

    result = uf.read(ll=6)

    assert result == b"Cached"
    # read_aux should not be called since we have cached data
    mock_read_aux.assert_not_called()

  def test_read_empty_file_assertion(self, mocker, tmp_path):
    """Test read asserts on empty remote file."""
    cache_dir = str(tmp_path)
    mocker.patch('openpilot.tools.lib.url_file.Paths.download_cache_root', return_value=cache_dir)
    mocker.patch('openpilot.tools.lib.url_file.os.makedirs')

    uf = URLFile("https://example.com/empty.txt", cache=True)
    uf._pos = 0

    mocker.patch.object(uf, 'get_length', return_value=-1)

    with pytest.raises(AssertionError) as exc_info:
      uf.read()

    assert 'Remote file is empty' in str(exc_info.value)
