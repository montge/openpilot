"""Tests for tools/lib/url_file.py - URL file reader with caching."""
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from openpilot.tools.lib.url_file import (
  hash_256, URLFile, URLFileException, CHUNK_SIZE, K,
)


class TestHash256(unittest.TestCase):
  """Test hash_256 function."""

  def test_returns_hex_string(self):
    """Test hash_256 returns hex string."""
    result = hash_256("https://example.com/file.txt")
    self.assertTrue(all(c in '0123456789abcdef' for c in result))

  def test_consistent_hash(self):
    """Test same input gives same hash."""
    url = "https://example.com/file.txt"
    h1 = hash_256(url)
    h2 = hash_256(url)
    self.assertEqual(h1, h2)

  def test_different_urls_different_hashes(self):
    """Test different URLs give different hashes."""
    h1 = hash_256("https://example.com/file1.txt")
    h2 = hash_256("https://example.com/file2.txt")
    self.assertNotEqual(h1, h2)

  def test_ignores_query_params(self):
    """Test hash ignores query parameters."""
    h1 = hash_256("https://example.com/file.txt")
    h2 = hash_256("https://example.com/file.txt?token=abc")
    self.assertEqual(h1, h2)

  def test_hash_length(self):
    """Test hash is 64 characters (SHA-256)."""
    result = hash_256("https://example.com/file.txt")
    self.assertEqual(len(result), 64)


class TestURLFileException(unittest.TestCase):
  """Test URLFileException."""

  def test_is_exception(self):
    """Test URLFileException is an Exception."""
    self.assertTrue(issubclass(URLFileException, Exception))

  def test_can_raise(self):
    """Test can raise URLFileException."""
    with self.assertRaises(URLFileException):
      raise URLFileException("test error")

  def test_has_message(self):
    """Test URLFileException carries message."""
    try:
      raise URLFileException("test message")
    except URLFileException as e:
      self.assertEqual(str(e), "test message")


class TestURLFileConstants(unittest.TestCase):
  """Test module constants."""

  def test_k_value(self):
    """Test K is 1000."""
    self.assertEqual(K, 1000)

  def test_chunk_size(self):
    """Test CHUNK_SIZE is 1MB."""
    self.assertEqual(CHUNK_SIZE, 1000 * K)
    self.assertEqual(CHUNK_SIZE, 1000000)


class TestURLFileInit(unittest.TestCase):
  """Test URLFile initialization."""

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_init_with_url(self, mock_cache_root):
    """Test URLFile initializes with URL."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")
    self.assertEqual(uf._url, "https://example.com/file.txt")

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_init_default_timeout(self, mock_cache_root):
    """Test URLFile has default timeout."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")
    self.assertEqual(uf._timeout.connect_timeout, 10)
    self.assertEqual(uf._timeout.read_timeout, 10)

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_init_custom_timeout(self, mock_cache_root):
    """Test URLFile with custom timeout."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt", timeout=30)
    self.assertEqual(uf._timeout.connect_timeout, 30)
    self.assertEqual(uf._timeout.read_timeout, 30)

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_init_position_zero(self, mock_cache_root):
    """Test URLFile starts at position 0."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")
    self.assertEqual(uf._pos, 0)

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_init_length_none(self, mock_cache_root):
    """Test URLFile length starts as None."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")
    self.assertIsNone(uf._length)

  @patch.dict(os.environ, {'FILEREADER_CACHE': '1'})
  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  @patch('openpilot.tools.lib.url_file.os.makedirs')
  def test_init_cache_env_enabled(self, mock_makedirs, mock_cache_root):
    """Test FILEREADER_CACHE=1 enables caching."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")
    self.assertFalse(uf._force_download)

  @patch.dict(os.environ, {'FILEREADER_CACHE': '0'})
  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_init_cache_env_disabled(self, mock_cache_root):
    """Test FILEREADER_CACHE=0 disables caching."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")
    self.assertTrue(uf._force_download)

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  @patch('openpilot.tools.lib.url_file.os.makedirs')
  def test_init_cache_override_true(self, mock_makedirs, mock_cache_root):
    """Test cache=True overrides env."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt", cache=True)
    self.assertFalse(uf._force_download)

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_init_cache_override_false(self, mock_cache_root):
    """Test cache=False overrides env."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt", cache=False)
    self.assertTrue(uf._force_download)


class TestURLFileContextManager(unittest.TestCase):
  """Test URLFile context manager."""

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_enter_returns_self(self, mock_cache_root):
    """Test __enter__ returns self."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")
    result = uf.__enter__()
    self.assertIs(result, uf)

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_exit_returns_none(self, mock_cache_root):
    """Test __exit__ returns None."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")
    result = uf.__exit__(None, None, None)
    self.assertIsNone(result)

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_context_manager_usage(self, mock_cache_root):
    """Test URLFile works as context manager."""
    mock_cache_root.return_value = "/tmp/cache"
    with URLFile("https://example.com/file.txt") as uf:
      self.assertIsInstance(uf, URLFile)


class TestURLFileSeek(unittest.TestCase):
  """Test URLFile.seek method."""

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_seek_updates_position(self, mock_cache_root):
    """Test seek updates position."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")
    uf.seek(100)
    self.assertEqual(uf._pos, 100)

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_seek_to_zero(self, mock_cache_root):
    """Test seek to zero."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")
    uf._pos = 100
    uf.seek(0)
    self.assertEqual(uf._pos, 0)


class TestURLFileName(unittest.TestCase):
  """Test URLFile.name property."""

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_name_returns_url(self, mock_cache_root):
    """Test name property returns URL."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")
    self.assertEqual(uf.name, "https://example.com/file.txt")


class TestURLFilePoolManager(unittest.TestCase):
  """Test URLFile.pool_manager method."""

  def test_pool_manager_singleton(self):
    """Test pool_manager returns same instance."""
    URLFile.reset()
    pm1 = URLFile.pool_manager()
    pm2 = URLFile.pool_manager()
    self.assertIs(pm1, pm2)

  def test_reset_clears_pool_manager(self):
    """Test reset clears pool manager."""
    URLFile.reset()
    pm1 = URLFile.pool_manager()
    URLFile.reset()
    self.assertIsNone(URLFile._pool_manager)
    pm2 = URLFile.pool_manager()
    self.assertIsNot(pm1, pm2)


class TestURLFileGetLengthOnline(unittest.TestCase):
  """Test URLFile.get_length_online method."""

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_returns_content_length(self, mock_cache_root):
    """Test get_length_online returns content-length."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")

    with patch.object(uf, '_request') as mock_request:
      mock_response = MagicMock()
      mock_response.status = 200
      mock_response.headers = {'content-length': '12345'}
      mock_request.return_value = mock_response

      result = uf.get_length_online()

      self.assertEqual(result, 12345)

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_returns_negative_on_error(self, mock_cache_root):
    """Test get_length_online returns -1 on error status."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")

    with patch.object(uf, '_request') as mock_request:
      mock_response = MagicMock()
      mock_response.status = 404
      mock_request.return_value = mock_response

      result = uf.get_length_online()

      self.assertEqual(result, -1)

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_returns_zero_when_no_header(self, mock_cache_root):
    """Test get_length_online returns 0 when no content-length."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")

    with patch.object(uf, '_request') as mock_request:
      mock_response = MagicMock()
      mock_response.status = 200
      mock_response.headers = {}
      mock_request.return_value = mock_response

      result = uf.get_length_online()

      self.assertEqual(result, 0)


class TestURLFileGetLength(unittest.TestCase):
  """Test URLFile.get_length method."""

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_returns_cached_length(self, mock_cache_root):
    """Test get_length returns cached length."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")
    uf._length = 5000

    result = uf.get_length()

    self.assertEqual(result, 5000)

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_caches_length(self, mock_cache_root):
    """Test get_length caches the result."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")

    with patch.object(uf, 'get_length_online', return_value=10000):
      uf.get_length()

    self.assertEqual(uf._length, 10000)


class TestURLFileGetMultiRange(unittest.TestCase):
  """Test URLFile.get_multi_range method."""

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_single_range(self, mock_cache_root):
    """Test get_multi_range with single range."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")

    with patch.object(uf, '_request') as mock_request:
      mock_response = MagicMock()
      mock_response.status = 206
      mock_response.headers = {'content-type': 'application/octet-stream'}
      mock_response.data = b"Hello World"
      mock_request.return_value = mock_response

      result = uf.get_multi_range([(0, 11)])

      self.assertEqual(result, [b"Hello World"])

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_raises_on_bad_status(self, mock_cache_root):
    """Test get_multi_range raises on bad status."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")

    with patch.object(uf, '_request') as mock_request:
      mock_response = MagicMock()
      mock_response.status = 404
      mock_request.return_value = mock_response

      with self.assertRaises(URLFileException):
        uf.get_multi_range([(0, 11)])

  @patch('openpilot.tools.lib.url_file.Paths.download_cache_root')
  def test_assert_end_greater_than_start(self, mock_cache_root):
    """Test get_multi_range asserts end > start."""
    mock_cache_root.return_value = "/tmp/cache"
    uf = URLFile("https://example.com/file.txt")

    with self.assertRaises(AssertionError):
      uf.get_multi_range([(10, 5)])  # end < start


if __name__ == '__main__':
  unittest.main()
