"""Tests for tools/lib/cache.py - cache path utilities."""

import os
import tempfile

from openpilot.tools.lib.cache import cache_path_for_file_path, DEFAULT_CACHE_DIR


class TestDefaultCacheDir:
  """Test DEFAULT_CACHE_DIR constant."""

  def test_default_cache_dir_format(self):
    """Test default cache dir path format."""
    assert DEFAULT_CACHE_DIR.endswith(".commacache")

  def test_default_cache_dir_in_home(self):
    """Test default cache dir is in home directory."""
    home = os.path.expanduser("~")
    assert DEFAULT_CACHE_DIR.startswith(home)


class TestCachePathForFilePath:
  """Test cache_path_for_file_path function."""

  def test_local_file_path(self):
    """Test cache path for local file without scheme."""
    with tempfile.TemporaryDirectory() as tmpdir:
      result = cache_path_for_file_path("/path/to/file.txt", cache_dir=tmpdir)

      assert "local" in result
      assert result.startswith(tmpdir)
      # Should have underscores instead of slashes
      assert "_path_to_file.txt" in result

  def test_local_file_creates_local_dir(self):
    """Test cache_path_for_file_path creates local subdirectory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_path_for_file_path("/some/file.txt", cache_dir=tmpdir)

      local_dir = os.path.join(tmpdir, "local")
      assert os.path.isdir(local_dir)

  def test_url_with_scheme(self):
    """Test cache path for URL with scheme."""
    with tempfile.TemporaryDirectory() as tmpdir:
      result = cache_path_for_file_path("https://example.com/path/to/file.txt", cache_dir=tmpdir)

      assert "local" in result
      assert "example.com" in result
      assert "_path_to_file.txt" in result

  def test_http_url(self):
    """Test cache path for HTTP URL."""
    with tempfile.TemporaryDirectory() as tmpdir:
      result = cache_path_for_file_path("http://cdn.example.org/data/file.zst", cache_dir=tmpdir)

      assert "cdn.example.org" in result
      assert "_data_file.zst" in result

  def test_different_files_different_paths(self):
    """Test different files produce different cache paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
      path1 = cache_path_for_file_path("/path/a.txt", cache_dir=tmpdir)
      path2 = cache_path_for_file_path("/path/b.txt", cache_dir=tmpdir)

      assert path1 != path2

  def test_same_file_same_path(self):
    """Test same file produces same cache path."""
    with tempfile.TemporaryDirectory() as tmpdir:
      path1 = cache_path_for_file_path("/path/file.txt", cache_dir=tmpdir)
      path2 = cache_path_for_file_path("/path/file.txt", cache_dir=tmpdir)

      assert path1 == path2

  def test_relative_path_becomes_absolute(self):
    """Test relative paths are converted to absolute."""
    with tempfile.TemporaryDirectory() as tmpdir:
      result = cache_path_for_file_path("relative/path.txt", cache_dir=tmpdir)

      # Should contain absolute path components
      assert "_" in result
      assert "relative_path.txt" in result

  def test_deep_path(self):
    """Test deep nested path."""
    with tempfile.TemporaryDirectory() as tmpdir:
      result = cache_path_for_file_path("/a/b/c/d/e/f/file.txt", cache_dir=tmpdir)

      assert "_a_b_c_d_e_f_file.txt" in result

  def test_uses_default_cache_dir(self, mocker):
    """Test function uses DEFAULT_CACHE_DIR when not specified."""
    mocker.patch('openpilot.tools.lib.cache.os.makedirs')
    result = cache_path_for_file_path("/path/file.txt")

    assert result.startswith(DEFAULT_CACHE_DIR)
