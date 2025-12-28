"""Tests for system/loggerd/xattr_cache.py - extended attribute caching."""
import errno
import tempfile
import os
import unittest
from unittest.mock import patch, MagicMock

from openpilot.system.loggerd.xattr_cache import getxattr, setxattr, _cached_attributes


class TestGetXattr(unittest.TestCase):
  """Test getxattr function."""

  def setUp(self):
    """Clear cache before each test."""
    _cached_attributes.clear()

  @patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr')
  def test_getxattr_returns_value(self, mock_xattr):
    """Test getxattr returns attribute value."""
    mock_xattr.return_value = b"test_value"

    result = getxattr("/path/to/file", "user.test")

    self.assertEqual(result, b"test_value")
    mock_xattr.assert_called_once_with("/path/to/file", "user.test")

  @patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr')
  def test_getxattr_caches_result(self, mock_xattr):
    """Test getxattr caches the result."""
    mock_xattr.return_value = b"cached_value"

    # First call
    result1 = getxattr("/path/to/file", "user.attr")
    # Second call should use cache
    result2 = getxattr("/path/to/file", "user.attr")

    self.assertEqual(result1, b"cached_value")
    self.assertEqual(result2, b"cached_value")
    # Should only call xattr.getxattr once
    mock_xattr.assert_called_once()

  @patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr')
  def test_getxattr_returns_none_for_enodata(self, mock_xattr):
    """Test getxattr returns None when attribute not set (ENODATA)."""
    error = OSError()
    error.errno = errno.ENODATA
    mock_xattr.side_effect = error

    result = getxattr("/path/to/file", "user.missing")

    self.assertIsNone(result)

  @patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr')
  def test_getxattr_caches_none(self, mock_xattr):
    """Test getxattr caches None result."""
    error = OSError()
    error.errno = errno.ENODATA
    mock_xattr.side_effect = error

    # First call
    result1 = getxattr("/path/to/file", "user.missing")
    # Second call should use cache
    result2 = getxattr("/path/to/file", "user.missing")

    self.assertIsNone(result1)
    self.assertIsNone(result2)
    # Should only call xattr.getxattr once
    mock_xattr.assert_called_once()

  @patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr')
  def test_getxattr_raises_other_errors(self, mock_xattr):
    """Test getxattr raises non-ENODATA errors."""
    error = OSError()
    error.errno = errno.EACCES  # Permission denied
    mock_xattr.side_effect = error

    with self.assertRaises(OSError) as ctx:
      getxattr("/path/to/file", "user.attr")

    self.assertEqual(ctx.exception.errno, errno.EACCES)

  @patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr')
  def test_getxattr_different_paths_cached_separately(self, mock_xattr):
    """Test different paths are cached separately."""
    mock_xattr.return_value = b"value"

    getxattr("/path/one", "user.attr")
    getxattr("/path/two", "user.attr")

    # Should call twice for different paths
    self.assertEqual(mock_xattr.call_count, 2)

  @patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr')
  def test_getxattr_different_attrs_cached_separately(self, mock_xattr):
    """Test different attributes are cached separately."""
    mock_xattr.return_value = b"value"

    getxattr("/path/file", "user.attr1")
    getxattr("/path/file", "user.attr2")

    # Should call twice for different attributes
    self.assertEqual(mock_xattr.call_count, 2)


class TestSetXattr(unittest.TestCase):
  """Test setxattr function."""

  def setUp(self):
    """Clear cache before each test."""
    _cached_attributes.clear()

  @patch('openpilot.system.loggerd.xattr_cache.xattr.setxattr')
  def test_setxattr_calls_xattr(self, mock_xattr):
    """Test setxattr calls xattr.setxattr."""
    setxattr("/path/to/file", "user.test", b"value")

    mock_xattr.assert_called_once_with("/path/to/file", "user.test", b"value")

  @patch('openpilot.system.loggerd.xattr_cache.xattr.setxattr')
  @patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr')
  def test_setxattr_invalidates_cache(self, mock_getxattr, mock_setxattr):
    """Test setxattr invalidates cached value."""
    mock_getxattr.return_value = b"old_value"

    # Populate cache
    getxattr("/path/file", "user.attr")
    self.assertEqual(_cached_attributes[("/path/file", "user.attr")], b"old_value")

    # Set new value
    setxattr("/path/file", "user.attr", b"new_value")

    # Cache should be cleared for this key
    self.assertNotIn(("/path/file", "user.attr"), _cached_attributes)

  @patch('openpilot.system.loggerd.xattr_cache.xattr.setxattr')
  def test_setxattr_does_not_error_on_uncached(self, mock_xattr):
    """Test setxattr works when key not in cache."""
    # Should not raise even though key is not cached
    setxattr("/new/path", "user.new", b"value")

    mock_xattr.assert_called_once()

  @patch('openpilot.system.loggerd.xattr_cache.xattr.setxattr')
  @patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr')
  def test_setxattr_only_invalidates_specific_key(self, mock_getxattr, mock_setxattr):
    """Test setxattr only invalidates the specific key."""
    mock_getxattr.return_value = b"value"

    # Populate cache with multiple keys
    getxattr("/path/file", "user.attr1")
    getxattr("/path/file", "user.attr2")

    # Set only one attribute
    setxattr("/path/file", "user.attr1", b"new")

    # Only attr1 should be cleared
    self.assertNotIn(("/path/file", "user.attr1"), _cached_attributes)
    self.assertIn(("/path/file", "user.attr2"), _cached_attributes)


class TestCacheIntegration(unittest.TestCase):
  """Integration tests for xattr caching behavior."""

  def setUp(self):
    """Clear cache before each test."""
    _cached_attributes.clear()

  @patch('openpilot.system.loggerd.xattr_cache.xattr.setxattr')
  @patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr')
  def test_set_then_get_refetches(self, mock_getxattr, mock_setxattr):
    """Test that get after set refetches the value."""
    mock_getxattr.side_effect = [b"old_value", b"new_value"]

    # Get initial value
    result1 = getxattr("/path", "user.attr")
    self.assertEqual(result1, b"old_value")

    # Set new value (invalidates cache)
    setxattr("/path", "user.attr", b"new_value")

    # Get should refetch
    result2 = getxattr("/path", "user.attr")
    self.assertEqual(result2, b"new_value")

    # Should have called getxattr twice
    self.assertEqual(mock_getxattr.call_count, 2)


if __name__ == '__main__':
  unittest.main()
