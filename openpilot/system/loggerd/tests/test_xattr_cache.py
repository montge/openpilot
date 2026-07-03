"""Tests for system/loggerd/xattr_cache.py - extended attribute caching."""

import errno

import pytest

from openpilot.system.loggerd.xattr_cache import getxattr, setxattr, _cached_attributes


class TestGetXattr:
  """Test getxattr function."""

  def setup_method(self):
    """Clear cache before each test."""
    _cached_attributes.clear()

  def test_getxattr_returns_value(self, mocker):
    """Test getxattr returns attribute value."""
    mock_xattr = mocker.patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr', return_value=b"test_value")

    result = getxattr("/path/to/file", "user.test")

    assert result == b"test_value"
    mock_xattr.assert_called_once_with("/path/to/file", "user.test")

  def test_getxattr_caches_result(self, mocker):
    """Test getxattr caches the result."""
    mock_xattr = mocker.patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr', return_value=b"cached_value")

    # First call
    result1 = getxattr("/path/to/file", "user.attr")
    # Second call should use cache
    result2 = getxattr("/path/to/file", "user.attr")

    assert result1 == b"cached_value"
    assert result2 == b"cached_value"
    # Should only call xattr.getxattr once
    mock_xattr.assert_called_once()

  def test_getxattr_returns_none_for_enodata(self, mocker):
    """Test getxattr returns None when attribute not set (ENODATA)."""
    error = OSError()
    error.errno = errno.ENODATA
    mocker.patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr', side_effect=error)

    result = getxattr("/path/to/file", "user.missing")

    assert result is None

  def test_getxattr_caches_none(self, mocker):
    """Test getxattr caches None result."""
    error = OSError()
    error.errno = errno.ENODATA
    mock_xattr = mocker.patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr', side_effect=error)

    # First call
    result1 = getxattr("/path/to/file", "user.missing")
    # Second call should use cache
    result2 = getxattr("/path/to/file", "user.missing")

    assert result1 is None
    assert result2 is None
    # Should only call xattr.getxattr once
    mock_xattr.assert_called_once()

  def test_getxattr_raises_other_errors(self, mocker):
    """Test getxattr raises non-ENODATA errors."""
    error = OSError()
    error.errno = errno.EACCES  # Permission denied
    mocker.patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr', side_effect=error)

    with pytest.raises(OSError) as ctx:
      getxattr("/path/to/file", "user.attr")

    assert ctx.value.errno == errno.EACCES

  def test_getxattr_different_paths_cached_separately(self, mocker):
    """Test different paths are cached separately."""
    mock_xattr = mocker.patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr', return_value=b"value")

    getxattr("/path/one", "user.attr")
    getxattr("/path/two", "user.attr")

    # Should call twice for different paths
    assert mock_xattr.call_count == 2

  def test_getxattr_different_attrs_cached_separately(self, mocker):
    """Test different attributes are cached separately."""
    mock_xattr = mocker.patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr', return_value=b"value")

    getxattr("/path/file", "user.attr1")
    getxattr("/path/file", "user.attr2")

    # Should call twice for different attributes
    assert mock_xattr.call_count == 2


class TestSetXattr:
  """Test setxattr function."""

  def setup_method(self):
    """Clear cache before each test."""
    _cached_attributes.clear()

  def test_setxattr_calls_xattr(self, mocker):
    """Test setxattr calls xattr.setxattr."""
    mock_xattr = mocker.patch('openpilot.system.loggerd.xattr_cache.xattr.setxattr')
    setxattr("/path/to/file", "user.test", b"value")

    mock_xattr.assert_called_once_with("/path/to/file", "user.test", b"value")

  def test_setxattr_invalidates_cache(self, mocker):
    """Test setxattr invalidates cached value."""
    mocker.patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr', return_value=b"old_value")
    mocker.patch('openpilot.system.loggerd.xattr_cache.xattr.setxattr')

    # Populate cache
    getxattr("/path/file", "user.attr")
    assert _cached_attributes[("/path/file", "user.attr")] == b"old_value"

    # Set new value
    setxattr("/path/file", "user.attr", b"new_value")

    # Cache should be cleared for this key
    assert ("/path/file", "user.attr") not in _cached_attributes

  def test_setxattr_does_not_error_on_uncached(self, mocker):
    """Test setxattr works when key not in cache."""
    mock_xattr = mocker.patch('openpilot.system.loggerd.xattr_cache.xattr.setxattr')
    # Should not raise even though key is not cached
    setxattr("/new/path", "user.new", b"value")

    mock_xattr.assert_called_once()

  def test_setxattr_only_invalidates_specific_key(self, mocker):
    """Test setxattr only invalidates the specific key."""
    mocker.patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr', return_value=b"value")
    mocker.patch('openpilot.system.loggerd.xattr_cache.xattr.setxattr')

    # Populate cache with multiple keys
    getxattr("/path/file", "user.attr1")
    getxattr("/path/file", "user.attr2")

    # Set only one attribute
    setxattr("/path/file", "user.attr1", b"new")

    # Only attr1 should be cleared
    assert ("/path/file", "user.attr1") not in _cached_attributes
    assert ("/path/file", "user.attr2") in _cached_attributes


class TestCacheIntegration:
  """Integration tests for xattr caching behavior."""

  def setup_method(self):
    """Clear cache before each test."""
    _cached_attributes.clear()

  def test_set_then_get_refetches(self, mocker):
    """Test that get after set refetches the value."""
    mock_getxattr = mocker.patch('openpilot.system.loggerd.xattr_cache.xattr.getxattr', side_effect=[b"old_value", b"new_value"])
    mocker.patch('openpilot.system.loggerd.xattr_cache.xattr.setxattr')

    # Get initial value
    result1 = getxattr("/path", "user.attr")
    assert result1 == b"old_value"

    # Set new value (invalidates cache)
    setxattr("/path", "user.attr", b"new_value")

    # Get should refetch
    result2 = getxattr("/path", "user.attr")
    assert result2 == b"new_value"

    # Should have called getxattr twice
    assert mock_getxattr.call_count == 2
