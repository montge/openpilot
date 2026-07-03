"""Tests for tools/lib/log_time_series.py - log message time series utilities."""

import numpy as np

from openpilot.tools.lib.log_time_series import (
  flatten_type_dict,
  get_message_dict,
  append_dict,
  potentially_ragged_array,
)


class TestFlattenTypeDict:
  """Test flatten_type_dict function."""

  def test_flatten_simple_dict(self):
    """Test flattening a simple dict."""
    d = {'a': 1, 'b': 2}
    result = flatten_type_dict(d)

    assert result == {'a': 1, 'b': 2}

  def test_flatten_nested_dict(self):
    """Test flattening a nested dict."""
    d = {'a': {'b': 1, 'c': 2}}
    result = flatten_type_dict(d)

    assert result == {'a/b': 1, 'a/c': 2}

  def test_flatten_list_value(self):
    """Test flattening with list values converts to numpy array."""
    d = {'a': [1, 2, 3]}
    result = flatten_type_dict(d)

    assert 'a' in result
    np.testing.assert_array_equal(result['a'], np.array([1, 2, 3]))

  def test_flatten_deeply_nested(self):
    """Test flattening deeply nested dict."""
    d = {'a': {'b': {'c': 42}}}
    result = flatten_type_dict(d)

    assert result == {'a/b/c': 42}


class TestGetMessageDict:
  """Test get_message_dict function."""

  def test_returns_none_for_qcomGnss(self, mocker):
    """Test get_message_dict returns None for qcomGnss type."""
    mock_msg = mocker.MagicMock()
    mock_msg.valid = True

    result = get_message_dict(mock_msg, 'qcomGnss')

    assert result is None

  def test_returns_none_for_ubloxGnss(self, mocker):
    """Test get_message_dict returns None for ubloxGnss type."""
    mock_msg = mocker.MagicMock()
    mock_msg.valid = True

    result = get_message_dict(mock_msg, 'ubloxGnss')

    assert result is None

  def test_returns_none_when_no_to_dict(self, mocker):
    """Test get_message_dict returns None when message has no to_dict."""
    mock_msg = mocker.MagicMock()
    mock_msg.valid = True
    mock_inner = mocker.MagicMock(spec=[])  # No to_dict attribute
    mock_msg._get.return_value = mock_inner

    result = get_message_dict(mock_msg, 'someType')

    assert result is None

  def test_returns_dict_with_valid_message(self, mocker):
    """Test get_message_dict returns flattened dict for valid message."""
    mock_msg = mocker.MagicMock()
    mock_msg.valid = True
    mock_inner = mocker.MagicMock()
    mock_inner.to_dict.return_value = {'field1': 123, 'field2': 456}
    mock_msg._get.return_value = mock_inner

    result = get_message_dict(mock_msg, 'carState')

    assert result is not None
    assert result['field1'] == 123
    assert result['field2'] == 456
    assert result['_valid'] is True


class TestAppendDict:
  """Test append_dict function."""

  def test_append_to_empty_values(self):
    """Test appending to empty values dict creates group."""
    values = {}
    d = {'a': 1, 'b': 2}

    append_dict('path', 0.5, d, values)

    assert 'path' in values
    assert values['path']['t'] == [0.5]
    assert values['path']['a'] == [1]
    assert values['path']['b'] == [2]

  def test_append_to_existing_group(self):
    """Test appending to existing group."""
    values = {'path': {'t': [0.1], 'a': [10]}}
    d = {'a': 20}

    append_dict('path', 0.2, d, values)

    assert values['path']['t'] == [0.1, 0.2]
    assert values['path']['a'] == [10, 20]

  def test_append_multiple_times(self):
    """Test appending multiple times."""
    values = {}

    append_dict('path', 0.1, {'x': 1}, values)
    append_dict('path', 0.2, {'x': 2}, values)
    append_dict('path', 0.3, {'x': 3}, values)

    assert values['path']['t'] == [0.1, 0.2, 0.3]
    assert values['path']['x'] == [1, 2, 3]


class TestPotentiallyRaggedArray:
  """Test potentially_ragged_array function."""

  def test_homogeneous_array(self):
    """Test with homogeneous array."""
    arr = [[1, 2], [3, 4], [5, 6]]
    result = potentially_ragged_array(arr)

    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4], [5, 6]]))
    assert result.dtype != object

  def test_ragged_array_different_lengths(self):
    """Test with ragged array (different lengths)."""
    arr = [[1, 2], [3, 4, 5], [6]]
    result = potentially_ragged_array(arr)

    # Should fall back to object dtype
    assert result.dtype == object
    assert len(result) == 3

  def test_ragged_array_mixed_types(self):
    """Test with inhomogeneous shapes."""
    arr = [np.array([1, 2]), np.array([3, 4, 5])]
    result = potentially_ragged_array(arr)

    assert result.dtype == object

  def test_simple_1d_array(self):
    """Test with simple 1D array."""
    arr = [1, 2, 3, 4]
    result = potentially_ragged_array(arr)

    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4]))

  def test_with_dtype_argument(self):
    """Test dtype argument is passed through."""
    arr = [1.5, 2.5, 3.5]
    result = potentially_ragged_array(arr, dtype=np.float32)

    assert result.dtype == np.float32
