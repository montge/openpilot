"""Tests for tools/lib/exceptions.py - custom exception classes."""

import pytest

from openpilot.tools.lib.exceptions import DataUnreadableError


class TestDataUnreadableError:
  """Test DataUnreadableError exception class."""

  def test_is_exception_subclass(self):
    """DataUnreadableError should be a subclass of Exception."""
    assert issubclass(DataUnreadableError, Exception)

  def test_can_be_raised(self):
    """DataUnreadableError can be raised and caught."""
    with pytest.raises(DataUnreadableError):
      raise DataUnreadableError("test message")

  def test_message_preserved(self):
    """Exception message should be preserved."""
    msg = "Unable to read data from source"
    try:
      raise DataUnreadableError(msg)
    except DataUnreadableError as e:
      assert str(e) == msg
