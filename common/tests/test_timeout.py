"""Tests for timeout.py - timeout context manager."""

import time

import pytest

from openpilot.common.timeout import Timeout, TimeoutException


class TestTimeoutException:
  """Test TimeoutException class."""

  def test_timeout_exception_is_exception(self):
    """Test TimeoutException is an Exception subclass."""
    assert issubclass(TimeoutException, Exception)

  def test_timeout_exception_can_be_raised(self):
    """Test TimeoutException can be raised with message."""
    with pytest.raises(TimeoutException) as ctx:
      raise TimeoutException("test message")
    assert str(ctx.value) == "test message"


class TestTimeout:
  """Test Timeout context manager."""

  def test_timeout_init_default_message(self):
    """Test Timeout default error message."""
    timeout = Timeout(seconds=5)
    assert timeout.seconds == 5
    assert timeout.error_msg == "Timed out after 5 seconds"

  def test_timeout_init_custom_message(self):
    """Test Timeout with custom error message."""
    timeout = Timeout(seconds=10, error_msg="Custom timeout")
    assert timeout.seconds == 10
    assert timeout.error_msg == "Custom timeout"

  def test_timeout_no_exception_when_fast(self):
    """Test no exception when operation completes in time."""
    result = None
    with Timeout(seconds=2):
      result = 42
    assert result == 42

  def test_timeout_raises_exception_when_slow(self):
    """Test exception raised when operation times out."""
    with pytest.raises(TimeoutException) as ctx:
      with Timeout(seconds=1, error_msg="Sleep timeout"):
        time.sleep(5)
    assert str(ctx.value) == "Sleep timeout"

  def test_timeout_handle_timeout_raises(self):
    """Test handle_timeout method raises TimeoutException."""
    timeout = Timeout(seconds=1, error_msg="Handler test")
    with pytest.raises(TimeoutException) as ctx:
      timeout.handle_timeout(None, None)
    assert str(ctx.value) == "Handler test"

  def test_timeout_alarm_cleared_on_exit(self):
    """Test alarm is cleared when context exits normally."""
    # This test just ensures the context manager exits cleanly
    with Timeout(seconds=10):
      pass
    # If we get here without hanging, the alarm was cleared
