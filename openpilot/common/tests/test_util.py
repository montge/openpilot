"""Tests for common/util.py - system utilities."""

import math
import tempfile
import os

import pytest

from openpilot.common.util import MovingAverage


class TestMovingAverage:
  """Test MovingAverage class."""

  def test_init_creates_buffer(self):
    """Test initialization creates proper buffer."""
    ma = MovingAverage(5)

    assert ma.window_size == 5
    assert len(ma.buffer) == 5
    assert ma.index == 0
    assert ma.count == 0
    assert ma.sum == 0.0

  def test_init_buffer_zeros(self):
    """Test buffer initialized with zeros."""
    ma = MovingAverage(3)

    for val in ma.buffer:
      assert val == 0.0

  def test_get_average_empty_returns_nan(self):
    """Test get_average returns NaN when empty."""
    ma = MovingAverage(5)

    avg = ma.get_average()

    assert math.isnan(avg)

  def test_add_single_value(self):
    """Test adding a single value."""
    ma = MovingAverage(5)

    ma.add_value(10.0)

    assert ma.count == 1
    assert ma.sum == 10.0
    assert ma.get_average() == 10.0

  def test_add_multiple_values_partial_window(self):
    """Test averaging with partial window."""
    ma = MovingAverage(5)

    ma.add_value(10.0)
    ma.add_value(20.0)
    ma.add_value(30.0)

    assert ma.count == 3
    assert ma.get_average() == 20.0  # (10+20+30)/3

  def test_add_values_full_window(self):
    """Test averaging with full window."""
    ma = MovingAverage(3)

    ma.add_value(10.0)
    ma.add_value(20.0)
    ma.add_value(30.0)

    assert ma.count == 3
    assert ma.get_average() == 20.0

  def test_circular_buffer_overwrites(self):
    """Test circular buffer overwrites old values."""
    ma = MovingAverage(3)

    ma.add_value(10.0)  # buffer: [10, 0, 0]
    ma.add_value(20.0)  # buffer: [10, 20, 0]
    ma.add_value(30.0)  # buffer: [10, 20, 30]
    ma.add_value(40.0)  # buffer: [40, 20, 30] - overwrites 10

    assert ma.count == 3  # Count stays at window size
    assert ma.get_average() == 30.0  # (40+20+30)/3

  def test_index_wraps_correctly(self):
    """Test index wraps around circular buffer."""
    ma = MovingAverage(3)

    for i in range(7):
      ma.add_value(float(i))

    # After 7 values in size-3 buffer:
    # index should be at 7 % 3 = 1
    assert ma.index == 1

  def test_sum_updates_correctly_on_overwrite(self):
    """Test sum is correct when old values are replaced."""
    ma = MovingAverage(2)

    ma.add_value(100.0)
    ma.add_value(200.0)
    assert ma.sum == 300.0

    ma.add_value(50.0)  # Replaces 100.0
    assert ma.sum == 250.0  # 200 + 50

  def test_window_size_one(self):
    """Test MovingAverage with window size of 1."""
    ma = MovingAverage(1)

    ma.add_value(10.0)
    assert ma.get_average() == 10.0

    ma.add_value(20.0)
    assert ma.get_average() == 20.0

  def test_large_window(self):
    """Test with larger window size."""
    ma = MovingAverage(100)

    for _ in range(50):
      ma.add_value(1.0)

    assert ma.count == 50
    assert ma.get_average() == 1.0

  def test_negative_values(self):
    """Test with negative values."""
    ma = MovingAverage(3)

    ma.add_value(-10.0)
    ma.add_value(-20.0)
    ma.add_value(-30.0)

    assert ma.get_average() == -20.0

  def test_mixed_positive_negative(self):
    """Test with mixed positive and negative values."""
    ma = MovingAverage(4)

    ma.add_value(10.0)
    ma.add_value(-10.0)
    ma.add_value(20.0)
    ma.add_value(-20.0)

    assert ma.get_average() == 0.0

  def test_zero_values(self):
    """Test with zero values."""
    ma = MovingAverage(3)

    ma.add_value(0.0)
    ma.add_value(0.0)

    assert ma.get_average() == 0.0

  def test_float_precision(self):
    """Test float precision in calculations."""
    ma = MovingAverage(3)

    ma.add_value(0.1)
    ma.add_value(0.2)
    ma.add_value(0.3)

    assert ma.get_average() == pytest.approx(0.2, abs=1e-10)

  def test_very_large_values(self):
    """Test with very large values."""
    ma = MovingAverage(2)

    ma.add_value(1e15)
    ma.add_value(2e15)

    assert ma.get_average() == 1.5e15

  def test_count_never_exceeds_window_size(self):
    """Test count never exceeds window size."""
    ma = MovingAverage(5)

    for i in range(100):
      ma.add_value(float(i))

    assert ma.count == 5

  def test_running_average_convergence(self):
    """Test running average converges to constant input."""
    ma = MovingAverage(10)

    # Feed constant value
    for _ in range(20):
      ma.add_value(42.0)

    assert ma.get_average() == 42.0

  def test_step_change_response(self):
    """Test response to step change in input."""
    ma = MovingAverage(4)

    # Initial values of 0
    for _ in range(4):
      ma.add_value(0.0)

    assert ma.get_average() == 0.0

    # Step change to 100
    ma.add_value(100.0)
    assert ma.get_average() == 25.0  # (0+0+0+100)/4

    ma.add_value(100.0)
    assert ma.get_average() == 50.0  # (0+0+100+100)/4


class TestSudoWrite:
  """Test sudo_write function."""

  def test_sudo_write_success(self, mocker):
    """Test sudo_write with writable file."""
    mocker.patch('openpilot.common.util.os.system')
    from openpilot.common.util import sudo_write

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
      path = f.name

    try:
      sudo_write("test", path)

      with open(path) as f:
        assert f.read() == "test"
    finally:
      os.unlink(path)

  def test_sudo_write_permission_error_handled(self, mocker):
    """Test sudo_write handles PermissionError."""
    mocker.patch('openpilot.common.util.os.system')
    from openpilot.common.util import sudo_write

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "test.txt")

      # First write succeeds
      sudo_write("content", path)

      with open(path) as f:
        assert f.read() == "content"

  def test_sudo_write_chmod_on_permission_error(self, mocker):
    """Test sudo_write calls chmod on PermissionError then succeeds."""
    mock_system = mocker.patch('openpilot.common.util.os.system')

    # Mock open to fail first, then succeed
    call_count = [0]
    real_open = open

    def mock_open_fn(path, mode='r', *args, **kwargs):
      if mode == 'w' and call_count[0] == 0:
        call_count[0] += 1
        raise PermissionError("Permission denied")
      return real_open(path, mode, *args, **kwargs)

    mocker.patch('builtins.open', mock_open_fn)
    from openpilot.common.util import sudo_write

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "test.txt")
      # Create the file first so second open succeeds
      with real_open(path, 'w') as f:
        f.write("")

      sudo_write("content", path)

      # Should have called chmod
      mock_system.assert_called_once()
      assert "chmod" in mock_system.call_args[0][0]

  def test_sudo_write_fallback_on_double_permission_error(self, mocker):
    """Test sudo_write uses echo fallback on double PermissionError."""
    mock_system = mocker.patch('openpilot.common.util.os.system')

    # Mock open to always fail with PermissionError
    def mock_open_fn(path, mode='r', *args, **kwargs):
      if mode == 'w':
        raise PermissionError("Permission denied")
      return open.__class__(path, mode, *args, **kwargs)

    mocker.patch('builtins.open', mock_open_fn)
    from openpilot.common.util import sudo_write

    sudo_write("testval", "/some/path")

    # Should have called chmod first, then echo fallback
    assert mock_system.call_count == 2
    assert "chmod" in mock_system.call_args_list[0][0][0]
    assert "echo testval" in mock_system.call_args_list[1][0][0]


class TestSudoRead:
  """Test sudo_read function."""

  def test_sudo_read_success(self, mocker):
    """Test sudo_read returns content."""
    mock_check_output = mocker.patch('openpilot.common.util.subprocess.check_output')
    from openpilot.common.util import sudo_read

    mock_check_output.return_value = "file content\n"

    result = sudo_read("/some/path")

    assert result == "file content"
    mock_check_output.assert_called_once()

  def test_sudo_read_failure_returns_empty(self, mocker):
    """Test sudo_read returns empty on failure."""
    mock_check_output = mocker.patch('openpilot.common.util.subprocess.check_output')
    from openpilot.common.util import sudo_read

    mock_check_output.side_effect = Exception("command failed")

    result = sudo_read("/nonexistent/path")

    assert result == ""
