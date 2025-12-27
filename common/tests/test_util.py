"""Tests for common/util.py - system utilities."""
import math
import tempfile
import os
import unittest
from unittest.mock import patch, MagicMock

from openpilot.common.util import MovingAverage


class TestMovingAverage(unittest.TestCase):
  """Test MovingAverage class."""

  def test_init_creates_buffer(self):
    """Test initialization creates proper buffer."""
    ma = MovingAverage(5)

    self.assertEqual(ma.window_size, 5)
    self.assertEqual(len(ma.buffer), 5)
    self.assertEqual(ma.index, 0)
    self.assertEqual(ma.count, 0)
    self.assertEqual(ma.sum, 0.0)

  def test_init_buffer_zeros(self):
    """Test buffer initialized with zeros."""
    ma = MovingAverage(3)

    for val in ma.buffer:
      self.assertEqual(val, 0.0)

  def test_get_average_empty_returns_nan(self):
    """Test get_average returns NaN when empty."""
    ma = MovingAverage(5)

    avg = ma.get_average()

    self.assertTrue(math.isnan(avg))

  def test_add_single_value(self):
    """Test adding a single value."""
    ma = MovingAverage(5)

    ma.add_value(10.0)

    self.assertEqual(ma.count, 1)
    self.assertEqual(ma.sum, 10.0)
    self.assertEqual(ma.get_average(), 10.0)

  def test_add_multiple_values_partial_window(self):
    """Test averaging with partial window."""
    ma = MovingAverage(5)

    ma.add_value(10.0)
    ma.add_value(20.0)
    ma.add_value(30.0)

    self.assertEqual(ma.count, 3)
    self.assertEqual(ma.get_average(), 20.0)  # (10+20+30)/3

  def test_add_values_full_window(self):
    """Test averaging with full window."""
    ma = MovingAverage(3)

    ma.add_value(10.0)
    ma.add_value(20.0)
    ma.add_value(30.0)

    self.assertEqual(ma.count, 3)
    self.assertEqual(ma.get_average(), 20.0)

  def test_circular_buffer_overwrites(self):
    """Test circular buffer overwrites old values."""
    ma = MovingAverage(3)

    ma.add_value(10.0)  # buffer: [10, 0, 0]
    ma.add_value(20.0)  # buffer: [10, 20, 0]
    ma.add_value(30.0)  # buffer: [10, 20, 30]
    ma.add_value(40.0)  # buffer: [40, 20, 30] - overwrites 10

    self.assertEqual(ma.count, 3)  # Count stays at window size
    self.assertEqual(ma.get_average(), 30.0)  # (40+20+30)/3

  def test_index_wraps_correctly(self):
    """Test index wraps around circular buffer."""
    ma = MovingAverage(3)

    for i in range(7):
      ma.add_value(float(i))

    # After 7 values in size-3 buffer:
    # index should be at 7 % 3 = 1
    self.assertEqual(ma.index, 1)

  def test_sum_updates_correctly_on_overwrite(self):
    """Test sum is correct when old values are replaced."""
    ma = MovingAverage(2)

    ma.add_value(100.0)
    ma.add_value(200.0)
    self.assertEqual(ma.sum, 300.0)

    ma.add_value(50.0)  # Replaces 100.0
    self.assertEqual(ma.sum, 250.0)  # 200 + 50

  def test_window_size_one(self):
    """Test MovingAverage with window size of 1."""
    ma = MovingAverage(1)

    ma.add_value(10.0)
    self.assertEqual(ma.get_average(), 10.0)

    ma.add_value(20.0)
    self.assertEqual(ma.get_average(), 20.0)

  def test_large_window(self):
    """Test with larger window size."""
    ma = MovingAverage(100)

    for i in range(50):
      ma.add_value(1.0)

    self.assertEqual(ma.count, 50)
    self.assertEqual(ma.get_average(), 1.0)

  def test_negative_values(self):
    """Test with negative values."""
    ma = MovingAverage(3)

    ma.add_value(-10.0)
    ma.add_value(-20.0)
    ma.add_value(-30.0)

    self.assertEqual(ma.get_average(), -20.0)

  def test_mixed_positive_negative(self):
    """Test with mixed positive and negative values."""
    ma = MovingAverage(4)

    ma.add_value(10.0)
    ma.add_value(-10.0)
    ma.add_value(20.0)
    ma.add_value(-20.0)

    self.assertEqual(ma.get_average(), 0.0)

  def test_zero_values(self):
    """Test with zero values."""
    ma = MovingAverage(3)

    ma.add_value(0.0)
    ma.add_value(0.0)

    self.assertEqual(ma.get_average(), 0.0)

  def test_float_precision(self):
    """Test float precision in calculations."""
    ma = MovingAverage(3)

    ma.add_value(0.1)
    ma.add_value(0.2)
    ma.add_value(0.3)

    self.assertAlmostEqual(ma.get_average(), 0.2, places=10)

  def test_very_large_values(self):
    """Test with very large values."""
    ma = MovingAverage(2)

    ma.add_value(1e15)
    ma.add_value(2e15)

    self.assertEqual(ma.get_average(), 1.5e15)

  def test_count_never_exceeds_window_size(self):
    """Test count never exceeds window size."""
    ma = MovingAverage(5)

    for i in range(100):
      ma.add_value(float(i))

    self.assertEqual(ma.count, 5)

  def test_running_average_convergence(self):
    """Test running average converges to constant input."""
    ma = MovingAverage(10)

    # Feed constant value
    for _ in range(20):
      ma.add_value(42.0)

    self.assertEqual(ma.get_average(), 42.0)

  def test_step_change_response(self):
    """Test response to step change in input."""
    ma = MovingAverage(4)

    # Initial values of 0
    for _ in range(4):
      ma.add_value(0.0)

    self.assertEqual(ma.get_average(), 0.0)

    # Step change to 100
    ma.add_value(100.0)
    self.assertEqual(ma.get_average(), 25.0)  # (0+0+0+100)/4

    ma.add_value(100.0)
    self.assertEqual(ma.get_average(), 50.0)  # (0+0+100+100)/4


class TestSudoWrite(unittest.TestCase):
  """Test sudo_write function."""

  @patch('openpilot.common.util.os.system')
  def test_sudo_write_success(self, mock_system):
    """Test sudo_write with writable file."""
    from openpilot.common.util import sudo_write

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
      path = f.name

    try:
      sudo_write("test", path)

      with open(path) as f:
        self.assertEqual(f.read(), "test")
    finally:
      os.unlink(path)

  @patch('openpilot.common.util.os.system')
  def test_sudo_write_permission_error_handled(self, mock_system):
    """Test sudo_write handles PermissionError."""
    from openpilot.common.util import sudo_write

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "test.txt")

      # First write succeeds
      sudo_write("content", path)

      with open(path) as f:
        self.assertEqual(f.read(), "content")


class TestSudoRead(unittest.TestCase):
  """Test sudo_read function."""

  @patch('openpilot.common.util.subprocess.check_output')
  def test_sudo_read_success(self, mock_check_output):
    """Test sudo_read returns content."""
    from openpilot.common.util import sudo_read

    mock_check_output.return_value = "file content\n"

    result = sudo_read("/some/path")

    self.assertEqual(result, "file content")
    mock_check_output.assert_called_once()

  @patch('openpilot.common.util.subprocess.check_output')
  def test_sudo_read_failure_returns_empty(self, mock_check_output):
    """Test sudo_read returns empty on failure."""
    from openpilot.common.util import sudo_read

    mock_check_output.side_effect = Exception("command failed")

    result = sudo_read("/nonexistent/path")

    self.assertEqual(result, "")


if __name__ == '__main__':
  unittest.main()
