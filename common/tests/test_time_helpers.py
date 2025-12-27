"""Tests for time_helpers.py - time validation utilities."""
import datetime
import unittest
from unittest.mock import patch, MagicMock

from openpilot.common.time_helpers import MIN_DATE, min_date, system_time_valid


class TestMinDate(unittest.TestCase):
  """Test MIN_DATE constant."""

  def test_min_date_is_datetime(self):
    """Test MIN_DATE is a datetime object."""
    self.assertIsInstance(MIN_DATE, datetime.datetime)

  def test_min_date_year(self):
    """Test MIN_DATE year is 2025."""
    self.assertEqual(MIN_DATE.year, 2025)


class TestMinDateFunction(unittest.TestCase):
  """Test min_date function."""

  @patch('openpilot.common.time_helpers.Path')
  def test_min_date_no_systemd(self, mock_path):
    """Test min_date returns MIN_DATE when systemd doesn't exist."""
    mock_path.return_value.exists.return_value = False

    result = min_date()

    self.assertEqual(result, MIN_DATE)

  @patch('openpilot.common.time_helpers.Path')
  def test_min_date_with_old_systemd(self, mock_path):
    """Test min_date returns MIN_DATE when systemd is older."""
    mock_path_instance = MagicMock()
    mock_path.return_value = mock_path_instance
    mock_path_instance.exists.return_value = True
    # Set mtime to an old date
    mock_stat = MagicMock()
    mock_stat.st_mtime = datetime.datetime(2020, 1, 1).timestamp()
    mock_path_instance.stat.return_value = mock_stat

    result = min_date()

    self.assertEqual(result, MIN_DATE)

  @patch('openpilot.common.time_helpers.Path')
  def test_min_date_with_newer_systemd(self, mock_path):
    """Test min_date returns systemd date + 1 day when newer."""
    mock_path_instance = MagicMock()
    mock_path.return_value = mock_path_instance
    mock_path_instance.exists.return_value = True
    # Set mtime to a newer date than MIN_DATE
    newer_date = datetime.datetime(2025, 6, 1)
    mock_stat = MagicMock()
    mock_stat.st_mtime = newer_date.timestamp()
    mock_path_instance.stat.return_value = mock_stat

    result = min_date()

    expected = newer_date + datetime.timedelta(days=1)
    self.assertEqual(result, expected)


class TestSystemTimeValid(unittest.TestCase):
  """Test system_time_valid function."""

  @patch('openpilot.common.time_helpers.min_date')
  @patch('openpilot.common.time_helpers.datetime')
  def test_system_time_valid_when_future(self, mock_datetime, mock_min_date):
    """Test returns True when system time is after min_date."""
    mock_min_date.return_value = datetime.datetime(2025, 1, 1)
    mock_datetime.datetime.now.return_value = datetime.datetime(2025, 12, 1)

    result = system_time_valid()

    self.assertTrue(result)

  @patch('openpilot.common.time_helpers.min_date')
  @patch('openpilot.common.time_helpers.datetime')
  def test_system_time_invalid_when_past(self, mock_datetime, mock_min_date):
    """Test returns False when system time is before min_date."""
    mock_min_date.return_value = datetime.datetime(2025, 6, 1)
    mock_datetime.datetime.now.return_value = datetime.datetime(2020, 1, 1)

    result = system_time_valid()

    self.assertFalse(result)


if __name__ == '__main__':
  unittest.main()
