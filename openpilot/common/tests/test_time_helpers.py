"""Tests for time_helpers.py - time validation utilities."""

import datetime


from openpilot.common.time_helpers import MIN_DATE, min_date, system_time_valid


class TestMinDate:
  """Test MIN_DATE constant."""

  def test_min_date_is_datetime(self):
    """Test MIN_DATE is a datetime object."""
    assert isinstance(MIN_DATE, datetime.datetime)

  def test_min_date_year(self):
    """Test MIN_DATE year is 2025."""
    assert MIN_DATE.year == 2025


class TestMinDateFunction:
  """Test min_date function."""

  def test_min_date_no_systemd(self, mocker):
    """Test min_date returns MIN_DATE when systemd doesn't exist."""
    mock_path = mocker.patch('openpilot.common.time_helpers.Path')
    mock_path.return_value.exists.return_value = False

    result = min_date()

    assert result == MIN_DATE

  def test_min_date_with_old_systemd(self, mocker):
    """Test min_date returns MIN_DATE when systemd is older."""
    mock_path = mocker.patch('openpilot.common.time_helpers.Path')
    mock_path_instance = mocker.MagicMock()
    mock_path.return_value = mock_path_instance
    mock_path_instance.exists.return_value = True
    # Set mtime to an old date
    mock_stat = mocker.MagicMock()
    mock_stat.st_mtime = datetime.datetime(2020, 1, 1).timestamp()
    mock_path_instance.stat.return_value = mock_stat

    result = min_date()

    assert result == MIN_DATE

  def test_min_date_with_newer_systemd(self, mocker):
    """Test min_date returns systemd date + 1 day when newer."""
    mock_path = mocker.patch('openpilot.common.time_helpers.Path')
    mock_path_instance = mocker.MagicMock()
    mock_path.return_value = mock_path_instance
    mock_path_instance.exists.return_value = True
    # Set mtime to a newer date than MIN_DATE
    newer_date = datetime.datetime(2025, 6, 1)
    mock_stat = mocker.MagicMock()
    mock_stat.st_mtime = newer_date.timestamp()
    mock_path_instance.stat.return_value = mock_stat

    result = min_date()

    expected = newer_date + datetime.timedelta(days=1)
    assert result == expected


class TestSystemTimeValid:
  """Test system_time_valid function."""

  def test_system_time_valid_when_future(self, mocker):
    """Test returns True when system time is after min_date."""
    mock_min_date = mocker.patch('openpilot.common.time_helpers.min_date')
    mock_datetime = mocker.patch('openpilot.common.time_helpers.datetime')
    mock_min_date.return_value = datetime.datetime(2025, 1, 1)
    mock_datetime.datetime.now.return_value = datetime.datetime(2025, 12, 1)

    result = system_time_valid()

    assert result is True

  def test_system_time_invalid_when_past(self, mocker):
    """Test returns False when system time is before min_date."""
    mock_min_date = mocker.patch('openpilot.common.time_helpers.min_date')
    mock_datetime = mocker.patch('openpilot.common.time_helpers.datetime')
    mock_min_date.return_value = datetime.datetime(2025, 6, 1)
    mock_datetime.datetime.now.return_value = datetime.datetime(2020, 1, 1)

    result = system_time_valid()

    assert result is False
