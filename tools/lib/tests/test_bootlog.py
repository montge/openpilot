"""Tests for tools/lib/bootlog.py - bootlog utilities."""

import pytest

from openpilot.tools.lib.bootlog import Bootlog, get_bootlog_from_id, get_bootlogs


class TestBootlogInit:
  """Test Bootlog initialization."""

  def test_init_valid_url_pipe_delimiter(self):
    """Test init with valid URL using pipe delimiter."""
    url = "https://example.com/aaaaaaaaaaaaaaaa|2024-01-15--12-30-45/bootlog"
    bl = Bootlog(url)

    assert bl.dongle_id == "aaaaaaaaaaaaaaaa"
    assert bl.id == "2024-01-15--12-30-45"

  def test_init_valid_url_slash_delimiter(self):
    """Test init with valid URL using slash delimiter."""
    url = "https://example.com/bbbbbbbbbbbbbbbb/2024-02-20--08-15-30/bootlog"
    bl = Bootlog(url)

    assert bl.dongle_id == "bbbbbbbbbbbbbbbb"
    assert bl.id == "2024-02-20--08-15-30"

  def test_init_invalid_url_raises(self):
    """Test init with invalid URL raises exception."""
    with pytest.raises(Exception, match="Unable to parse"):
      Bootlog("invalid-url")


class TestBootlogProperties:
  """Test Bootlog properties."""

  def test_url_property(self):
    """Test url property returns original URL."""
    url = "https://example.com/cccccccccccccccc|2024-03-10--14-45-00/bootlog"
    bl = Bootlog(url)

    assert bl.url == url

  def test_dongle_id_property(self):
    """Test dongle_id property."""
    url = "https://example.com/dddddddddddddddd|2024-04-05--10-20-30/bootlog"
    bl = Bootlog(url)

    assert bl.dongle_id == "dddddddddddddddd"

  def test_id_property(self):
    """Test id property."""
    url = "https://example.com/eeeeeeeeeeeeeeee|2024-05-15--16-30-00/bootlog"
    bl = Bootlog(url)

    assert bl.id == "2024-05-15--16-30-00"


class TestBootlogComparison:
  """Test Bootlog comparison methods."""

  def test_str(self):
    """Test __str__ returns dongle_id/id format."""
    url = "https://example.com/ffffffffffffffff|2024-06-20--09-00-00/bootlog"
    bl = Bootlog(url)

    assert str(bl) == "ffffffffffffffff/2024-06-20--09-00-00"

  def test_eq_same(self):
    """Test equality with same bootlog."""
    url1 = "https://example.com/0000000000000000|2024-07-01--12-00-00/bootlog"
    url2 = "https://other.com/0000000000000000|2024-07-01--12-00-00/bootlog"
    bl1 = Bootlog(url1)
    bl2 = Bootlog(url2)

    assert bl1 == bl2

  def test_eq_different(self):
    """Test equality with different bootlog."""
    url1 = "https://example.com/1111111111111111|2024-08-10--15-30-00/bootlog"
    url2 = "https://example.com/1111111111111111|2024-08-10--15-30-01/bootlog"
    bl1 = Bootlog(url1)
    bl2 = Bootlog(url2)

    assert bl1 != bl2

  def test_eq_non_bootlog(self):
    """Test equality with non-Bootlog returns False."""
    url = "https://example.com/2222222222222222|2024-09-20--18-00-00/bootlog"
    bl = Bootlog(url)

    assert bl != "not a bootlog"
    assert bl != 123

  def test_lt(self):
    """Test less than comparison."""
    url1 = "https://example.com/3333333333333333|2024-10-01--00-00-00/bootlog"
    url2 = "https://example.com/3333333333333333|2024-10-02--00-00-00/bootlog"
    bl1 = Bootlog(url1)
    bl2 = Bootlog(url2)

    assert bl1 < bl2

  def test_lt_non_bootlog(self):
    """Test less than with non-Bootlog returns False."""
    url = "https://example.com/4444444444444444|2024-11-15--14-30-00/bootlog"
    bl = Bootlog(url)

    assert not (bl < "not a bootlog")

  def test_ordering(self):
    """Test bootlogs can be sorted."""
    urls = [
      "https://example.com/5555555555555555|2024-12-03--00-00-00/bootlog",
      "https://example.com/5555555555555555|2024-12-01--00-00-00/bootlog",
      "https://example.com/5555555555555555|2024-12-02--00-00-00/bootlog",
    ]
    bootlogs = [Bootlog(url) for url in urls]
    sorted_bootlogs = sorted(bootlogs)

    assert sorted_bootlogs[0].id == "2024-12-01--00-00-00"
    assert sorted_bootlogs[1].id == "2024-12-02--00-00-00"
    assert sorted_bootlogs[2].id == "2024-12-03--00-00-00"


class TestGetBootlogs:
  """Test get_bootlogs function."""

  def test_get_bootlogs(self, mocker):
    """Test get_bootlogs returns list of Bootlog objects."""
    mock_api = mocker.MagicMock()
    mock_api.get.return_value = [
      "https://example.com/aaaaaaaaaaaaaaaa|2024-01-01--00-00-00/bootlog",
      "https://example.com/aaaaaaaaaaaaaaaa|2024-01-02--00-00-00/bootlog",
    ]
    mocker.patch('openpilot.tools.lib.bootlog.CommaApi', return_value=mock_api)
    mocker.patch('openpilot.tools.lib.bootlog.get_token', return_value="fake_token")

    result = get_bootlogs("aaaaaaaaaaaaaaaa")

    assert len(result) == 2
    assert all(isinstance(b, Bootlog) for b in result)
    mock_api.get.assert_called_once_with('v1/devices/aaaaaaaaaaaaaaaa/bootlogs')


class TestGetBootlogFromId:
  """Test get_bootlog_from_id function."""

  def test_get_bootlog_from_id_found(self, mocker):
    """Test get_bootlog_from_id returns bootlog when found."""
    mock_api = mocker.MagicMock()
    mock_api.get.return_value = [
      "https://example.com/bbbbbbbbbbbbbbbb|2024-01-01--00-00-00/bootlog",
      "https://example.com/bbbbbbbbbbbbbbbb|2024-01-02--00-00-00/bootlog",
    ]
    mocker.patch('openpilot.tools.lib.bootlog.CommaApi', return_value=mock_api)
    mocker.patch('openpilot.tools.lib.bootlog.get_token', return_value="fake_token")

    result = get_bootlog_from_id("bbbbbbbbbbbbbbbb|2024-01-02--00-00-00")

    assert result is not None
    assert result.id == "2024-01-02--00-00-00"

  def test_get_bootlog_from_id_not_found(self, mocker):
    """Test get_bootlog_from_id returns None when not found."""
    mock_api = mocker.MagicMock()
    mock_api.get.return_value = [
      "https://example.com/cccccccccccccccc|2024-01-01--00-00-00/bootlog",
    ]
    mocker.patch('openpilot.tools.lib.bootlog.CommaApi', return_value=mock_api)
    mocker.patch('openpilot.tools.lib.bootlog.get_token', return_value="fake_token")

    result = get_bootlog_from_id("cccccccccccccccc|2024-01-02--00-00-00")

    assert result is None
