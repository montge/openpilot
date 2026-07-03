"""Tests for tools/lib/file_sources.py - file source utilities."""

import pytest

from openpilot.tools.lib.file_sources import (
  comma_api_source,
  comma_car_segments_source,
  eval_source,
  internal_source,
  openpilotci_source,
)
from openpilot.tools.lib.route import FileName, SegmentRange


class TestEvalSource:
  """Test eval_source function."""

  def test_eval_source_with_list_urls(self, mocker):
    """Test eval_source with list of URLs per segment."""
    mocker.patch('openpilot.tools.lib.file_sources.file_exists', side_effect=[False, True])

    files = {0: ["https://example.com/rlog.zst", "https://example.com/rlog.bz2"]}
    result = eval_source(files)

    assert result == {0: "https://example.com/rlog.bz2"}

  def test_eval_source_with_string_url(self, mocker):
    """Test eval_source with single string URL per segment."""
    mocker.patch('openpilot.tools.lib.file_sources.file_exists', return_value=True)

    files = {0: "https://example.com/file.txt"}
    result = eval_source(files)

    assert result == {0: "https://example.com/file.txt"}

  def test_eval_source_no_valid_file(self, mocker):
    """Test eval_source when no files exist."""
    mocker.patch('openpilot.tools.lib.file_sources.file_exists', return_value=False)

    files = {0: ["https://example.com/missing1", "https://example.com/missing2"]}
    result = eval_source(files)

    assert result == {}

  def test_eval_source_multiple_segments(self, mocker):
    """Test eval_source with multiple segments."""
    mocker.patch('openpilot.tools.lib.file_sources.file_exists', side_effect=[True, True])

    files = {
      0: ["https://example.com/seg0/file"],
      1: ["https://example.com/seg1/file"],
    }
    result = eval_source(files)

    assert len(result) == 2
    assert 0 in result
    assert 1 in result


class TestInternalSource:
  """Test internal_source function."""

  def test_internal_source_not_available(self, mocker):
    """Test internal_source raises when not available."""
    mocker.patch('openpilot.tools.lib.file_sources.internal_source_available', return_value=False)

    sr = SegmentRange("aaaaaaaaaaaaaaaa/2024-01-15--12-30-45/0")

    with pytest.raises(Exception, match="Internal source not available"):
      internal_source(sr, [0], FileName.RLOG)

  def test_internal_source_available(self, mocker):
    """Test internal_source returns URLs when available."""
    mocker.patch('openpilot.tools.lib.file_sources.internal_source_available', return_value=True)
    mocker.patch('openpilot.tools.lib.file_sources.file_exists', return_value=True)

    sr = SegmentRange("bbbbbbbbbbbbbbbb/2024-02-20--08-15-30/0")

    result = internal_source(sr, [0], FileName.RLOG, endpoint_url="https://internal.example.com/")

    assert 0 in result
    assert "bbbbbbbbbbbbbbbb" in result[0]
    assert "2024-02-20--08-15-30" in result[0]


class TestOpenpilotciSource:
  """Test openpilotci_source function."""

  def test_openpilotci_source(self, mocker):
    """Test openpilotci_source returns URLs."""
    mocker.patch('openpilot.tools.lib.file_sources.get_url', return_value="https://ci.example.com/file")
    mocker.patch('openpilot.tools.lib.file_sources.file_exists', return_value=True)

    sr = SegmentRange("cccccccccccccccc/2024-03-10--14-45-00/0")

    result = openpilotci_source(sr, [0], FileName.RLOG)

    assert 0 in result


class TestCommaCarSegmentsSource:
  """Test comma_car_segments_source function."""

  def test_comma_car_segments_source(self, mocker):
    """Test comma_car_segments_source returns URLs."""
    mocker.patch('openpilot.tools.lib.file_sources.get_comma_segments_url', return_value="https://segments.example.com/file")
    mocker.patch('openpilot.tools.lib.file_sources.file_exists', return_value=True)

    sr = SegmentRange("dddddddddddddddd/2024-04-05--10-20-30/0")

    result = comma_car_segments_source(sr, [0], FileName.RLOG)

    assert 0 in result


class TestCommaApiSource:
  """Test comma_api_source function."""

  def test_comma_api_source_rlog(self, mocker):
    """Test comma_api_source with RLOG files."""
    mock_route = mocker.MagicMock()
    mock_route.log_paths.return_value = ["https://api.example.com/seg0/rlog.zst", "https://api.example.com/seg1/rlog.zst"]
    mocker.patch('openpilot.tools.lib.file_sources.Route', return_value=mock_route)

    sr = SegmentRange("eeeeeeeeeeeeeeee/2024-05-15--16-30-00/0:1")

    result = comma_api_source(sr, [0, 1], FileName.RLOG)

    assert len(result) == 2

  def test_comma_api_source_qlog(self, mocker):
    """Test comma_api_source with QLOG files."""
    mock_route = mocker.MagicMock()
    mock_route.qlog_paths.return_value = ["https://api.example.com/seg0/qlog.zst"]
    mocker.patch('openpilot.tools.lib.file_sources.Route', return_value=mock_route)

    sr = SegmentRange("ffffffffffffffff/2024-06-20--09-00-00/0")

    result = comma_api_source(sr, [0], FileName.QLOG)

    assert len(result) == 1

  def test_comma_api_source_missing_segment(self, mocker):
    """Test comma_api_source skips None paths."""
    mock_route = mocker.MagicMock()
    mock_route.log_paths.return_value = ["https://api.example.com/seg0/rlog.zst", None]
    mocker.patch('openpilot.tools.lib.file_sources.Route', return_value=mock_route)

    sr = SegmentRange("0000000000000000/2024-07-01--12-00-00/0:1")

    result = comma_api_source(sr, [0, 1], FileName.RLOG)

    assert len(result) == 1
    assert 0 in result
    assert 1 not in result
