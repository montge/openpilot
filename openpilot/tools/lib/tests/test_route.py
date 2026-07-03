"""Tests for tools/lib/route.py - route and segment name parsing."""

import os
import tempfile

import pytest

from openpilot.tools.lib.route import (
  FileName,
  Route,
  RouteName,
  Segment,
  SegmentName,
  SegmentRange,
)


class TestFileName:
  """Test FileName constants."""

  def test_rlog_formats(self):
    """Test RLOG file formats."""
    assert "rlog.zst" in FileName.RLOG
    assert "rlog.bz2" in FileName.RLOG

  def test_qlog_formats(self):
    """Test QLOG file formats."""
    assert "qlog.zst" in FileName.QLOG
    assert "qlog.bz2" in FileName.QLOG

  def test_camera_formats(self):
    """Test camera file formats."""
    assert FileName.QCAMERA == ('qcamera.ts',)
    assert FileName.FCAMERA == ('fcamera.hevc',)
    assert FileName.ECAMERA == ('ecamera.hevc',)
    assert FileName.DCAMERA == ('dcamera.hevc',)

  def test_bootlog_formats(self):
    """Test BOOTLOG file formats."""
    assert "bootlog.zst" in FileName.BOOTLOG
    assert "bootlog.bz2" in FileName.BOOTLOG


class TestRouteName:
  """Test RouteName class."""

  def test_parse_pipe_delimiter(self):
    """Test parsing route name with pipe delimiter."""
    route = RouteName("a]a]a]a]a]a]a]a]|2024-01-15--12-30-45")

    assert route.dongle_id == "a]a]a]a]a]a]a]a]"
    assert route.time_str == "2024-01-15--12-30-45"

  def test_parse_slash_delimiter(self):
    """Test parsing route name with slash delimiter."""
    route = RouteName("b]b]b]b]b]b]b]b]/2024-02-20--08-15-30")

    assert route.dongle_id == "b]b]b]b]b]b]b]b]"
    assert route.time_str == "2024-02-20--08-15-30"

  def test_canonical_name(self):
    """Test canonical_name property uses pipe delimiter."""
    route = RouteName("c]c]c]c]c]c]c]c]/2024-03-10--14-45-00")

    assert route.canonical_name == "c]c]c]c]c]c]c]c]|2024-03-10--14-45-00"

  def test_log_id_same_as_time_str(self):
    """Test log_id returns time_str."""
    route = RouteName("d]d]d]d]d]d]d]d]|2024-04-05--10-20-30")

    assert route.log_id == route.time_str
    assert route.log_id == "2024-04-05--10-20-30"

  def test_azure_prefix(self):
    """Test azure_prefix property."""
    route = RouteName("e]e]e]e]e]e]e]e]|2024-05-15--16-30-00")

    assert route.azure_prefix == "e]e]e]e]e]e]e]e]/2024-05-15--16-30-00"

  def test_str_returns_canonical(self):
    """Test __str__ returns canonical name."""
    route = RouteName("f]f]f]f]f]f]f]f]|2024-06-20--09-00-00")

    assert str(route) == route.canonical_name

  def test_invalid_dongle_id_length(self):
    """Test assertion on invalid dongle_id length."""
    with pytest.raises(AssertionError):
      RouteName("short|2024-01-01--00-00-00")

  def test_invalid_time_str_length(self):
    """Test assertion on invalid time_str length."""
    with pytest.raises(AssertionError):
      RouteName("a]a]a]a]a]a]a]a]|2024-01-01")


class TestSegmentName:
  """Test SegmentName class."""

  def test_parse_double_dash_segment(self):
    """Test parsing segment name with -- delimiter."""
    seg = SegmentName("a]a]a]a]a]a]a]a]|2024-01-15--12-30-45--5")

    assert seg.dongle_id == "a]a]a]a]a]a]a]a]"
    assert seg.time_str == "2024-01-15--12-30-45"
    assert seg.segment_num == 5

  def test_parse_slash_segment(self):
    """Test parsing segment name with / delimiter."""
    seg = SegmentName("b]b]b]b]b]b]b]b]|2024-02-20--08-15-30/10")

    assert seg.segment_num == 10
    assert seg.dongle_id == "b]b]b]b]b]b]b]b]"

  def test_canonical_name(self):
    """Test canonical_name uses -- delimiter."""
    seg = SegmentName("c]c]c]c]c]c]c]c]/2024-03-10--14-45-00/3")

    assert seg.canonical_name == "c]c]c]c]c]c]c]c]|2024-03-10--14-45-00--3"

  def test_data_name(self):
    """Test data_name uses / delimiter."""
    seg = SegmentName("d]d]d]d]d]d]d]d]|2024-04-05--10-20-30--7")

    assert seg.data_name == "d]d]d]d]d]d]d]d]|2024-04-05--10-20-30/7"

  def test_azure_prefix(self):
    """Test azure_prefix property."""
    seg = SegmentName("e]e]e]e]e]e]e]e]|2024-05-15--16-30-00--2")

    assert seg.azure_prefix == "e]e]e]e]e]e]e]e]/2024-05-15--16-30-00/2"

  def test_log_id_same_as_time_str(self):
    """Test log_id returns time_str."""
    seg = SegmentName("f]f]f]f]f]f]f]f]|2024-06-20--09-00-00--0")

    assert seg.log_id == seg.time_str

  def test_route_name_property(self):
    """Test route_name property returns RouteName."""
    seg = SegmentName("g]g]g]g]g]g]g]g]|2024-07-01--12-00-00--1")

    assert isinstance(seg.route_name, RouteName)
    assert seg.route_name.dongle_id == "g]g]g]g]g]g]g]g]"

  def test_str_returns_canonical(self):
    """Test __str__ returns canonical name."""
    seg = SegmentName("h]h]h]h]h]h]h]h]|2024-08-10--15-30-00--4")

    assert str(seg) == seg.canonical_name

  def test_allow_route_name_without_segment(self):
    """Test allow_route_name flag for route-only names."""
    seg = SegmentName("i]i]i]i]i]i]i]i]|2024-09-20--18-00-00", allow_route_name=True)

    assert seg.segment_num == -1

  def test_with_data_dir_prefix(self):
    """Test parsing with data dir prefix."""
    seg = SegmentName("/data/media/0/realdata/j]j]j]j]j]j]j]j]|2024-10-05--10-00-00--6")

    assert seg.dongle_id == "j]j]j]j]j]j]j]j]"
    assert seg.segment_num == 6
    assert seg.data_dir == "/data/media/0/realdata"

  def test_without_data_dir(self):
    """Test data_dir is None when not present."""
    seg = SegmentName("k]k]k]k]k]k]k]k]|2024-11-15--14-30-00--8")

    assert seg.data_dir is None


class TestSegmentNameStaticMethods:
  """Test SegmentName static factory methods."""

  def test_from_file_name(self):
    """Test from_file_name static method."""
    seg = SegmentName.from_file_name("/some/path/l]l]l]l]l]l]l]l]|2024-01-01--00-00-00/5/rlog.bz2")

    assert seg.dongle_id == "l]l]l]l]l]l]l]l]"
    assert seg.segment_num == 5

  def test_from_device_key(self):
    """Test from_device_key static method."""
    seg = SegmentName.from_device_key("m]m]m]m]m]m]m]m]", "2024-02-15--12-00-00--3/rlog.bz2")

    assert seg.dongle_id == "m]m]m]m]m]m]m]m]"
    assert seg.segment_num == 3

  def test_from_file_key(self):
    """Test from_file_key static method."""
    seg = SegmentName.from_file_key("n]n]n]n]n]n]n]n]/2024-03-20--08-30-00/7/rlog.zst")

    assert seg.dongle_id == "n]n]n]n]n]n]n]n]"
    assert seg.segment_num == 7

  def test_from_azure_prefix(self):
    """Test from_azure_prefix static method."""
    seg = SegmentName.from_azure_prefix("o]o]o]o]o]o]o]o]/2024-04-10--16-45-00/2")

    assert seg.dongle_id == "o]o]o]o]o]o]o]o]"
    assert seg.time_str == "2024-04-10--16-45-00"
    assert seg.segment_num == 2


class TestSegmentRange:
  """Test SegmentRange class."""

  def test_parse_route_only(self):
    """Test parsing route name without slice."""
    sr = SegmentRange("aaaaaaaaaaaaaaaa/2024-01-15--12-30-45")

    assert sr.dongle_id == "aaaaaaaaaaaaaaaa"
    assert sr.log_id == "2024-01-15--12-30-45"

  def test_route_name_property(self):
    """Test route_name property."""
    sr = SegmentRange("bbbbbbbbbbbbbbbb/2024-02-20--08-15-30")

    assert sr.route_name == "bbbbbbbbbbbbbbbb/2024-02-20--08-15-30"

  def test_slice_property_empty(self):
    """Test slice property when no slice specified."""
    sr = SegmentRange("cccccccccccccccc/2024-03-10--14-45-00")

    assert sr.slice == ""

  def test_slice_property_with_segment(self):
    """Test slice property with segment number."""
    sr = SegmentRange("dddddddddddddddd/2024-04-05--10-20-30/5")

    assert sr.slice == "5"

  def test_slice_property_with_range(self):
    """Test slice property with range."""
    sr = SegmentRange("eeeeeeeeeeeeeeee/2024-05-15--16-30-00/0:10")

    assert sr.slice == "0:10"

  def test_selector_property_none(self):
    """Test selector property when not specified."""
    sr = SegmentRange("ffffffffffffffff/2024-06-20--09-00-00")

    assert sr.selector is None

  def test_selector_property_rlog(self):
    """Test selector property with rlog selector."""
    sr = SegmentRange("0000000000000000/2024-07-01--12-00-00/0/r")

    assert sr.selector == "r"

  def test_str_representation(self):
    """Test __str__ representation."""
    sr = SegmentRange("1111111111111111/2024-08-10--15-30-00")

    assert str(sr) == "1111111111111111/2024-08-10--15-30-00"

  def test_str_with_slice(self):
    """Test __str__ with slice."""
    sr = SegmentRange("2222222222222222/2024-09-20--18-00-00/5")

    assert str(sr) == "2222222222222222/2024-09-20--18-00-00/5"

  def test_repr_same_as_str(self):
    """Test __repr__ returns same as __str__."""
    sr = SegmentRange("3333333333333333/2024-10-05--10-00-00")

    assert repr(sr) == str(sr)

  def test_invalid_segment_range_raises(self):
    """Test invalid segment range raises assertion."""
    with pytest.raises(AssertionError):
      SegmentRange("invalid-format")


class TestSegmentRangeSegIdxs:
  """Test SegmentRange.seg_idxs with mocked API."""

  def test_single_segment(self, mocker):
    """Test seg_idxs with single segment number."""
    mock_max_seg = mocker.patch('openpilot.tools.lib.route.get_max_seg_number_cached')
    sr = SegmentRange("aaaaaaaaaaaaaaaa/2024-01-15--12-30-45/5")

    result = sr.seg_idxs

    assert result == [5]
    mock_max_seg.assert_not_called()

  def test_range_with_end(self, mocker):
    """Test seg_idxs with range that has end."""
    mock_max_seg = mocker.patch('openpilot.tools.lib.route.get_max_seg_number_cached')
    sr = SegmentRange("bbbbbbbbbbbbbbbb/2024-02-20--08-15-30/0:5")

    result = sr.seg_idxs

    # Note: slice(0, 5) on range(6) gives [0, 1, 2, 3, 4]
    assert result == [0, 1, 2, 3, 4]
    mock_max_seg.assert_not_called()

  def test_range_without_end(self, mocker):
    """Test seg_idxs with open-ended range."""
    mock_max_seg = mocker.patch('openpilot.tools.lib.route.get_max_seg_number_cached')
    mock_max_seg.return_value = 10
    sr = SegmentRange("cccccccccccccccc/2024-03-10--14-45-00/5:")

    result = sr.seg_idxs

    assert result == [5, 6, 7, 8, 9, 10]
    mock_max_seg.assert_called_once()

  def test_negative_index(self, mocker):
    """Test seg_idxs with negative index."""
    mock_max_seg = mocker.patch('openpilot.tools.lib.route.get_max_seg_number_cached')
    mock_max_seg.return_value = 10
    sr = SegmentRange("dddddddddddddddd/2024-04-05--10-20-30/-1")

    result = sr.seg_idxs

    assert result == [10]

  def test_step_in_range(self, mocker):
    """Test seg_idxs with step in range."""
    mocker.patch('openpilot.tools.lib.route.get_max_seg_number_cached')
    sr = SegmentRange("eeeeeeeeeeeeeeee/2024-05-15--16-30-00/0:10:2")

    result = sr.seg_idxs

    # Note: slice(0, 10, 2) on range(11) gives [0, 2, 4, 6, 8]
    assert result == [0, 2, 4, 6, 8]

  def test_negative_start_in_range(self, mocker):
    """Test seg_idxs with negative start in range."""
    mock_max_seg = mocker.patch('openpilot.tools.lib.route.get_max_seg_number_cached')
    mock_max_seg.return_value = 10
    sr = SegmentRange("ffffffffffffffff/2024-06-20--09-00-00/-3:")

    result = sr.seg_idxs

    assert result == [8, 9, 10]

  def test_negative_end_in_range(self, mocker):
    """Test seg_idxs with negative end in range."""
    mock_max_seg = mocker.patch('openpilot.tools.lib.route.get_max_seg_number_cached')
    mock_max_seg.return_value = 10
    sr = SegmentRange("0000000000000000/2024-07-01--12-00-00/0:-2")

    result = sr.seg_idxs

    assert result == [0, 1, 2, 3, 4, 5, 6, 7, 8]


class TestGetMaxSegNumberCached:
  """Test get_max_seg_number_cached function."""

  def test_success(self, mocker):
    """Test successful API call."""
    from openpilot.tools.lib.route import get_max_seg_number_cached

    # Clear the cache before testing
    get_max_seg_number_cached.cache_clear()

    mock_api = mocker.MagicMock()
    mock_api.get.return_value = {"maxqlog": 15}
    mocker.patch('openpilot.tools.lib.route.CommaApi', return_value=mock_api)
    mocker.patch('openpilot.tools.lib.route.get_token', return_value="fake_token")

    sr = SegmentRange("aaaaaaaaaaaaaaaa/2024-01-15--12-30-45")
    result = get_max_seg_number_cached(sr)

    assert result == 15

  def test_api_error_raises(self, mocker):
    """Test API error is wrapped in exception."""
    from openpilot.tools.lib.route import get_max_seg_number_cached

    # Clear the cache before testing
    get_max_seg_number_cached.cache_clear()

    mock_api = mocker.MagicMock()
    mock_api.get.side_effect = Exception("API Error")
    mocker.patch('openpilot.tools.lib.route.CommaApi', return_value=mock_api)
    mocker.patch('openpilot.tools.lib.route.get_token', return_value="fake_token")

    sr = SegmentRange("bbbbbbbbbbbbbbbb/2024-02-20--08-15-30")

    with pytest.raises(Exception, match="unable to get max_segment_number"):
      get_max_seg_number_cached(sr)


class TestSegment:
  """Test Segment class."""

  def test_init(self):
    """Test Segment initialization."""
    seg = Segment(
      "aaaaaaaaaaaaaaaa|2024-01-15--12-30-45--5",
      "/path/to/rlog.zst",
      "/path/to/qlog.zst",
      "/path/to/fcamera.hevc",
      "/path/to/dcamera.hevc",
      "/path/to/ecamera.hevc",
      "/path/to/qcamera.ts",
      "https://example.com/route",
    )

    assert seg.log_path == "/path/to/rlog.zst"
    assert seg.qlog_path == "/path/to/qlog.zst"
    assert seg.camera_path == "/path/to/fcamera.hevc"
    assert seg.dcamera_path == "/path/to/dcamera.hevc"
    assert seg.ecamera_path == "/path/to/ecamera.hevc"
    assert seg.qcamera_path == "/path/to/qcamera.ts"
    assert seg.url == "https://example.com/route/5"

  def test_name_property(self):
    """Test name property returns SegmentName."""
    seg = Segment("bbbbbbbbbbbbbbbb|2024-02-20--08-15-30--3", None, None, None, None, None, None, "https://example.com/route")

    assert isinstance(seg.name, SegmentName)
    assert seg.name.segment_num == 3

  def test_events_success(self, mocker):
    """Test events property fetches from API."""
    mock_response = mocker.MagicMock()
    mock_response.json.return_value = [{"type": "event1"}, {"type": "event2"}]
    mock_response.raise_for_status = mocker.MagicMock()
    mocker.patch('openpilot.tools.lib.route.requests.get', return_value=mock_response)

    seg = Segment("cccccccccccccccc|2024-03-10--14-45-00--0", None, None, None, None, None, None, "https://example.com/route")

    events = seg.events

    assert events == [{"type": "event1"}, {"type": "event2"}]

  def test_events_cached(self, mocker):
    """Test events property is cached."""
    mock_response = mocker.MagicMock()
    mock_response.json.return_value = [{"type": "cached"}]
    mock_response.raise_for_status = mocker.MagicMock()
    mock_get = mocker.patch('openpilot.tools.lib.route.requests.get', return_value=mock_response)

    seg = Segment("dddddddddddddddd|2024-04-05--10-20-30--1", None, None, None, None, None, None, "https://example.com/route")

    _ = seg.events
    _ = seg.events

    # Should only be called once due to caching
    assert mock_get.call_count == 1

  def test_events_error_raises_api_error(self, mocker):
    """Test events property raises APIError on failure."""
    from openpilot.tools.lib.api import APIError

    mocker.patch('openpilot.tools.lib.route.requests.get', side_effect=Exception("Network error"))

    seg = Segment("eeeeeeeeeeeeeeee|2024-05-15--16-30-00--2", None, None, None, None, None, None, "https://example.com/route")

    with pytest.raises(APIError, match="error getting events"):
      _ = seg.events


class TestRouteRemote:
  """Test Route class with remote segments."""

  def test_route_remote_segments(self, mocker):
    """Test Route loads segments from remote API."""
    mock_api = mocker.MagicMock()
    mock_api.get.side_effect = [
      # First call: route files
      {
        "logs": [
          "https://example.com/aaaaaaaaaaaaaaaa/2024-01-15--12-30-45/0/rlog.zst",
          "https://example.com/aaaaaaaaaaaaaaaa/2024-01-15--12-30-45/1/rlog.zst",
        ],
        "qlogs": [
          "https://example.com/aaaaaaaaaaaaaaaa/2024-01-15--12-30-45/0/qlog.zst",
          "https://example.com/aaaaaaaaaaaaaaaa/2024-01-15--12-30-45/1/qlog.zst",
        ],
      },
      # Second call: metadata
      {"url": "https://example.com/route"},
    ]
    mocker.patch('openpilot.tools.lib.route.CommaApi', return_value=mock_api)
    mocker.patch('openpilot.tools.lib.route.get_token', return_value="fake_token")

    route = Route("aaaaaaaaaaaaaaaa|2024-01-15--12-30-45")

    assert len(route.segments) == 2
    assert route.max_seg_number == 1

  def test_route_name_property(self, mocker):
    """Test Route name property."""
    mock_api = mocker.MagicMock()
    mock_api.get.side_effect = [
      {"logs": ["https://example.com/bbbbbbbbbbbbbbbb/2024-02-20--08-15-30/0/rlog.zst"]},
      {"url": "https://example.com/route"},
    ]
    mocker.patch('openpilot.tools.lib.route.CommaApi', return_value=mock_api)
    mocker.patch('openpilot.tools.lib.route.get_token', return_value="fake_token")

    route = Route("bbbbbbbbbbbbbbbb|2024-02-20--08-15-30")

    assert isinstance(route.name, RouteName)
    assert route.name.dongle_id == "bbbbbbbbbbbbbbbb"

  def test_route_log_paths(self, mocker):
    """Test Route log_paths method."""
    mock_api = mocker.MagicMock()
    mock_api.get.side_effect = [
      {
        "logs": [
          "https://example.com/cccccccccccccccc/2024-03-10--14-45-00/0/rlog.zst",
          "https://example.com/cccccccccccccccc/2024-03-10--14-45-00/2/rlog.zst",
        ],
      },
      {"url": "https://example.com/route"},
    ]
    mocker.patch('openpilot.tools.lib.route.CommaApi', return_value=mock_api)
    mocker.patch('openpilot.tools.lib.route.get_token', return_value="fake_token")

    route = Route("cccccccccccccccc|2024-03-10--14-45-00")

    log_paths = route.log_paths()

    # Should have 3 entries: 0, 1 (None), 2
    assert len(log_paths) == 3
    assert log_paths[0] is not None
    assert log_paths[1] is None
    assert log_paths[2] is not None

  def test_route_qlog_paths(self, mocker):
    """Test Route qlog_paths method."""
    mock_api = mocker.MagicMock()
    mock_api.get.side_effect = [
      {"qlogs": ["https://example.com/dddddddddddddddd/2024-04-05--10-20-30/0/qlog.zst"]},
      {"url": "https://example.com/route"},
    ]
    mocker.patch('openpilot.tools.lib.route.CommaApi', return_value=mock_api)
    mocker.patch('openpilot.tools.lib.route.get_token', return_value="fake_token")

    route = Route("dddddddddddddddd|2024-04-05--10-20-30")

    qlog_paths = route.qlog_paths()

    assert len(qlog_paths) == 1
    assert qlog_paths[0] is not None

  def test_route_camera_paths(self, mocker):
    """Test Route camera_paths method."""
    mock_api = mocker.MagicMock()
    mock_api.get.side_effect = [
      {"cameras": ["https://example.com/eeeeeeeeeeeeeeee/2024-05-15--16-30-00/0/fcamera.hevc"]},
      {"url": "https://example.com/route"},
    ]
    mocker.patch('openpilot.tools.lib.route.CommaApi', return_value=mock_api)
    mocker.patch('openpilot.tools.lib.route.get_token', return_value="fake_token")

    route = Route("eeeeeeeeeeeeeeee|2024-05-15--16-30-00")

    camera_paths = route.camera_paths()

    assert len(camera_paths) == 1

  def test_route_dcamera_paths(self, mocker):
    """Test Route dcamera_paths method."""
    mock_api = mocker.MagicMock()
    mock_api.get.side_effect = [
      {"dcameras": ["https://example.com/ffffffffffffffff/2024-06-20--09-00-00/0/dcamera.hevc"]},
      {"url": "https://example.com/route"},
    ]
    mocker.patch('openpilot.tools.lib.route.CommaApi', return_value=mock_api)
    mocker.patch('openpilot.tools.lib.route.get_token', return_value="fake_token")

    route = Route("ffffffffffffffff|2024-06-20--09-00-00")

    dcamera_paths = route.dcamera_paths()

    assert len(dcamera_paths) == 1

  def test_route_ecamera_paths(self, mocker):
    """Test Route ecamera_paths method."""
    mock_api = mocker.MagicMock()
    mock_api.get.side_effect = [
      {"ecameras": ["https://example.com/0000000000000000/2024-07-01--12-00-00/0/ecamera.hevc"]},
      {"url": "https://example.com/route"},
    ]
    mocker.patch('openpilot.tools.lib.route.CommaApi', return_value=mock_api)
    mocker.patch('openpilot.tools.lib.route.get_token', return_value="fake_token")

    route = Route("0000000000000000|2024-07-01--12-00-00")

    ecamera_paths = route.ecamera_paths()

    assert len(ecamera_paths) == 1

  def test_route_qcamera_paths(self, mocker):
    """Test Route qcamera_paths method."""
    mock_api = mocker.MagicMock()
    mock_api.get.side_effect = [
      {"qcameras": ["https://example.com/1111111111111111/2024-08-10--15-30-00/0/qcamera.ts"]},
      {"url": "https://example.com/route"},
    ]
    mocker.patch('openpilot.tools.lib.route.CommaApi', return_value=mock_api)
    mocker.patch('openpilot.tools.lib.route.get_token', return_value="fake_token")

    route = Route("1111111111111111|2024-08-10--15-30-00")

    qcamera_paths = route.qcamera_paths()

    assert len(qcamera_paths) == 1

  def test_route_metadata_cached(self, mocker):
    """Test Route metadata property is cached."""
    mock_api = mocker.MagicMock()
    mock_api.get.side_effect = [
      {"logs": ["https://example.com/2222222222222222/2024-09-20--18-00-00/0/rlog.zst"]},
      {"url": "https://example.com/route", "extra": "data"},
    ]
    mocker.patch('openpilot.tools.lib.route.CommaApi', return_value=mock_api)
    mocker.patch('openpilot.tools.lib.route.get_token', return_value="fake_token")

    route = Route("2222222222222222|2024-09-20--18-00-00")

    # Access metadata twice
    _ = route.metadata
    metadata = route.metadata

    assert metadata["extra"] == "data"


class TestRouteLocal:
  """Test Route class with local data directory."""

  def test_route_local_op_segment_dir(self, mocker):
    """Test Route loads segments from local openpilot segment directory."""
    mocker.patch('openpilot.tools.lib.route.get_token', return_value="fake_token")
    mock_api = mocker.MagicMock()
    mock_api.get.return_value = {"url": "https://example.com/route"}
    mocker.patch('openpilot.tools.lib.route.CommaApi', return_value=mock_api)

    with tempfile.TemporaryDirectory() as tmpdir:
      # Create segment directory structure like openpilot uses
      seg_dir = os.path.join(tmpdir, "aaaaaaaaaaaaaaaa|2024-01-15--12-30-45--0")
      os.makedirs(seg_dir)
      open(os.path.join(seg_dir, "rlog.zst"), 'w').close()
      open(os.path.join(seg_dir, "qlog.zst"), 'w').close()

      route = Route("aaaaaaaaaaaaaaaa|2024-01-15--12-30-45", data_dir=tmpdir)

      assert len(route.segments) == 1
      assert route.segments[0].log_path is not None

  def test_route_local_canonical_dir(self, mocker):
    """Test Route loads segments from canonical directory structure."""
    mocker.patch('openpilot.tools.lib.route.get_token', return_value="fake_token")
    mock_api = mocker.MagicMock()
    mock_api.get.return_value = {"url": "https://example.com/route"}
    mocker.patch('openpilot.tools.lib.route.CommaApi', return_value=mock_api)

    with tempfile.TemporaryDirectory() as tmpdir:
      # Create canonical directory structure: route_name/segment_num/files
      route_dir = os.path.join(tmpdir, "bbbbbbbbbbbbbbbb|2024-02-20--08-15-30")
      seg0_dir = os.path.join(route_dir, "0")
      seg1_dir = os.path.join(route_dir, "1")
      os.makedirs(seg0_dir)
      os.makedirs(seg1_dir)
      open(os.path.join(seg0_dir, "rlog.zst"), 'w').close()
      open(os.path.join(seg1_dir, "rlog.bz2"), 'w').close()

      route = Route("bbbbbbbbbbbbbbbb|2024-02-20--08-15-30", data_dir=tmpdir)

      assert len(route.segments) == 2

  def test_route_local_no_segments_raises(self, mocker):
    """Test Route raises ValueError when no segments found."""
    mocker.patch('openpilot.tools.lib.route.get_token', return_value="fake_token")
    mock_api = mocker.MagicMock()
    mock_api.get.return_value = {"url": "https://example.com/route"}
    mocker.patch('openpilot.tools.lib.route.CommaApi', return_value=mock_api)

    with tempfile.TemporaryDirectory() as tmpdir:
      with pytest.raises(ValueError, match="Could not find segments"):
        Route("cccccccccccccccc|2024-03-10--14-45-00", data_dir=tmpdir)

  def test_route_local_with_all_file_types(self, mocker):
    """Test Route loads all file types from local directory."""
    mocker.patch('openpilot.tools.lib.route.get_token', return_value="fake_token")
    mock_api = mocker.MagicMock()
    mock_api.get.return_value = {"url": "https://example.com/route"}
    mocker.patch('openpilot.tools.lib.route.CommaApi', return_value=mock_api)

    with tempfile.TemporaryDirectory() as tmpdir:
      seg_dir = os.path.join(tmpdir, "dddddddddddddddd|2024-04-05--10-20-30--0")
      os.makedirs(seg_dir)
      open(os.path.join(seg_dir, "rlog.zst"), 'w').close()
      open(os.path.join(seg_dir, "qlog.zst"), 'w').close()
      open(os.path.join(seg_dir, "fcamera.hevc"), 'w').close()
      open(os.path.join(seg_dir, "dcamera.hevc"), 'w').close()
      open(os.path.join(seg_dir, "ecamera.hevc"), 'w').close()
      open(os.path.join(seg_dir, "qcamera.ts"), 'w').close()

      route = Route("dddddddddddddddd|2024-04-05--10-20-30", data_dir=tmpdir)

      seg = route.segments[0]
      assert seg.log_path is not None
      assert seg.qlog_path is not None
      assert seg.camera_path is not None
      assert seg.dcamera_path is not None
      assert seg.ecamera_path is not None
      assert seg.qcamera_path is not None

  def test_route_local_skips_non_digit_dirs(self, mocker):
    """Test Route skips non-digit directories in canonical structure."""
    mocker.patch('openpilot.tools.lib.route.get_token', return_value="fake_token")
    mock_api = mocker.MagicMock()
    mock_api.get.return_value = {"url": "https://example.com/route"}
    mocker.patch('openpilot.tools.lib.route.CommaApi', return_value=mock_api)

    with tempfile.TemporaryDirectory() as tmpdir:
      route_dir = os.path.join(tmpdir, "eeeeeeeeeeeeeeee|2024-05-15--16-30-00")
      seg0_dir = os.path.join(route_dir, "0")
      invalid_dir = os.path.join(route_dir, "invalid")
      os.makedirs(seg0_dir)
      os.makedirs(invalid_dir)
      open(os.path.join(seg0_dir, "rlog.zst"), 'w').close()
      open(os.path.join(invalid_dir, "rlog.zst"), 'w').close()

      route = Route("eeeeeeeeeeeeeeee|2024-05-15--16-30-00", data_dir=tmpdir)

      # Should only have 1 segment (the "0" directory)
      assert len(route.segments) == 1

  def test_route_local_explorer_format(self, mocker):
    """Test Route loads segments from explorer file format."""
    mocker.patch('openpilot.tools.lib.route.get_token', return_value="fake_token")
    mock_api = mocker.MagicMock()
    mock_api.get.return_value = {"url": "https://example.com/route"}
    mocker.patch('openpilot.tools.lib.route.CommaApi', return_value=mock_api)

    with tempfile.TemporaryDirectory() as tmpdir:
      # Explorer format: dongle_id|log_id--segment_num--file_name
      explorer_file = "ffffffffffffffff|2024-06-20--09-00-00--0--rlog.zst"
      open(os.path.join(tmpdir, explorer_file), 'w').close()

      route = Route("ffffffffffffffff|2024-06-20--09-00-00", data_dir=tmpdir)

      assert len(route.segments) == 1
