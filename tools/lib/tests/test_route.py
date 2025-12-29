"""Tests for tools/lib/route.py - route and segment name parsing."""

import pytest

from openpilot.tools.lib.route import (
  FileName,
  RouteName,
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
