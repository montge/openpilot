"""Tests for tools/lib/route.py - route and segment name parsing."""
import unittest
from unittest.mock import patch, MagicMock

from openpilot.tools.lib.route import (
  FileName, RouteName, SegmentName, SegmentRange,
)


class TestFileName(unittest.TestCase):
  """Test FileName constants."""

  def test_rlog_formats(self):
    """Test RLOG file formats."""
    self.assertIn("rlog.zst", FileName.RLOG)
    self.assertIn("rlog.bz2", FileName.RLOG)

  def test_qlog_formats(self):
    """Test QLOG file formats."""
    self.assertIn("qlog.zst", FileName.QLOG)
    self.assertIn("qlog.bz2", FileName.QLOG)

  def test_camera_formats(self):
    """Test camera file formats."""
    self.assertEqual(FileName.QCAMERA, ('qcamera.ts',))
    self.assertEqual(FileName.FCAMERA, ('fcamera.hevc',))
    self.assertEqual(FileName.ECAMERA, ('ecamera.hevc',))
    self.assertEqual(FileName.DCAMERA, ('dcamera.hevc',))

  def test_bootlog_formats(self):
    """Test BOOTLOG file formats."""
    self.assertIn("bootlog.zst", FileName.BOOTLOG)
    self.assertIn("bootlog.bz2", FileName.BOOTLOG)


class TestRouteName(unittest.TestCase):
  """Test RouteName class."""

  def test_parse_pipe_delimiter(self):
    """Test parsing route name with pipe delimiter."""
    route = RouteName("a]a]a]a]a]a]a]a]|2024-01-15--12-30-45")

    self.assertEqual(route.dongle_id, "a]a]a]a]a]a]a]a]")
    self.assertEqual(route.time_str, "2024-01-15--12-30-45")

  def test_parse_slash_delimiter(self):
    """Test parsing route name with slash delimiter."""
    route = RouteName("b]b]b]b]b]b]b]b]/2024-02-20--08-15-30")

    self.assertEqual(route.dongle_id, "b]b]b]b]b]b]b]b]")
    self.assertEqual(route.time_str, "2024-02-20--08-15-30")

  def test_canonical_name(self):
    """Test canonical_name property uses pipe delimiter."""
    route = RouteName("c]c]c]c]c]c]c]c]/2024-03-10--14-45-00")

    self.assertEqual(route.canonical_name, "c]c]c]c]c]c]c]c]|2024-03-10--14-45-00")

  def test_log_id_same_as_time_str(self):
    """Test log_id returns time_str."""
    route = RouteName("d]d]d]d]d]d]d]d]|2024-04-05--10-20-30")

    self.assertEqual(route.log_id, route.time_str)
    self.assertEqual(route.log_id, "2024-04-05--10-20-30")

  def test_azure_prefix(self):
    """Test azure_prefix property."""
    route = RouteName("e]e]e]e]e]e]e]e]|2024-05-15--16-30-00")

    self.assertEqual(route.azure_prefix, "e]e]e]e]e]e]e]e]/2024-05-15--16-30-00")

  def test_str_returns_canonical(self):
    """Test __str__ returns canonical name."""
    route = RouteName("f]f]f]f]f]f]f]f]|2024-06-20--09-00-00")

    self.assertEqual(str(route), route.canonical_name)

  def test_invalid_dongle_id_length(self):
    """Test assertion on invalid dongle_id length."""
    with self.assertRaises(AssertionError):
      RouteName("short|2024-01-01--00-00-00")

  def test_invalid_time_str_length(self):
    """Test assertion on invalid time_str length."""
    with self.assertRaises(AssertionError):
      RouteName("a]a]a]a]a]a]a]a]|2024-01-01")


class TestSegmentName(unittest.TestCase):
  """Test SegmentName class."""

  def test_parse_double_dash_segment(self):
    """Test parsing segment name with -- delimiter."""
    seg = SegmentName("a]a]a]a]a]a]a]a]|2024-01-15--12-30-45--5")

    self.assertEqual(seg.dongle_id, "a]a]a]a]a]a]a]a]")
    self.assertEqual(seg.time_str, "2024-01-15--12-30-45")
    self.assertEqual(seg.segment_num, 5)

  def test_parse_slash_segment(self):
    """Test parsing segment name with / delimiter."""
    seg = SegmentName("b]b]b]b]b]b]b]b]|2024-02-20--08-15-30/10")

    self.assertEqual(seg.segment_num, 10)
    self.assertEqual(seg.dongle_id, "b]b]b]b]b]b]b]b]")

  def test_canonical_name(self):
    """Test canonical_name uses -- delimiter."""
    seg = SegmentName("c]c]c]c]c]c]c]c]/2024-03-10--14-45-00/3")

    self.assertEqual(seg.canonical_name, "c]c]c]c]c]c]c]c]|2024-03-10--14-45-00--3")

  def test_data_name(self):
    """Test data_name uses / delimiter."""
    seg = SegmentName("d]d]d]d]d]d]d]d]|2024-04-05--10-20-30--7")

    self.assertEqual(seg.data_name, "d]d]d]d]d]d]d]d]|2024-04-05--10-20-30/7")

  def test_azure_prefix(self):
    """Test azure_prefix property."""
    seg = SegmentName("e]e]e]e]e]e]e]e]|2024-05-15--16-30-00--2")

    self.assertEqual(seg.azure_prefix, "e]e]e]e]e]e]e]e]/2024-05-15--16-30-00/2")

  def test_log_id_same_as_time_str(self):
    """Test log_id returns time_str."""
    seg = SegmentName("f]f]f]f]f]f]f]f]|2024-06-20--09-00-00--0")

    self.assertEqual(seg.log_id, seg.time_str)

  def test_route_name_property(self):
    """Test route_name property returns RouteName."""
    seg = SegmentName("g]g]g]g]g]g]g]g]|2024-07-01--12-00-00--1")

    self.assertIsInstance(seg.route_name, RouteName)
    self.assertEqual(seg.route_name.dongle_id, "g]g]g]g]g]g]g]g]")

  def test_str_returns_canonical(self):
    """Test __str__ returns canonical name."""
    seg = SegmentName("h]h]h]h]h]h]h]h]|2024-08-10--15-30-00--4")

    self.assertEqual(str(seg), seg.canonical_name)

  def test_allow_route_name_without_segment(self):
    """Test allow_route_name flag for route-only names."""
    seg = SegmentName("i]i]i]i]i]i]i]i]|2024-09-20--18-00-00", allow_route_name=True)

    self.assertEqual(seg.segment_num, -1)

  def test_with_data_dir_prefix(self):
    """Test parsing with data dir prefix."""
    seg = SegmentName("/data/media/0/realdata/j]j]j]j]j]j]j]j]|2024-10-05--10-00-00--6")

    self.assertEqual(seg.dongle_id, "j]j]j]j]j]j]j]j]")
    self.assertEqual(seg.segment_num, 6)
    self.assertEqual(seg.data_dir, "/data/media/0/realdata")

  def test_without_data_dir(self):
    """Test data_dir is None when not present."""
    seg = SegmentName("k]k]k]k]k]k]k]k]|2024-11-15--14-30-00--8")

    self.assertIsNone(seg.data_dir)


class TestSegmentNameStaticMethods(unittest.TestCase):
  """Test SegmentName static factory methods."""

  def test_from_file_name(self):
    """Test from_file_name static method."""
    seg = SegmentName.from_file_name(
      "/some/path/l]l]l]l]l]l]l]l]|2024-01-01--00-00-00/5/rlog.bz2"
    )

    self.assertEqual(seg.dongle_id, "l]l]l]l]l]l]l]l]")
    self.assertEqual(seg.segment_num, 5)

  def test_from_device_key(self):
    """Test from_device_key static method."""
    seg = SegmentName.from_device_key(
      "m]m]m]m]m]m]m]m]",
      "2024-02-15--12-00-00--3/rlog.bz2"
    )

    self.assertEqual(seg.dongle_id, "m]m]m]m]m]m]m]m]")
    self.assertEqual(seg.segment_num, 3)

  def test_from_file_key(self):
    """Test from_file_key static method."""
    seg = SegmentName.from_file_key(
      "n]n]n]n]n]n]n]n]/2024-03-20--08-30-00/7/rlog.zst"
    )

    self.assertEqual(seg.dongle_id, "n]n]n]n]n]n]n]n]")
    self.assertEqual(seg.segment_num, 7)

  def test_from_azure_prefix(self):
    """Test from_azure_prefix static method."""
    seg = SegmentName.from_azure_prefix("o]o]o]o]o]o]o]o]/2024-04-10--16-45-00/2")

    self.assertEqual(seg.dongle_id, "o]o]o]o]o]o]o]o]")
    self.assertEqual(seg.time_str, "2024-04-10--16-45-00")
    self.assertEqual(seg.segment_num, 2)


class TestSegmentRange(unittest.TestCase):
  """Test SegmentRange class."""

  def test_parse_route_only(self):
    """Test parsing route name without slice."""
    sr = SegmentRange("aaaaaaaaaaaaaaaa/2024-01-15--12-30-45")

    self.assertEqual(sr.dongle_id, "aaaaaaaaaaaaaaaa")
    self.assertEqual(sr.log_id, "2024-01-15--12-30-45")

  def test_route_name_property(self):
    """Test route_name property."""
    sr = SegmentRange("bbbbbbbbbbbbbbbb/2024-02-20--08-15-30")

    self.assertEqual(sr.route_name, "bbbbbbbbbbbbbbbb/2024-02-20--08-15-30")

  def test_slice_property_empty(self):
    """Test slice property when no slice specified."""
    sr = SegmentRange("cccccccccccccccc/2024-03-10--14-45-00")

    self.assertEqual(sr.slice, "")

  def test_slice_property_with_segment(self):
    """Test slice property with segment number."""
    sr = SegmentRange("dddddddddddddddd/2024-04-05--10-20-30/5")

    self.assertEqual(sr.slice, "5")

  def test_slice_property_with_range(self):
    """Test slice property with range."""
    sr = SegmentRange("eeeeeeeeeeeeeeee/2024-05-15--16-30-00/0:10")

    self.assertEqual(sr.slice, "0:10")

  def test_selector_property_none(self):
    """Test selector property when not specified."""
    sr = SegmentRange("ffffffffffffffff/2024-06-20--09-00-00")

    self.assertIsNone(sr.selector)

  def test_selector_property_rlog(self):
    """Test selector property with rlog selector."""
    sr = SegmentRange("0000000000000000/2024-07-01--12-00-00/0/r")

    self.assertEqual(sr.selector, "r")

  def test_str_representation(self):
    """Test __str__ representation."""
    sr = SegmentRange("1111111111111111/2024-08-10--15-30-00")

    self.assertEqual(str(sr), "1111111111111111/2024-08-10--15-30-00")

  def test_str_with_slice(self):
    """Test __str__ with slice."""
    sr = SegmentRange("2222222222222222/2024-09-20--18-00-00/5")

    self.assertEqual(str(sr), "2222222222222222/2024-09-20--18-00-00/5")

  def test_repr_same_as_str(self):
    """Test __repr__ returns same as __str__."""
    sr = SegmentRange("3333333333333333/2024-10-05--10-00-00")

    self.assertEqual(repr(sr), str(sr))

  def test_invalid_segment_range_raises(self):
    """Test invalid segment range raises assertion."""
    with self.assertRaises(AssertionError):
      SegmentRange("invalid-format")


class TestSegmentRangeSegIdxs(unittest.TestCase):
  """Test SegmentRange.seg_idxs with mocked API."""

  @patch('openpilot.tools.lib.route.get_max_seg_number_cached')
  def test_single_segment(self, mock_max_seg):
    """Test seg_idxs with single segment number."""
    sr = SegmentRange("aaaaaaaaaaaaaaaa/2024-01-15--12-30-45/5")

    result = sr.seg_idxs

    self.assertEqual(result, [5])
    mock_max_seg.assert_not_called()

  @patch('openpilot.tools.lib.route.get_max_seg_number_cached')
  def test_range_with_end(self, mock_max_seg):
    """Test seg_idxs with range that has end."""
    sr = SegmentRange("bbbbbbbbbbbbbbbb/2024-02-20--08-15-30/0:5")

    result = sr.seg_idxs

    # Note: slice(0, 5) on range(6) gives [0, 1, 2, 3, 4]
    self.assertEqual(result, [0, 1, 2, 3, 4])
    mock_max_seg.assert_not_called()

  @patch('openpilot.tools.lib.route.get_max_seg_number_cached')
  def test_range_without_end(self, mock_max_seg):
    """Test seg_idxs with open-ended range."""
    mock_max_seg.return_value = 10
    sr = SegmentRange("cccccccccccccccc/2024-03-10--14-45-00/5:")

    result = sr.seg_idxs

    self.assertEqual(result, [5, 6, 7, 8, 9, 10])
    mock_max_seg.assert_called_once()

  @patch('openpilot.tools.lib.route.get_max_seg_number_cached')
  def test_negative_index(self, mock_max_seg):
    """Test seg_idxs with negative index."""
    mock_max_seg.return_value = 10
    sr = SegmentRange("dddddddddddddddd/2024-04-05--10-20-30/-1")

    result = sr.seg_idxs

    self.assertEqual(result, [10])

  @patch('openpilot.tools.lib.route.get_max_seg_number_cached')
  def test_step_in_range(self, mock_max_seg):
    """Test seg_idxs with step in range."""
    sr = SegmentRange("eeeeeeeeeeeeeeee/2024-05-15--16-30-00/0:10:2")

    result = sr.seg_idxs

    # Note: slice(0, 10, 2) on range(11) gives [0, 2, 4, 6, 8]
    self.assertEqual(result, [0, 2, 4, 6, 8])


if __name__ == '__main__':
  unittest.main()
