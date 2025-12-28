"""Tests for tools/lib/helpers.py - regex patterns for routes and segments."""
import re
import pytest

from openpilot.tools.lib.helpers import RE


class TestDongleIdPattern:
  """Test DONGLE_ID regex pattern."""

  def test_valid_dongle_id(self):
    """Test valid 16-character hex dongle ID."""
    pattern = re.compile(f'^{RE.DONGLE_ID}$')

    assert pattern.match('0123456789abcdef')
    assert pattern.match('deadbeef01234567')
    assert pattern.match('a' * 16)

  def test_invalid_dongle_id_too_short(self):
    """Test dongle ID too short."""
    pattern = re.compile(f'^{RE.DONGLE_ID}$')

    assert pattern.match('0123456789abcde') is None  # 15 chars

  def test_invalid_dongle_id_too_long(self):
    """Test dongle ID too long."""
    pattern = re.compile(f'^{RE.DONGLE_ID}$')

    assert pattern.match('0123456789abcdef0') is None  # 17 chars

  def test_invalid_dongle_id_uppercase(self):
    """Test dongle ID with uppercase is invalid."""
    pattern = re.compile(f'^{RE.DONGLE_ID}$')

    assert pattern.match('0123456789ABCDEF') is None

  def test_invalid_dongle_id_non_hex(self):
    """Test dongle ID with non-hex chars is invalid."""
    pattern = re.compile(f'^{RE.DONGLE_ID}$')

    assert pattern.match('0123456789abcdeg') is None  # 'g' is not hex


class TestTimestampPattern:
  """Test TIMESTAMP regex pattern."""

  def test_valid_timestamp(self):
    """Test valid timestamp format."""
    pattern = re.compile(f'^{RE.TIMESTAMP}$')

    assert pattern.match('2024-01-15--12-30-45')
    assert pattern.match('2023-12-31--23-59-59')
    assert pattern.match('2000-01-01--00-00-00')

  def test_timestamp_captures_group(self):
    """Test timestamp captures named group."""
    pattern = re.compile(f'^{RE.TIMESTAMP}$')
    match = pattern.match('2024-01-15--12-30-45')

    assert match.group('timestamp') == '2024-01-15--12-30-45'

  def test_invalid_timestamp_format(self):
    """Test invalid timestamp formats."""
    pattern = re.compile(f'^{RE.TIMESTAMP}$')

    assert pattern.match('2024-1-15--12-30-45') is None  # single digit month
    assert pattern.match('2024/01/15--12-30-45') is None  # wrong separator
    assert pattern.match('2024-01-15-12-30-45') is None  # single dash


class TestLogIdV2Pattern:
  """Test LOG_ID_V2 regex pattern."""

  def test_valid_log_id_v2(self):
    """Test valid v2 log ID format."""
    pattern = re.compile(f'^{RE.LOG_ID_V2}$')

    assert pattern.match('deadbeef--abcde12345')
    assert pattern.match('01234567--zzzzzzzzzz')

  def test_log_id_v2_captures_groups(self):
    """Test v2 log ID captures named groups."""
    pattern = re.compile(f'^{RE.LOG_ID_V2}$')
    match = pattern.match('deadbeef--abcde12345')

    assert match.group('count') == 'deadbeef'
    assert match.group('uid') == 'abcde12345'


class TestRouteNamePattern:
  """Test ROUTE_NAME regex pattern."""

  def test_valid_route_with_timestamp(self):
    """Test valid route name with timestamp."""
    pattern = re.compile(f'^{RE.ROUTE_NAME}$')

    assert pattern.match('0123456789abcdef|2024-01-15--12-30-45')
    assert pattern.match('0123456789abcdef_2024-01-15--12-30-45')
    assert pattern.match('0123456789abcdef/2024-01-15--12-30-45')

  def test_valid_route_with_v2_log_id(self):
    """Test valid route name with v2 log ID."""
    pattern = re.compile(f'^{RE.ROUTE_NAME}$')

    assert pattern.match('0123456789abcdef|deadbeef--abcde12345')

  def test_route_name_captures_groups(self):
    """Test route name captures named groups."""
    pattern = re.compile(f'^{RE.ROUTE_NAME}$')
    match = pattern.match('0123456789abcdef|2024-01-15--12-30-45')

    assert match.group('dongle_id') == '0123456789abcdef'
    assert match.group('log_id') == '2024-01-15--12-30-45'
    assert match.group('route_name') == '0123456789abcdef|2024-01-15--12-30-45'


class TestSegmentNamePattern:
  """Test SEGMENT_NAME regex pattern."""

  def test_valid_segment_name(self):
    """Test valid segment name."""
    pattern = re.compile(f'^{RE.SEGMENT_NAME}$')

    assert pattern.match('0123456789abcdef|2024-01-15--12-30-45--0')
    assert pattern.match('0123456789abcdef|2024-01-15--12-30-45--10')
    assert pattern.match('0123456789abcdef|2024-01-15--12-30-45/5')

  def test_segment_name_captures_segment_num(self):
    """Test segment name captures segment number."""
    pattern = re.compile(f'^{RE.SEGMENT_NAME}$')
    match = pattern.match('0123456789abcdef|2024-01-15--12-30-45--42')

    assert match.group('segment_num') == '42'


class TestSlicePattern:
  """Test SLICE regex pattern."""

  def test_full_slice(self):
    """Test full slice with start:end:step."""
    pattern = re.compile(f'^{RE.SLICE}$')
    match = pattern.match('0:10:2')

    assert match.group('start') == '0'
    assert match.group('end') == '10'
    assert match.group('step') == '2'

  def test_start_only(self):
    """Test slice with start only."""
    pattern = re.compile(f'^{RE.SLICE}$')
    match = pattern.match('5')

    assert match.group('start') == '5'

  def test_start_end(self):
    """Test slice with start:end."""
    pattern = re.compile(f'^{RE.SLICE}$')
    match = pattern.match('0:10')

    assert match.group('start') == '0'
    assert match.group('end') == '10'

  def test_negative_indices(self):
    """Test slice with negative indices."""
    pattern = re.compile(f'^{RE.SLICE}$')
    match = pattern.match('-5:-1')

    assert match.group('start') == '-5'
    assert match.group('end') == '-1'


class TestSegmentRangePattern:
  """Test SEGMENT_RANGE regex pattern."""

  def test_route_with_slice(self):
    """Test segment range with slice."""
    pattern = re.compile(f'^{RE.SEGMENT_RANGE}$')

    assert pattern.match('0123456789abcdef|2024-01-15--12-30-45--0:10')
    assert pattern.match('0123456789abcdef|2024-01-15--12-30-45/0:10:2')

  def test_route_with_selector(self):
    """Test segment range with selector."""
    pattern = re.compile(f'^{RE.SEGMENT_RANGE}$')

    match = pattern.match('0123456789abcdef|2024-01-15--12-30-45--0:10/q')
    assert match.group('selector') == 'q'

    match = pattern.match('0123456789abcdef|2024-01-15--12-30-45/5/a')
    assert match.group('selector') == 'a'

  def test_route_only(self):
    """Test segment range with route only."""
    pattern = re.compile(f'^{RE.SEGMENT_RANGE}$')

    match = pattern.match('0123456789abcdef|2024-01-15--12-30-45')
    assert match is not None
    assert match.group('route_name') == '0123456789abcdef|2024-01-15--12-30-45'


class TestExplorerFilePattern:
  """Test EXPLORER_FILE regex pattern."""

  def test_valid_explorer_file(self):
    """Test valid explorer file pattern."""
    pattern = re.compile(RE.EXPLORER_FILE)

    match = pattern.match('0123456789abcdef|2024-01-15--12-30-45--0--rlog.bz2')
    assert match is not None
    assert match.group('file_name') == 'rlog.bz2'

  def test_various_file_extensions(self):
    """Test various file extensions."""
    pattern = re.compile(RE.EXPLORER_FILE)

    assert pattern.match('0123456789abcdef|2024-01-15--12-30-45--0--qlog.bz2')
    assert pattern.match('0123456789abcdef|2024-01-15--12-30-45--0--fcamera.hevc')
    assert pattern.match('0123456789abcdef|2024-01-15--12-30-45--0--ecamera.hevc')


class TestOpSegmentDirPattern:
  """Test OP_SEGMENT_DIR regex pattern."""

  def test_valid_segment_dir(self):
    """Test valid segment directory pattern."""
    pattern = re.compile(RE.OP_SEGMENT_DIR)

    match = pattern.match('0123456789abcdef|2024-01-15--12-30-45--0')
    assert match is not None
    assert match.group('segment_name') == '0123456789abcdef|2024-01-15--12-30-45--0'

  def test_invalid_segment_dir_with_file(self):
    """Test segment directory with file is invalid."""
    pattern = re.compile(RE.OP_SEGMENT_DIR)

    assert pattern.match('0123456789abcdef|2024-01-15--12-30-45--0--rlog.bz2') is None
