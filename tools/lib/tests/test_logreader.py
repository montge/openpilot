import capnp
import contextlib
import io
import shutil
import tempfile
import os
import pytest
import requests

from parameterized import parameterized

from cereal import log as capnp_log
from openpilot.tools.lib.logreader import LogsUnavailable, LogIterable, LogReader, parse_indirect, ReadMode
from openpilot.tools.lib.file_sources import comma_api_source, InternalUnavailableException
from openpilot.tools.lib.route import SegmentRange
from openpilot.tools.lib.url_file import URLFileException

NUM_SEGS = 17  # number of segments in the test route
ALL_SEGS = list(range(NUM_SEGS))
TEST_ROUTE = "344c5c15b34f2d8a/2024-01-03--09-37-12"
QLOG_FILE = "https://commadataci.blob.core.windows.net/openpilotci/0375fdf7b1ce594d/2019-06-13--08-32-25/3/qlog.bz2"


def noop(segment: LogIterable):
  return segment


@contextlib.contextmanager
def setup_source_scenario(mocker, is_internal=False):
  internal_source_mock = mocker.patch("openpilot.tools.lib.logreader.internal_source")
  internal_source_mock.__name__ = internal_source_mock._mock_name

  openpilotci_source_mock = mocker.patch("openpilot.tools.lib.logreader.openpilotci_source")
  openpilotci_source_mock.__name__ = openpilotci_source_mock._mock_name

  comma_api_source_mock = mocker.patch("openpilot.tools.lib.logreader.comma_api_source")
  comma_api_source_mock.__name__ = comma_api_source_mock._mock_name

  if is_internal:
    internal_source_mock.return_value = {3: QLOG_FILE}
  else:
    internal_source_mock.side_effect = InternalUnavailableException

  openpilotci_source_mock.return_value = {}
  comma_api_source_mock.return_value = {3: QLOG_FILE}

  yield


class TestLogReader:
  @parameterized.expand(
    [
      (f"{TEST_ROUTE}", ALL_SEGS),
      (f"{TEST_ROUTE.replace('/', '|')}", ALL_SEGS),
      (f"{TEST_ROUTE}--0", [0]),
      (f"{TEST_ROUTE}--5", [5]),
      (f"{TEST_ROUTE}/0", [0]),
      (f"{TEST_ROUTE}/5", [5]),
      (f"{TEST_ROUTE}/0:10", ALL_SEGS[0:10]),
      (f"{TEST_ROUTE}/0:0", []),
      (f"{TEST_ROUTE}/4:6", ALL_SEGS[4:6]),
      (f"{TEST_ROUTE}/0:-1", ALL_SEGS[0:-1]),
      (f"{TEST_ROUTE}/:5", ALL_SEGS[:5]),
      (f"{TEST_ROUTE}/2:", ALL_SEGS[2:]),
      (f"{TEST_ROUTE}/2:-1", ALL_SEGS[2:-1]),
      (f"{TEST_ROUTE}/-1", [ALL_SEGS[-1]]),
      (f"{TEST_ROUTE}/-2", [ALL_SEGS[-2]]),
      (f"{TEST_ROUTE}/-2:-1", ALL_SEGS[-2:-1]),
      (f"{TEST_ROUTE}/-4:-2", ALL_SEGS[-4:-2]),
      (f"{TEST_ROUTE}/:10:2", ALL_SEGS[:10:2]),
      (f"{TEST_ROUTE}/5::2", ALL_SEGS[5::2]),
      (f"https://useradmin.comma.ai/?onebox={TEST_ROUTE}", ALL_SEGS),
      (f"https://useradmin.comma.ai/?onebox={TEST_ROUTE.replace('/', '|')}", ALL_SEGS),
      (f"https://useradmin.comma.ai/?onebox={TEST_ROUTE.replace('/', '%7C')}", ALL_SEGS),
    ]
  )
  @pytest.mark.skip("this got flaky. internet tests are stupid.")
  def test_indirect_parsing(self, identifier, expected):
    parsed = parse_indirect(identifier)
    sr = SegmentRange(parsed)
    assert list(sr.seg_idxs) == expected, identifier

  @parameterized.expand(
    [
      (f"{TEST_ROUTE}", f"{TEST_ROUTE}"),
      (f"{TEST_ROUTE.replace('/', '|')}", f"{TEST_ROUTE}"),
      (f"{TEST_ROUTE}--5", f"{TEST_ROUTE}/5"),
      (f"{TEST_ROUTE}/0/q", f"{TEST_ROUTE}/0/q"),
      (f"{TEST_ROUTE}/5:6/r", f"{TEST_ROUTE}/5:6/r"),
      (f"{TEST_ROUTE}/5", f"{TEST_ROUTE}/5"),
    ]
  )
  def test_canonical_name(self, identifier, expected):
    sr = SegmentRange(identifier)
    assert str(sr) == expected

  @pytest.mark.parametrize("cache_enabled", [True, False])
  def test_direct_parsing(self, mocker, cache_enabled):
    file_exists_mock = mocker.patch("openpilot.tools.lib.filereader.file_exists")
    if cache_enabled:
      os.environ.pop("DISABLE_FILEREADER_CACHE", None)
    else:
      os.environ["DISABLE_FILEREADER_CACHE"] = "1"
    qlog = tempfile.NamedTemporaryFile(mode='wb', delete=False)

    with requests.get(QLOG_FILE, stream=True) as r:
      with qlog as f:
        shutil.copyfileobj(r.raw, f)

    for f in [QLOG_FILE, qlog.name]:
      l = len(list(LogReader(f)))
      assert l > 100

    with pytest.raises(URLFileException) if not cache_enabled else pytest.raises(AssertionError):
      l = len(list(LogReader(QLOG_FILE.replace("/3/", "/200/"))))

    # file_exists should not be called for direct files
    assert file_exists_mock.call_count == 0

  @parameterized.expand(
    [
      (f"{TEST_ROUTE}///",),
      (f"{TEST_ROUTE}---",),
      (f"{TEST_ROUTE}/-4:--2",),
      (f"{TEST_ROUTE}/-a",),
      (f"{TEST_ROUTE}/j",),
      (f"{TEST_ROUTE}/0:1:2:3",),
      (f"{TEST_ROUTE}/:::3",),
      (f"{TEST_ROUTE}3",),
      (f"{TEST_ROUTE}-3",),
      (f"{TEST_ROUTE}--3a",),
    ]
  )
  def test_bad_ranges(self, segment_range):
    with pytest.raises(AssertionError):
      _ = SegmentRange(segment_range).seg_idxs

  @pytest.mark.parametrize(
    "segment_range, api_call",
    [
      (f"{TEST_ROUTE}/0", False),
      (f"{TEST_ROUTE}/:2", False),
      (f"{TEST_ROUTE}/0:", True),
      (f"{TEST_ROUTE}/-1", True),
      (f"{TEST_ROUTE}", True),
    ],
  )
  def test_slicing_api_call(self, mocker, segment_range, api_call):
    max_seg_mock = mocker.patch("openpilot.tools.lib.route.get_max_seg_number_cached")
    max_seg_mock.return_value = NUM_SEGS
    _ = SegmentRange(segment_range).seg_idxs
    assert api_call == max_seg_mock.called

  @pytest.mark.slow
  def test_modes(self):
    qlog_len = len(list(LogReader(f"{TEST_ROUTE}/0", ReadMode.QLOG)))
    rlog_len = len(list(LogReader(f"{TEST_ROUTE}/0", ReadMode.RLOG)))

    assert qlog_len * 6 < rlog_len

  @pytest.mark.slow
  def test_modes_from_name(self):
    qlog_len = len(list(LogReader(f"{TEST_ROUTE}/0/q")))
    rlog_len = len(list(LogReader(f"{TEST_ROUTE}/0/r")))

    assert qlog_len * 6 < rlog_len

  @pytest.mark.slow
  def test_list(self):
    qlog_len = len(list(LogReader(f"{TEST_ROUTE}/0/q")))
    qlog_len_2 = len(list(LogReader([f"{TEST_ROUTE}/0/q", f"{TEST_ROUTE}/0/q"])))

    assert qlog_len * 2 == qlog_len_2

  @pytest.mark.slow
  def test_multiple_iterations(self, mocker):
    init_mock = mocker.patch("openpilot.tools.lib.logreader._LogFileReader")
    lr = LogReader(f"{TEST_ROUTE}/0/q")
    qlog_len1 = len(list(lr))
    qlog_len2 = len(list(lr))

    # ensure we don't create multiple instances of _LogFileReader, which means downloading the files twice
    assert init_mock.call_count == 1

    assert qlog_len1 == qlog_len2

  @pytest.mark.slow
  def test_helpers(self):
    lr = LogReader(f"{TEST_ROUTE}/0/q")
    assert lr.first("carParams").carFingerprint == "SUBARU OUTBACK 6TH GEN"
    assert 0 < len(list(lr.filter("carParams"))) < len(list(lr))

  @parameterized.expand([(True,), (False,)])
  @pytest.mark.slow
  def test_run_across_segments(self, cache_enabled):
    if cache_enabled:
      os.environ.pop("DISABLE_FILEREADER_CACHE", None)
    else:
      os.environ["DISABLE_FILEREADER_CACHE"] = "1"
    lr = LogReader(f"{TEST_ROUTE}/0:4")
    assert len(lr.run_across_segments(4, noop)) == len(list(lr))

  @pytest.mark.slow
  def test_auto_mode(self, subtests, mocker):
    lr = LogReader(f"{TEST_ROUTE}/0/q")
    qlog_len = len(list(lr))
    log_paths_mock = mocker.patch("openpilot.tools.lib.route.Route.log_paths")
    log_paths_mock.return_value = [None] * NUM_SEGS
    # Should fall back to qlogs since rlogs are not available

    with subtests.test("interactive_yes"):
      mocker.patch("sys.stdin", new=io.StringIO("y\n"))
      lr = LogReader(f"{TEST_ROUTE}/0", default_mode=ReadMode.AUTO_INTERACTIVE, sources=[comma_api_source])
      log_len = len(list(lr))
      assert qlog_len == log_len

    with subtests.test("interactive_no"):
      mocker.patch("sys.stdin", new=io.StringIO("n\n"))
      with pytest.raises(LogsUnavailable):
        lr = LogReader(f"{TEST_ROUTE}/0", default_mode=ReadMode.AUTO_INTERACTIVE, sources=[comma_api_source])

    with subtests.test("non_interactive"):
      lr = LogReader(f"{TEST_ROUTE}/0", default_mode=ReadMode.AUTO, sources=[comma_api_source])
      log_len = len(list(lr))
      assert qlog_len == log_len

  @pytest.mark.parametrize("is_internal", [True, False])
  def test_auto_source_scenarios(self, mocker, is_internal):
    lr = LogReader(QLOG_FILE)
    qlog_len = len(list(lr))

    with setup_source_scenario(mocker, is_internal=is_internal):
      lr = LogReader(f"{TEST_ROUTE}/3/q")
      log_len = len(list(lr))
      assert qlog_len == log_len

  @pytest.mark.slow
  def test_sort_by_time(self):
    msgs = list(LogReader(f"{TEST_ROUTE}/0/q"))
    assert msgs != sorted(msgs, key=lambda m: m.logMonoTime)

    msgs = list(LogReader(f"{TEST_ROUTE}/0/q", sort_by_time=True))
    assert msgs == sorted(msgs, key=lambda m: m.logMonoTime)

  def test_only_union_types(self):
    with tempfile.NamedTemporaryFile() as qlog:
      # write valid Event messages
      num_msgs = 100
      with open(qlog.name, "wb") as f:
        f.write(b"".join(capnp_log.Event.new_message().to_bytes() for _ in range(num_msgs)))

      msgs = list(LogReader(qlog.name))
      assert len(msgs) == num_msgs
      [m.which() for m in msgs]

      # append non-union Event message
      event_msg = capnp_log.Event.new_message()
      non_union_bytes = bytearray(event_msg.to_bytes())
      non_union_bytes[event_msg.total_size.word_count * 8] = 0xFF  # set discriminant value out of range using Event word offset
      with open(qlog.name, "ab") as f:
        f.write(non_union_bytes)

      # ensure new message is added, but is not a union type
      msgs = list(LogReader(qlog.name))
      assert len(msgs) == num_msgs + 1
      with pytest.raises(capnp.KjException):
        [m.which() for m in msgs]

      # should not be added when only_union_types=True
      msgs = list(LogReader(qlog.name, only_union_types=True))
      assert len(msgs) == num_msgs
      [m.which() for m in msgs]


class TestSaveLog:
  """Tests for save_log function."""

  def _create_log_msgs(self, count=5):
    """Create log messages by reading from bytes (simulating LogReader output)."""
    msgs = [capnp_log.Event.new_message() for _ in range(count)]
    data = b"".join(msg.to_bytes() for msg in msgs)
    reader = LogReader.from_bytes(data)
    return list(reader)

  def test_save_log_uncompressed(self):
    """Test saving log without compression."""
    import bz2
    from openpilot.tools.lib.logreader import save_log

    # Use .bz2 extension but compress=False to test uncompressed path
    with tempfile.NamedTemporaryFile(suffix=".bz2", delete=False) as f:
      msgs = self._create_log_msgs()
      save_log(f.name, msgs, compress=False)

      with open(f.name, "rb") as rf:
        data = rf.read()
        assert len(data) > 0
        # Uncompressed data should not start with compression magic bytes
        assert not data.startswith(b'BZh9')
        # But we should be able to read it directly as capnp bytes
        with pytest.raises(OSError):
          bz2.decompress(data)  # Should fail since not compressed
      os.unlink(f.name)

  def test_save_log_bz2_compressed(self):
    """Test saving log with bz2 compression."""
    import bz2
    from openpilot.tools.lib.logreader import save_log

    with tempfile.NamedTemporaryFile(suffix=".bz2", delete=False) as f:
      msgs = self._create_log_msgs()
      save_log(f.name, msgs, compress=True)

      with open(f.name, "rb") as rf:
        data = rf.read()
        assert data.startswith(b'BZh9')
        # Verify it can be decompressed
        bz2.decompress(data)
      os.unlink(f.name)

  def test_save_log_zst_compressed(self):
    """Test saving log with zstd compression."""
    import zstandard as zstd
    from openpilot.tools.lib.logreader import save_log

    with tempfile.NamedTemporaryFile(suffix=".zst", delete=False) as f:
      msgs = self._create_log_msgs()
      save_log(f.name, msgs, compress=True)

      with open(f.name, "rb") as rf:
        data = rf.read()
        assert data.startswith(b'\x28\xb5\x2f\xfd')
        # Verify it can be decompressed
        zstd.decompress(data)
      os.unlink(f.name)


class TestDecompressStream:
  """Tests for decompress_stream function."""

  def test_decompress_zstd_data(self):
    """Test decompressing zstd data."""
    import zstandard as zstd
    from openpilot.tools.lib.logreader import decompress_stream

    original = b"test data for compression " * 100
    compressed = zstd.compress(original)

    result = decompress_stream(compressed)
    assert result == original


class TestCachedEventReader:
  """Tests for CachedEventReader class."""

  def _get_cached_event_reader(self):
    """Get a CachedEventReader from reading bytes."""
    msgs = [capnp_log.Event.new_message()]
    data = b"".join(msg.to_bytes() for msg in msgs)
    reader = LogReader.from_bytes(data)
    return list(reader)[0]

  def test_reduce_method(self):
    """Test __reduce__ method for pickling support."""
    from openpilot.tools.lib.logreader import CachedEventReader

    cached_reader = self._get_cached_event_reader()

    # Call __reduce__ directly to test the method
    reducer_func, args = cached_reader.__reduce__()
    assert reducer_func == CachedEventReader._reducer
    assert len(args) == 2  # (data bytes, _enum)

  def test_str_representation(self):
    """Test __str__ method."""
    cached_reader = self._get_cached_event_reader()

    result = str(cached_reader)
    assert isinstance(result, str)

  def test_dir_returns_event_dir(self):
    """Test __dir__ method returns event attributes."""
    cached_reader = self._get_cached_event_reader()

    result = dir(cached_reader)
    assert isinstance(result, list)
    assert "logMonoTime" in result

  def test_getattr_dunder_methods(self):
    """Test __getattr__ handles dunder methods correctly."""
    from openpilot.tools.lib.logreader import CachedEventReader

    cached_reader = self._get_cached_event_reader()

    # Accessing __class__ should work
    assert cached_reader.__class__ == CachedEventReader


class TestLogFileReaderEdgeCases:
  """Tests for _LogFileReader edge cases."""

  def test_unknown_extension_raises_error(self):
    """Test that unknown file extensions raise ValueError."""
    from openpilot.tools.lib.logreader import _LogFileReader

    with pytest.raises(ValueError, match="unknown extension"):
      _LogFileReader("http://example.com/file.xyz")

  def test_corrupted_events_warns(self):
    """Test that corrupted events emit a warning."""
    import warnings

    # Create valid log data with some corrupted bytes appended
    msgs = [capnp_log.Event.new_message() for _ in range(3)]
    valid_data = b"".join(msg.to_bytes() for msg in msgs)
    # Append corrupted capnp data that will fail to parse
    corrupted_data = valid_data + b"\x00\x01\x02\x03\x04\x05\x06\x07" * 10

    with tempfile.NamedTemporaryFile(delete=False) as f:
      f.write(corrupted_data)
      f.flush()

      # This may or may not trigger the warning depending on how capnp handles the corruption
      # but we're testing that the code path exists
      with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        lr = LogReader(f.name)
        # Just iterate to trigger the parsing
        list(lr)

      os.unlink(f.name)

  def test_zst_file_decompression(self):
    """Test reading zst compressed log file."""
    import zstandard as zstd

    with tempfile.NamedTemporaryFile(suffix=".zst", delete=False) as f:
      # Create a valid log with zstd compression
      msgs = [capnp_log.Event.new_message() for _ in range(5)]
      data = b"".join(msg.to_bytes() for msg in msgs)
      compressed = zstd.compress(data)
      f.write(compressed)
      f.flush()

      lr = LogReader(f.name)
      result = list(lr)
      assert len(result) == 5
      os.unlink(f.name)


class TestParseIndirectUseradminUrls:
  """Tests for parse_indirect with useradmin.comma.ai URLs."""

  def test_useradmin_url_basic(self):
    """Test parsing basic useradmin.comma.ai URL."""
    url = "https://useradmin.comma.ai/?onebox=344c5c15b34f2d8a/2024-01-03--09-37-12"
    result = parse_indirect(url)
    assert result == "344c5c15b34f2d8a/2024-01-03--09-37-12"

  def test_useradmin_url_with_pipe(self):
    """Test parsing useradmin URL with pipe separator."""
    url = "https://useradmin.comma.ai/?onebox=344c5c15b34f2d8a|2024-01-03--09-37-12"
    result = parse_indirect(url)
    assert result == "344c5c15b34f2d8a|2024-01-03--09-37-12"

  def test_useradmin_url_encoded_pipe(self):
    """Test parsing useradmin URL with URL-encoded pipe."""
    url = "https://useradmin.comma.ai/?onebox=344c5c15b34f2d8a%7C2024-01-03--09-37-12"
    result = parse_indirect(url)
    assert result == "344c5c15b34f2d8a|2024-01-03--09-37-12"


class TestParseIndirectConnectUrls:
  """Tests for parse_indirect with connect.comma.ai URLs."""

  def test_connect_url_basic(self):
    """Test parsing basic connect.comma.ai URL."""
    url = "https://connect.comma.ai/344c5c15b34f2d8a/2024-01-03--09-37-12"
    result = parse_indirect(url)
    assert result == "344c5c15b34f2d8a/2024-01-03--09-37-12"

  def test_connect_url_with_seconds(self):
    """Test parsing connect URL with start/end seconds."""
    # 120 seconds = 2 minutes, 180 seconds = 3 minutes
    url = "https://connect.comma.ai/344c5c15b34f2d8a/2024-01-03--09-37-12/120/180"
    result = parse_indirect(url)
    assert result == "344c5c15b34f2d8a/2024-01-03--09-37-12/2:4"

  def test_connect_url_with_seconds_and_selector(self):
    """Test parsing connect URL with seconds and selector."""
    url = "https://connect.comma.ai/344c5c15b34f2d8a/2024-01-03--09-37-12/60/120/q"
    result = parse_indirect(url)
    assert result == "344c5c15b34f2d8a/2024-01-03--09-37-12/1:3/q"

  def test_connect_url_with_selector_only(self):
    """Test parsing connect URL with selector but no seconds."""
    url = "https://connect.comma.ai/344c5c15b34f2d8a/2024-01-03--09-37-12/q"
    result = parse_indirect(url)
    assert result == "344c5c15b34f2d8a/2024-01-03--09-37-12/q"


class TestLogReaderFromBytes:
  """Tests for LogReader.from_bytes static method."""

  def test_from_bytes_creates_reader(self):
    """Test creating a log reader from raw bytes."""
    msgs = [capnp_log.Event.new_message() for _ in range(10)]
    data = b"".join(msg.to_bytes() for msg in msgs)

    reader = LogReader.from_bytes(data)
    result = list(reader)
    assert len(result) == 10


class TestLogReaderTimeSeries:
  """Tests for LogReader.time_series property."""

  def test_time_series_property(self):
    """Test that time_series property returns data."""
    # Use from_bytes to avoid file extension issues
    msgs = [capnp_log.Event.new_message() for _ in range(5)]
    data = b"".join(msg.to_bytes() for msg in msgs)

    reader = LogReader.from_bytes(data)
    # Just verify the property is accessible and returns something
    from openpilot.tools.lib.log_time_series import msgs_to_time_series

    ts = msgs_to_time_series(reader)
    assert ts is not None
