"""Tests for utils.py - common utilities."""

import io
import math
import os
import tempfile

import pytest

from openpilot.common.utils import (
  CallbackReader,
  MovingAverage,
  atomic_write,
  strip_deprecated_keys,
  run_cmd,
  run_cmd_default,
  retry,
  sudo_read,
  sudo_write,
  LOG_COMPRESSION_LEVEL,
  managed_proc,
  get_upload_stream,
)


class TestCallbackReader:
  """Test CallbackReader class."""

  def test_callback_reader_init(self, mocker):
    """Test CallbackReader initialization."""
    f = io.BytesIO(b"test data")
    callback = mocker.MagicMock()
    reader = CallbackReader(f, callback, "arg1", "arg2")

    assert reader.f == f
    assert reader.callback == callback
    assert reader.cb_args == ("arg1", "arg2")
    assert reader.total_read == 0

  def test_callback_reader_read_calls_callback(self, mocker):
    """Test read calls callback with bytes read."""
    f = io.BytesIO(b"test data")
    callback = mocker.MagicMock()
    reader = CallbackReader(f, callback)

    data = reader.read(4)

    assert data == b"test"
    assert reader.total_read == 4
    callback.assert_called_once_with(4)

  def test_callback_reader_read_accumulates(self, mocker):
    """Test multiple reads accumulate total."""
    f = io.BytesIO(b"test data here")
    callback = mocker.MagicMock()
    reader = CallbackReader(f, callback)

    reader.read(4)
    reader.read(5)

    assert reader.total_read == 9
    assert callback.call_count == 2

  def test_callback_reader_getattr_delegates(self):
    """Test getattr delegates to wrapped file."""
    f = io.BytesIO(b"test")
    reader = CallbackReader(f, lambda x: None)

    # seek is delegated to the underlying file
    reader.seek(0)
    assert reader.tell() == 0


class TestAtomicWrite:
  """Test atomic_write context manager."""

  def test_atomic_write_creates_file(self):
    """Test atomic_write creates the file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "test.txt")

      with atomic_write(path) as f:
        f.write("test content")

      with open(path) as f:
        assert f.read() == "test content"

  def test_atomic_write_raises_on_existing_file(self):
    """Test atomic_write raises if file exists and overwrite=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "existing.txt")
      with open(path, 'w') as f:
        f.write("existing")

      with pytest.raises(FileExistsError):
        with atomic_write(path, overwrite=False) as f:
          f.write("new")

  def test_atomic_write_overwrites_when_enabled(self):
    """Test atomic_write overwrites when overwrite=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "existing.txt")
      with open(path, 'w') as f:
        f.write("old content")

      with atomic_write(path, overwrite=True) as f:
        f.write("new content")

      with open(path) as f:
        assert f.read() == "new content"

  def test_atomic_write_binary_mode(self):
    """Test atomic_write in binary mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "binary.bin")

      with atomic_write(path, mode='wb') as f:
        f.write(b"\x00\x01\x02")

      with open(path, 'rb') as f:
        assert f.read() == b"\x00\x01\x02"


class TestStripDeprecatedKeys:
  """Test strip_deprecated_keys function."""

  def test_strip_top_level_deprecated(self):
    """Test stripping deprecated keys at top level."""
    d = {'key1': 1, 'key2_DEPRECATED': 2, 'key3': 3}
    result = strip_deprecated_keys(d)
    assert result == {'key1': 1, 'key3': 3}

  def test_strip_nested_deprecated(self):
    """Test stripping deprecated keys in nested dicts."""
    d = {
      'key1': 1,
      'nested': {
        'a': 1,
        'b_DEPRECATED': 2,
      },
    }
    result = strip_deprecated_keys(d)
    assert result == {'key1': 1, 'nested': {'a': 1}}

  def test_strip_no_deprecated(self):
    """Test dict without deprecated keys unchanged."""
    d = {'key1': 1, 'key2': 2}
    result = strip_deprecated_keys(d)
    assert result == {'key1': 1, 'key2': 2}

  def test_strip_empty_dict(self):
    """Test empty dict unchanged."""
    d = {}
    result = strip_deprecated_keys(d)
    assert result == {}

  def test_strip_non_string_keys_unchanged(self):
    """Test non-string keys are preserved (not checked for DEPRECATED)."""
    d = {1: 'one', 2: 'two', 'str_key': 'value'}
    result = strip_deprecated_keys(d)
    assert result == {1: 'one', 2: 'two', 'str_key': 'value'}


class TestRunCmd:
  """Test run_cmd function."""

  def test_run_cmd_echo(self):
    """Test run_cmd with echo."""
    result = run_cmd(['echo', 'hello'])
    assert result == 'hello'

  def test_run_cmd_with_cwd(self):
    """Test run_cmd with cwd."""
    with tempfile.TemporaryDirectory() as tmpdir:
      result = run_cmd(['pwd'], cwd=tmpdir)
      # realpath: on macOS tempdirs live under /var, a symlink to /private/var
      assert os.path.realpath(result) == os.path.realpath(tmpdir)


class TestRunCmdDefault:
  """Test run_cmd_default function."""

  def test_run_cmd_default_success(self):
    """Test run_cmd_default returns output on success."""
    result = run_cmd_default(['echo', 'hello'])
    assert result == 'hello'

  def test_run_cmd_default_failure(self):
    """Test run_cmd_default returns default on failure."""
    result = run_cmd_default(['false'], default='fallback')
    assert result == 'fallback'

  def test_run_cmd_default_empty_default(self):
    """Test run_cmd_default with empty default."""
    result = run_cmd_default(['false'])
    assert result == ''


class TestRetry:
  """Test retry decorator."""

  def test_retry_success_first_try(self):
    """Test retry with success on first try."""
    call_count = [0]

    @retry(attempts=3)
    def success_func():
      call_count[0] += 1
      return "success"

    result = success_func()

    assert result == "success"
    assert call_count[0] == 1

  def test_retry_success_after_failures(self, mocker):
    """Test retry succeeds after initial failures."""
    mocker.patch('openpilot.common.utils.time.sleep')
    call_count = [0]

    @retry(attempts=3, delay=0.1)
    def eventually_succeeds():
      call_count[0] += 1
      if call_count[0] < 3:
        raise ValueError("fail")
      return "success"

    result = eventually_succeeds()

    assert result == "success"
    assert call_count[0] == 3

  def test_retry_raises_after_all_attempts(self, mocker):
    """Test retry raises exception after all attempts fail."""
    mocker.patch('openpilot.common.utils.time.sleep')

    @retry(attempts=2, delay=0.1)
    def always_fails():
      raise ValueError("always fails")

    with pytest.raises(Exception) as ctx:
      always_fails()

    assert "failed after retry" in str(ctx.value)

  def test_retry_ignore_failure(self, mocker):
    """Test retry with ignore_failure=True doesn't raise."""
    mocker.patch('openpilot.common.utils.time.sleep')

    @retry(attempts=2, delay=0.1, ignore_failure=True)
    def always_fails():
      raise ValueError("always fails")

    result = always_fails()  # Should not raise
    assert result is None


class TestManagedProc:
  """Test managed_proc context manager."""

  def test_managed_proc_starts_process(self):
    """Test managed_proc starts a process."""
    with managed_proc(['echo', 'hello'], env=os.environ.copy()) as proc:
      assert proc is not None
      stdout, _ = proc.communicate()
      assert stdout.decode().strip() == 'hello'

  def test_managed_proc_terminates_on_exit(self):
    """Test managed_proc terminates process on exit."""
    with managed_proc(['sleep', '100'], env=os.environ.copy()) as proc:
      pass  # Don't wait - just exit context to test termination

    # Process should be terminated after exiting context
    # Returncode is set after termination (negative for signals)
    assert proc.returncode is not None
    # Consume any remaining output to avoid resource warnings
    proc.stdout.close()
    proc.stderr.close()

  def test_managed_proc_with_completed_process(self):
    """Test managed_proc handles already completed process."""
    with managed_proc(['true'], env=os.environ.copy()) as proc:
      # Wait for process to complete naturally
      proc.communicate()

    # Should not raise even though process already exited
    assert proc.returncode == 0

  def test_managed_proc_returns_popen(self):
    """Test managed_proc yields a Popen object."""
    from subprocess import Popen

    with managed_proc(['echo', 'test'], env=os.environ.copy()) as proc:
      assert isinstance(proc, Popen)
      proc.communicate()  # Consume output to avoid resource warnings


class TestGetUploadStream:
  """Test get_upload_stream function."""

  def test_uncompressed_stream(self):
    """Test get_upload_stream without compression."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
      f.write(b"test content")
      path = f.name

    try:
      stream, size = get_upload_stream(path, should_compress=False)

      assert size == 12  # len("test content")
      assert stream.read() == b"test content"
      stream.close()
    finally:
      os.unlink(path)

  def test_compressed_stream(self):
    """Test get_upload_stream with compression."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
      # Write some compressible content
      f.write(b"a" * 1000)
      path = f.name

    try:
      stream, size = get_upload_stream(path, should_compress=True)

      # Compressed size should be much smaller
      assert size < 1000
      # Stream should be readable
      compressed_data = stream.read()
      assert len(compressed_data) > 0
      stream.close()
    finally:
      os.unlink(path)

  def test_compressed_stream_can_decompress(self):
    """Test compressed stream can be decompressed."""
    import zstandard as zstd

    original_content = b"This is test content that should be compressible" * 10
    with tempfile.NamedTemporaryFile(delete=False) as f:
      f.write(original_content)
      path = f.name

    try:
      stream, size = get_upload_stream(path, should_compress=True)
      compressed_data = stream.read()
      stream.close()

      # Decompress using streaming decompressor (handles missing content size)
      decompressor = zstd.ZstdDecompressor()
      decompressed = decompressor.stream_reader(io.BytesIO(compressed_data)).read()
      assert decompressed == original_content
    finally:
      os.unlink(path)

  def test_uncompressed_size_matches_file(self):
    """Test uncompressed size matches actual file size."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
      content = b"x" * 500
      f.write(content)
      path = f.name

    try:
      stream, size = get_upload_stream(path, should_compress=False)
      stream.close()

      assert size == os.path.getsize(path)
      assert size == 500
    finally:
      os.unlink(path)


class TestManagedProcTimeoutKill:
  """Test managed_proc timeout and kill behavior."""

  def test_managed_proc_kills_on_timeout(self, mocker):
    """Test managed_proc kills process when wait times out."""
    from subprocess import TimeoutExpired

    # Create a mock process that doesn't respond to terminate
    mock_proc = mocker.MagicMock()
    mock_proc.poll.return_value = None  # Process still running
    mock_proc.wait.side_effect = TimeoutExpired('cmd', 5)

    # Patch Popen to return our mock
    mocker.patch('openpilot.common.utils.Popen', return_value=mock_proc)

    with managed_proc(['dummy'], env={}):
      pass

    # Should have called terminate, then wait, then kill
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once_with(timeout=5)
    mock_proc.kill.assert_called_once()


class TestConstants:
  """Test module constants."""

  def test_log_compression_level(self):
    """Test LOG_COMPRESSION_LEVEL is reasonable."""
    assert LOG_COMPRESSION_LEVEL > 0
    assert LOG_COMPRESSION_LEVEL < 22  # zstd max


class TestMovingAverage:
  """Test MovingAverage class."""

  def test_init_creates_buffer(self):
    """Test initialization creates proper buffer."""
    ma = MovingAverage(5)

    assert ma.window_size == 5
    assert len(ma.buffer) == 5
    assert ma.index == 0
    assert ma.count == 0
    assert ma.sum == 0.0

  def test_init_buffer_zeros(self):
    """Test buffer initialized with zeros."""
    ma = MovingAverage(3)

    for val in ma.buffer:
      assert val == 0.0

  def test_get_average_empty_returns_nan(self):
    """Test get_average returns NaN when empty."""
    ma = MovingAverage(5)

    avg = ma.get_average()

    assert math.isnan(avg)

  def test_add_single_value(self):
    """Test adding a single value."""
    ma = MovingAverage(5)

    ma.add_value(10.0)

    assert ma.count == 1
    assert ma.sum == 10.0
    assert ma.get_average() == 10.0

  def test_add_multiple_values_partial_window(self):
    """Test averaging with partial window."""
    ma = MovingAverage(5)

    ma.add_value(10.0)
    ma.add_value(20.0)
    ma.add_value(30.0)

    assert ma.count == 3
    assert ma.get_average() == 20.0  # (10+20+30)/3

  def test_circular_buffer_overwrites(self):
    """Test circular buffer overwrites old values."""
    ma = MovingAverage(3)

    ma.add_value(10.0)  # buffer: [10, 0, 0]
    ma.add_value(20.0)  # buffer: [10, 20, 0]
    ma.add_value(30.0)  # buffer: [10, 20, 30]
    ma.add_value(40.0)  # buffer: [40, 20, 30] - overwrites 10

    assert ma.count == 3  # Count stays at window size
    assert ma.get_average() == 30.0  # (40+20+30)/3

  def test_index_wraps_correctly(self):
    """Test index wraps around circular buffer."""
    ma = MovingAverage(3)

    for i in range(7):
      ma.add_value(float(i))

    # After 7 values in size-3 buffer: index should be at 7 % 3 = 1
    assert ma.index == 1

  def test_sum_updates_correctly_on_overwrite(self):
    """Test sum is correct when old values are replaced."""
    ma = MovingAverage(2)

    ma.add_value(100.0)
    ma.add_value(200.0)
    assert ma.sum == 300.0

    ma.add_value(50.0)  # Replaces 100.0
    assert ma.sum == 250.0  # 200 + 50

  def test_window_size_one(self):
    """Test MovingAverage with window size of 1."""
    ma = MovingAverage(1)

    ma.add_value(10.0)
    assert ma.get_average() == 10.0

    ma.add_value(20.0)
    assert ma.get_average() == 20.0

  def test_negative_values(self):
    """Test with negative values."""
    ma = MovingAverage(3)

    ma.add_value(-10.0)
    ma.add_value(-20.0)
    ma.add_value(-30.0)

    assert ma.get_average() == -20.0

  def test_float_precision(self):
    """Test float precision in calculations."""
    ma = MovingAverage(3)

    ma.add_value(0.1)
    ma.add_value(0.2)
    ma.add_value(0.3)

    assert ma.get_average() == pytest.approx(0.2, abs=1e-10)

  def test_count_never_exceeds_window_size(self):
    """Test count never exceeds window size."""
    ma = MovingAverage(5)

    for i in range(100):
      ma.add_value(float(i))

    assert ma.count == 5

  def test_step_change_response(self):
    """Test response to step change in input."""
    ma = MovingAverage(4)

    # Initial values of 0
    for _ in range(4):
      ma.add_value(0.0)

    assert ma.get_average() == 0.0

    # Step change to 100
    ma.add_value(100.0)
    assert ma.get_average() == 25.0  # (0+0+0+100)/4

    ma.add_value(100.0)
    assert ma.get_average() == 50.0  # (0+0+100+100)/4


class TestSudoWrite:
  """Test sudo_write function (current impl: plain write, then sudo chmod retry, then sudo echo fallback)."""

  def test_sudo_write_success(self, mocker):
    """Test sudo_write with writable path does not shell out."""
    mock_run = mocker.patch('openpilot.common.utils.subprocess.run')

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "test.txt")

      sudo_write("test", path)

      with open(path) as f:
        assert f.read() == "test"
      mock_run.assert_not_called()

  def test_sudo_write_chmod_on_permission_error(self, mocker):
    """Test sudo_write runs sudo chmod on PermissionError then retries the write."""
    mock_run = mocker.patch('openpilot.common.utils.subprocess.run')

    # Mock open to fail first, then succeed
    call_count = [0]
    real_open = open

    def mock_open_fn(path, mode='r', *args, **kwargs):
      if mode == 'w' and call_count[0] == 0:
        call_count[0] += 1
        raise PermissionError("Permission denied")
      return real_open(path, mode, *args, **kwargs)

    mocker.patch('builtins.open', mock_open_fn)

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "test.txt")

      sudo_write("content", path)

      with real_open(path) as f:
        assert f.read() == "content"
      mock_run.assert_called_once()
      assert "chmod" in mock_run.call_args[0][0]

  def test_sudo_write_fallback_on_double_permission_error(self, mocker):
    """Test sudo_write uses sudo echo fallback when chmod does not help."""
    mock_run = mocker.patch('openpilot.common.utils.subprocess.run')

    # Mock open to always fail with PermissionError on write
    real_open = open

    def mock_open_fn(path, mode='r', *args, **kwargs):
      if mode == 'w':
        raise PermissionError("Permission denied")
      return real_open(path, mode, *args, **kwargs)

    mocker.patch('builtins.open', mock_open_fn)

    sudo_write("testval", "/some/path")

    # Should have run chmod first, then the echo fallback
    assert mock_run.call_count == 2
    assert "chmod" in mock_run.call_args_list[0][0][0]
    assert "echo testval" in mock_run.call_args_list[1][0][0]


class TestSudoRead:
  """Test sudo_read function."""

  def test_sudo_read_success(self, mocker):
    """Test sudo_read returns stripped content."""
    mock_check_output = mocker.patch('openpilot.common.utils.subprocess.check_output')
    mock_check_output.return_value = "file content\n"

    result = sudo_read("/some/path")

    assert result == "file content"
    mock_check_output.assert_called_once_with(["sudo", "cat", "--", "/some/path"], encoding='utf8')

  def test_sudo_read_failure_returns_empty(self, mocker):
    """Test sudo_read returns empty string on failure."""
    mock_check_output = mocker.patch('openpilot.common.utils.subprocess.check_output')
    mock_check_output.side_effect = Exception("command failed")

    result = sudo_read("/nonexistent/path")

    assert result == ""
