"""Tests for utils.py - common utilities."""
import io
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from openpilot.common.utils import (
  CallbackReader, atomic_write, strip_deprecated_keys,
  run_cmd, run_cmd_default, retry, LOG_COMPRESSION_LEVEL,
  managed_proc, get_upload_stream,
)


class TestCallbackReader(unittest.TestCase):
  """Test CallbackReader class."""

  def test_callback_reader_init(self):
    """Test CallbackReader initialization."""
    f = io.BytesIO(b"test data")
    callback = MagicMock()
    reader = CallbackReader(f, callback, "arg1", "arg2")

    self.assertEqual(reader.f, f)
    self.assertEqual(reader.callback, callback)
    self.assertEqual(reader.cb_args, ("arg1", "arg2"))
    self.assertEqual(reader.total_read, 0)

  def test_callback_reader_read_calls_callback(self):
    """Test read calls callback with bytes read."""
    f = io.BytesIO(b"test data")
    callback = MagicMock()
    reader = CallbackReader(f, callback)

    data = reader.read(4)

    self.assertEqual(data, b"test")
    self.assertEqual(reader.total_read, 4)
    callback.assert_called_once_with(4)

  def test_callback_reader_read_accumulates(self):
    """Test multiple reads accumulate total."""
    f = io.BytesIO(b"test data here")
    callback = MagicMock()
    reader = CallbackReader(f, callback)

    reader.read(4)
    reader.read(5)

    self.assertEqual(reader.total_read, 9)
    self.assertEqual(callback.call_count, 2)

  def test_callback_reader_getattr_delegates(self):
    """Test getattr delegates to wrapped file."""
    f = io.BytesIO(b"test")
    reader = CallbackReader(f, lambda x: None)

    # seek is delegated to the underlying file
    reader.seek(0)
    self.assertEqual(reader.tell(), 0)


class TestAtomicWrite(unittest.TestCase):
  """Test atomic_write context manager."""

  def test_atomic_write_creates_file(self):
    """Test atomic_write creates the file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "test.txt")

      with atomic_write(path) as f:
        f.write("test content")

      with open(path) as f:
        self.assertEqual(f.read(), "test content")

  def test_atomic_write_raises_on_existing_file(self):
    """Test atomic_write raises if file exists and overwrite=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "existing.txt")
      with open(path, 'w') as f:
        f.write("existing")

      with self.assertRaises(FileExistsError):
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
        self.assertEqual(f.read(), "new content")

  def test_atomic_write_binary_mode(self):
    """Test atomic_write in binary mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "binary.bin")

      with atomic_write(path, mode='wb') as f:
        f.write(b"\x00\x01\x02")

      with open(path, 'rb') as f:
        self.assertEqual(f.read(), b"\x00\x01\x02")


class TestStripDeprecatedKeys(unittest.TestCase):
  """Test strip_deprecated_keys function."""

  def test_strip_top_level_deprecated(self):
    """Test stripping deprecated keys at top level."""
    d = {'key1': 1, 'key2_DEPRECATED': 2, 'key3': 3}
    result = strip_deprecated_keys(d)
    self.assertEqual(result, {'key1': 1, 'key3': 3})

  def test_strip_nested_deprecated(self):
    """Test stripping deprecated keys in nested dicts."""
    d = {
      'key1': 1,
      'nested': {
        'a': 1,
        'b_DEPRECATED': 2,
      }
    }
    result = strip_deprecated_keys(d)
    self.assertEqual(result, {'key1': 1, 'nested': {'a': 1}})

  def test_strip_no_deprecated(self):
    """Test dict without deprecated keys unchanged."""
    d = {'key1': 1, 'key2': 2}
    result = strip_deprecated_keys(d)
    self.assertEqual(result, {'key1': 1, 'key2': 2})

  def test_strip_empty_dict(self):
    """Test empty dict unchanged."""
    d = {}
    result = strip_deprecated_keys(d)
    self.assertEqual(result, {})


class TestRunCmd(unittest.TestCase):
  """Test run_cmd function."""

  def test_run_cmd_echo(self):
    """Test run_cmd with echo."""
    result = run_cmd(['echo', 'hello'])
    self.assertEqual(result, 'hello')

  def test_run_cmd_with_cwd(self):
    """Test run_cmd with cwd."""
    with tempfile.TemporaryDirectory() as tmpdir:
      result = run_cmd(['pwd'], cwd=tmpdir)
      self.assertEqual(result, tmpdir)


class TestRunCmdDefault(unittest.TestCase):
  """Test run_cmd_default function."""

  def test_run_cmd_default_success(self):
    """Test run_cmd_default returns output on success."""
    result = run_cmd_default(['echo', 'hello'])
    self.assertEqual(result, 'hello')

  def test_run_cmd_default_failure(self):
    """Test run_cmd_default returns default on failure."""
    result = run_cmd_default(['false'], default='fallback')
    self.assertEqual(result, 'fallback')

  def test_run_cmd_default_empty_default(self):
    """Test run_cmd_default with empty default."""
    result = run_cmd_default(['false'])
    self.assertEqual(result, '')


class TestRetry(unittest.TestCase):
  """Test retry decorator."""

  def test_retry_success_first_try(self):
    """Test retry with success on first try."""
    call_count = [0]

    @retry(attempts=3)
    def success_func():
      call_count[0] += 1
      return "success"

    result = success_func()

    self.assertEqual(result, "success")
    self.assertEqual(call_count[0], 1)

  @patch('openpilot.common.utils.cloudlog')
  @patch('openpilot.common.utils.time.sleep')
  def test_retry_success_after_failures(self, mock_sleep, mock_cloudlog):
    """Test retry succeeds after initial failures."""
    call_count = [0]

    @retry(attempts=3, delay=0.1)
    def eventually_succeeds():
      call_count[0] += 1
      if call_count[0] < 3:
        raise ValueError("fail")
      return "success"

    result = eventually_succeeds()

    self.assertEqual(result, "success")
    self.assertEqual(call_count[0], 3)

  @patch('openpilot.common.utils.cloudlog')
  @patch('openpilot.common.utils.time.sleep')
  def test_retry_raises_after_all_attempts(self, mock_sleep, mock_cloudlog):
    """Test retry raises exception after all attempts fail."""
    @retry(attempts=2, delay=0.1)
    def always_fails():
      raise ValueError("always fails")

    with self.assertRaises(Exception) as ctx:
      always_fails()

    assert "failed after retry" in str(ctx.exception)

  @patch('openpilot.common.utils.cloudlog')
  @patch('openpilot.common.utils.time.sleep')
  def test_retry_ignore_failure(self, mock_sleep, mock_cloudlog):
    """Test retry with ignore_failure=True doesn't raise."""
    @retry(attempts=2, delay=0.1, ignore_failure=True)
    def always_fails():
      raise ValueError("always fails")

    result = always_fails()  # Should not raise
    self.assertIsNone(result)


class TestManagedProc(unittest.TestCase):
  """Test managed_proc context manager."""

  def test_managed_proc_starts_process(self):
    """Test managed_proc starts a process."""
    with managed_proc(['echo', 'hello'], env=os.environ.copy()) as proc:
      self.assertIsNotNone(proc)
      stdout, _ = proc.communicate()
      self.assertEqual(stdout.decode().strip(), 'hello')

  def test_managed_proc_terminates_on_exit(self):
    """Test managed_proc terminates process on exit."""
    import signal

    with managed_proc(['sleep', '100'], env=os.environ.copy()) as proc:
      pid = proc.pid  # Store pid before exiting context
      # Don't wait - just exit context to test termination

    # Process should be terminated after exiting context
    # Returncode is set after termination (negative for signals)
    self.assertIsNotNone(proc.returncode)
    # Consume any remaining output to avoid resource warnings
    proc.stdout.close()
    proc.stderr.close()

  def test_managed_proc_with_completed_process(self):
    """Test managed_proc handles already completed process."""
    with managed_proc(['true'], env=os.environ.copy()) as proc:
      # Wait for process to complete naturally
      proc.communicate()

    # Should not raise even though process already exited
    self.assertEqual(proc.returncode, 0)

  def test_managed_proc_returns_popen(self):
    """Test managed_proc yields a Popen object."""
    from subprocess import Popen

    with managed_proc(['echo', 'test'], env=os.environ.copy()) as proc:
      self.assertIsInstance(proc, Popen)
      proc.communicate()  # Consume output to avoid resource warnings


class TestGetUploadStream(unittest.TestCase):
  """Test get_upload_stream function."""

  def test_uncompressed_stream(self):
    """Test get_upload_stream without compression."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
      f.write(b"test content")
      path = f.name

    try:
      stream, size = get_upload_stream(path, should_compress=False)

      self.assertEqual(size, 12)  # len("test content")
      self.assertEqual(stream.read(), b"test content")
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
      self.assertLess(size, 1000)
      # Stream should be readable
      compressed_data = stream.read()
      self.assertGreater(len(compressed_data), 0)
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
      self.assertEqual(decompressed, original_content)
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

      self.assertEqual(size, os.path.getsize(path))
      self.assertEqual(size, 500)
    finally:
      os.unlink(path)


class TestConstants(unittest.TestCase):
  """Test module constants."""

  def test_log_compression_level(self):
    """Test LOG_COMPRESSION_LEVEL is reasonable."""
    self.assertGreater(LOG_COMPRESSION_LEVEL, 0)
    self.assertLess(LOG_COMPRESSION_LEVEL, 22)  # zstd max


if __name__ == '__main__':
  unittest.main()
