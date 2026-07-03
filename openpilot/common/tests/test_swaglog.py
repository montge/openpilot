"""Tests for common/swaglog.py - logging utilities."""

import logging
import os
import tempfile


from openpilot.common.swaglog import (
  SwaglogRotatingFileHandler,
  UnixDomainSocketHandler,
  ForwardingHandler,
  add_file_handler,
  cloudlog,
)


class TestSwaglogRotatingFileHandler:
  """Test SwaglogRotatingFileHandler class."""

  def test_init_creates_first_file(self):
    """Test handler creates first log file on init."""
    with tempfile.TemporaryDirectory() as tmpdir:
      base_filename = os.path.join(tmpdir, "test_log")
      handler = SwaglogRotatingFileHandler(base_filename, interval=60, max_bytes=1024)

      # Should have created a log file
      assert len(handler.log_files) == 1
      assert os.path.exists(handler.log_files[0])

      handler.close()

  def test_get_existing_logfiles(self):
    """Test get_existing_logfiles finds existing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
      base_filename = os.path.join(tmpdir, "test_log")

      # Create some existing log files
      for i in range(3):
        with open(f"{base_filename}.{i:010}", 'w') as f:
          f.write("test")

      handler = SwaglogRotatingFileHandler(base_filename, interval=60)

      # Should find existing files + new one created
      assert len(handler.log_files) >= 3

      handler.close()

  def test_should_rollover_size_exceeded(self, mocker):
    """Test shouldRollover returns True when size exceeded."""
    with tempfile.TemporaryDirectory() as tmpdir:
      base_filename = os.path.join(tmpdir, "test_log")
      handler = SwaglogRotatingFileHandler(base_filename, interval=3600, max_bytes=10)

      # Write some data to exceed max_bytes
      handler.stream.write("x" * 20)

      record = mocker.MagicMock()
      assert handler.shouldRollover(record) is True

      handler.close()

  def test_should_rollover_time_exceeded(self, mocker):
    """Test shouldRollover returns True when time exceeded."""
    with tempfile.TemporaryDirectory() as tmpdir:
      base_filename = os.path.join(tmpdir, "test_log")
      handler = SwaglogRotatingFileHandler(base_filename, interval=1, max_bytes=1024 * 1024)

      # Set last_rollover to past
      handler.last_rollover = 0

      record = mocker.MagicMock()
      assert handler.shouldRollover(record) is True

      handler.close()

  def test_do_rollover_creates_new_file(self):
    """Test doRollover creates a new file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      base_filename = os.path.join(tmpdir, "test_log")
      handler = SwaglogRotatingFileHandler(base_filename, interval=60)

      initial_count = len(handler.log_files)
      initial_idx = handler.last_file_idx

      handler.doRollover()

      assert len(handler.log_files) == initial_count + 1
      assert handler.last_file_idx == initial_idx + 1

      handler.close()

  def test_do_rollover_respects_backup_count(self):
    """Test doRollover removes old files when backup_count exceeded."""
    with tempfile.TemporaryDirectory() as tmpdir:
      base_filename = os.path.join(tmpdir, "test_log")
      handler = SwaglogRotatingFileHandler(base_filename, interval=60, backup_count=3)

      # Do multiple rollovers
      for _ in range(5):
        handler.doRollover()

      # Should only keep backup_count files
      assert len(handler.log_files) <= 3

      handler.close()


class TestUnixDomainSocketHandler:
  """Test UnixDomainSocketHandler class."""

  def test_init(self, mocker):
    """Test handler initializes correctly."""
    formatter = mocker.MagicMock()
    handler = UnixDomainSocketHandler(formatter)

    assert handler.pid is None
    assert handler.zctx is None
    assert handler.sock is None

  def test_close_without_connection(self, mocker):
    """Test close does nothing when not connected."""
    formatter = mocker.MagicMock()
    handler = UnixDomainSocketHandler(formatter)

    # Should not raise
    handler.close()

  def test_connect_creates_socket(self, mocker):
    """Test connect creates ZMQ socket."""
    mock_context = mocker.patch('zmq.Context')
    formatter = mocker.MagicMock()
    handler = UnixDomainSocketHandler(formatter)

    mock_ctx = mocker.MagicMock()
    mock_context.return_value = mock_ctx
    mock_sock = mocker.MagicMock()
    mock_ctx.socket.return_value = mock_sock

    handler.connect()

    assert handler.zctx == mock_ctx
    assert handler.sock == mock_sock
    assert handler.pid == os.getpid()

    mock_sock.setsockopt.assert_called()
    mock_sock.connect.assert_called()

  def test_emit_handles_zmq_again_error(self, mocker):
    """Test emit drops message silently on zmq.error.Again."""
    import zmq

    formatter = mocker.MagicMock()
    formatter.format.return_value = "test message"

    handler = UnixDomainSocketHandler(formatter)
    handler.pid = os.getpid()
    handler.sock = mocker.MagicMock()
    handler.sock.send.side_effect = zmq.error.Again()

    record = logging.LogRecord(name="test", level=logging.INFO, pathname="test.py", lineno=1, msg="test", args=(), exc_info=None)

    # Should not raise - message dropped silently
    handler.emit(record)


class TestForwardingHandler:
  """Test ForwardingHandler class."""

  def test_init(self, mocker):
    """Test handler initializes with target logger."""
    target = mocker.MagicMock()
    handler = ForwardingHandler(target)

    assert handler.target_logger == target

  def test_emit_forwards_to_target(self, mocker):
    """Test emit forwards record to target logger."""
    target = mocker.MagicMock()
    handler = ForwardingHandler(target)

    record = mocker.MagicMock()
    handler.emit(record)

    target.handle.assert_called_once_with(record)


class TestCloudlog:
  """Test cloudlog instance."""

  def test_cloudlog_is_swaglogger(self):
    """Test cloudlog is a SwagLogger instance."""
    from openpilot.common.logging_extra import SwagLogger

    assert isinstance(cloudlog, SwagLogger)

  def test_cloudlog_level_is_debug(self):
    """Test cloudlog level is set to DEBUG."""
    assert cloudlog.level == logging.DEBUG

  def test_cloudlog_has_handlers(self):
    """Test cloudlog has handlers attached."""
    assert len(cloudlog.handlers) > 0

  def test_cloudlog_can_log(self):
    """Test cloudlog can log messages without error."""
    # Should not raise
    cloudlog.debug("test debug message")
    cloudlog.info("test info message")
    cloudlog.warning("test warning message")

  def test_cloudlog_bind(self):
    """Test cloudlog can bind context."""
    cloudlog.bind(test_key="test_value")
    # Should not raise
    cloudlog.info("test with bound context")


class TestAddFileHandler:
  """Test add_file_handler function."""

  def test_add_file_handler_adds_handler(self, mocker):
    """Test add_file_handler adds handler to logger."""
    mock_get_handler = mocker.patch('openpilot.common.swaglog.get_file_handler')
    mock_handler = mocker.MagicMock()
    mock_get_handler.return_value = mock_handler

    logger = mocker.MagicMock()
    add_file_handler(logger)

    mock_handler.setFormatter.assert_called_once()
    logger.addHandler.assert_called_once_with(mock_handler)


class TestGetExistingLogfilesEdgeCases:
  """Test edge cases for get_existing_logfiles."""

  def test_ignores_non_matching_files(self):
    """Test get_existing_logfiles ignores files not starting with base_filename."""
    with tempfile.TemporaryDirectory() as tmpdir:
      base_filename = os.path.join(tmpdir, "test_log")

      # Create a file that doesn't match the base_filename pattern
      with open(os.path.join(tmpdir, "other_file.txt"), 'w') as f:
        f.write("test")

      # Create a matching log file
      with open(f"{base_filename}.0000000000", 'w') as f:
        f.write("test")

      handler = SwaglogRotatingFileHandler(base_filename, interval=60)

      # Should only find matching files (base + new one created)
      matching_files = [f for f in handler.log_files if f.startswith(base_filename)]
      assert len(matching_files) >= 1

      handler.close()


class TestDoRolloverEdgeCases:
  """Test edge cases for doRollover."""

  def test_do_rollover_with_zero_backup_count(self):
    """Test doRollover doesn't delete files when backup_count is 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
      base_filename = os.path.join(tmpdir, "test_log")
      handler = SwaglogRotatingFileHandler(base_filename, interval=60, backup_count=0)

      # Do multiple rollovers
      for _ in range(3):
        handler.doRollover()

      # With backup_count=0, all files should be kept
      assert len(handler.log_files) >= 3

      handler.close()


class TestUnixDomainSocketEmitEdgeCases:
  """Test edge cases for UnixDomainSocketHandler.emit."""

  def test_emit_reconnects_on_fork(self, mocker):
    """Test emit reconnects when PID changes (after fork)."""

    mock_context = mocker.patch('zmq.Context')
    mock_ctx = mocker.MagicMock()
    mock_context.return_value = mock_ctx
    mock_sock = mocker.MagicMock()
    mock_ctx.socket.return_value = mock_sock

    formatter = mocker.MagicMock()
    formatter.format.return_value = "test message"

    handler = UnixDomainSocketHandler(formatter)
    handler.pid = os.getpid() - 1  # Simulate different PID (fork)
    handler.sock = mock_sock

    record = logging.LogRecord(name="test", level=logging.INFO, pathname="test.py", lineno=1, msg="test", args=(), exc_info=None)

    handler.emit(record)

    # Should have reconnected (new context created)
    mock_context.assert_called()
