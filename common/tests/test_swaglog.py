"""Tests for common/swaglog.py - logging utilities."""
import logging
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock, PropertyMock

from openpilot.common.swaglog import (
  SwaglogRotatingFileHandler, UnixDomainSocketHandler, ForwardingHandler,
  add_file_handler, cloudlog,
)
from openpilot.common.logging_extra import SwagFormatter


class TestSwaglogRotatingFileHandler(unittest.TestCase):
  """Test SwaglogRotatingFileHandler class."""

  def test_init_creates_first_file(self):
    """Test handler creates first log file on init."""
    with tempfile.TemporaryDirectory() as tmpdir:
      base_filename = os.path.join(tmpdir, "test_log")
      handler = SwaglogRotatingFileHandler(base_filename, interval=60, max_bytes=1024)

      # Should have created a log file
      self.assertEqual(len(handler.log_files), 1)
      self.assertTrue(os.path.exists(handler.log_files[0]))

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
      self.assertGreaterEqual(len(handler.log_files), 3)

      handler.close()

  def test_should_rollover_size_exceeded(self):
    """Test shouldRollover returns True when size exceeded."""
    with tempfile.TemporaryDirectory() as tmpdir:
      base_filename = os.path.join(tmpdir, "test_log")
      handler = SwaglogRotatingFileHandler(base_filename, interval=3600, max_bytes=10)

      # Write some data to exceed max_bytes
      handler.stream.write("x" * 20)

      record = MagicMock()
      self.assertTrue(handler.shouldRollover(record))

      handler.close()

  def test_should_rollover_time_exceeded(self):
    """Test shouldRollover returns True when time exceeded."""
    with tempfile.TemporaryDirectory() as tmpdir:
      base_filename = os.path.join(tmpdir, "test_log")
      handler = SwaglogRotatingFileHandler(base_filename, interval=1, max_bytes=1024*1024)

      # Set last_rollover to past
      handler.last_rollover = 0

      record = MagicMock()
      self.assertTrue(handler.shouldRollover(record))

      handler.close()

  def test_do_rollover_creates_new_file(self):
    """Test doRollover creates a new file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      base_filename = os.path.join(tmpdir, "test_log")
      handler = SwaglogRotatingFileHandler(base_filename, interval=60)

      initial_count = len(handler.log_files)
      initial_idx = handler.last_file_idx

      handler.doRollover()

      self.assertEqual(len(handler.log_files), initial_count + 1)
      self.assertEqual(handler.last_file_idx, initial_idx + 1)

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
      self.assertLessEqual(len(handler.log_files), 3)

      handler.close()


class TestUnixDomainSocketHandler(unittest.TestCase):
  """Test UnixDomainSocketHandler class."""

  def test_init(self):
    """Test handler initializes correctly."""
    formatter = MagicMock()
    handler = UnixDomainSocketHandler(formatter)

    self.assertIsNone(handler.pid)
    self.assertIsNone(handler.zctx)
    self.assertIsNone(handler.sock)

  def test_close_without_connection(self):
    """Test close does nothing when not connected."""
    formatter = MagicMock()
    handler = UnixDomainSocketHandler(formatter)

    # Should not raise
    handler.close()

  @patch('zmq.Context')
  def test_connect_creates_socket(self, mock_context):
    """Test connect creates ZMQ socket."""
    formatter = MagicMock()
    handler = UnixDomainSocketHandler(formatter)

    mock_ctx = MagicMock()
    mock_context.return_value = mock_ctx
    mock_sock = MagicMock()
    mock_ctx.socket.return_value = mock_sock

    handler.connect()

    self.assertEqual(handler.zctx, mock_ctx)
    self.assertEqual(handler.sock, mock_sock)
    self.assertEqual(handler.pid, os.getpid())

    mock_sock.setsockopt.assert_called()
    mock_sock.connect.assert_called()


class TestForwardingHandler(unittest.TestCase):
  """Test ForwardingHandler class."""

  def test_init(self):
    """Test handler initializes with target logger."""
    target = MagicMock()
    handler = ForwardingHandler(target)

    self.assertEqual(handler.target_logger, target)

  def test_emit_forwards_to_target(self):
    """Test emit forwards record to target logger."""
    target = MagicMock()
    handler = ForwardingHandler(target)

    record = MagicMock()
    handler.emit(record)

    target.handle.assert_called_once_with(record)


class TestCloudlog(unittest.TestCase):
  """Test cloudlog instance."""

  def test_cloudlog_is_swaglogger(self):
    """Test cloudlog is a SwagLogger instance."""
    from openpilot.common.logging_extra import SwagLogger
    self.assertIsInstance(cloudlog, SwagLogger)

  def test_cloudlog_level_is_debug(self):
    """Test cloudlog level is set to DEBUG."""
    self.assertEqual(cloudlog.level, logging.DEBUG)

  def test_cloudlog_has_handlers(self):
    """Test cloudlog has handlers attached."""
    self.assertGreater(len(cloudlog.handlers), 0)

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


class TestAddFileHandler(unittest.TestCase):
  """Test add_file_handler function."""

  @patch('openpilot.common.swaglog.get_file_handler')
  def test_add_file_handler_adds_handler(self, mock_get_handler):
    """Test add_file_handler adds handler to logger."""
    mock_handler = MagicMock()
    mock_get_handler.return_value = mock_handler

    logger = MagicMock()
    add_file_handler(logger)

    mock_handler.setFormatter.assert_called_once()
    logger.addHandler.assert_called_once_with(mock_handler)


if __name__ == '__main__':
  unittest.main()
