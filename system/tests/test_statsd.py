"""Tests for system/statsd.py - statistics logging."""
import unittest
from unittest.mock import patch, MagicMock
import os

from openpilot.system.statsd import METRIC_TYPE, StatLog


class TestMetricType(unittest.TestCase):
  """Test METRIC_TYPE class."""

  def test_gauge_value(self):
    """Test GAUGE is 'g'."""
    self.assertEqual(METRIC_TYPE.GAUGE, 'g')

  def test_sample_value(self):
    """Test SAMPLE is 'sa'."""
    self.assertEqual(METRIC_TYPE.SAMPLE, 'sa')


class TestStatLogInit(unittest.TestCase):
  """Test StatLog initialization."""

  def test_init_defaults(self):
    """Test StatLog initializes with None values."""
    sl = StatLog()

    self.assertIsNone(sl.pid)
    self.assertIsNone(sl.zctx)
    self.assertIsNone(sl.sock)


class TestStatLogConnect(unittest.TestCase):
  """Test StatLog.connect method."""

  @patch('openpilot.system.statsd.zmq.Context')
  def test_connect_creates_context(self, mock_ctx_class):
    """Test connect creates zmq Context."""
    mock_ctx = MagicMock()
    mock_ctx_class.return_value = mock_ctx
    mock_sock = MagicMock()
    mock_ctx.socket.return_value = mock_sock

    sl = StatLog()
    sl.connect()

    mock_ctx_class.assert_called_once()
    self.assertEqual(sl.zctx, mock_ctx)

  @patch('openpilot.system.statsd.zmq.Context')
  def test_connect_creates_push_socket(self, mock_ctx_class):
    """Test connect creates PUSH socket."""
    import zmq
    mock_ctx = MagicMock()
    mock_ctx_class.return_value = mock_ctx
    mock_sock = MagicMock()
    mock_ctx.socket.return_value = mock_sock

    sl = StatLog()
    sl.connect()

    mock_ctx.socket.assert_called_once_with(zmq.PUSH)

  @patch('openpilot.system.statsd.zmq.Context')
  def test_connect_sets_linger(self, mock_ctx_class):
    """Test connect sets LINGER socket option."""
    import zmq
    mock_ctx = MagicMock()
    mock_ctx_class.return_value = mock_ctx
    mock_sock = MagicMock()
    mock_ctx.socket.return_value = mock_sock

    sl = StatLog()
    sl.connect()

    mock_sock.setsockopt.assert_called_once_with(zmq.LINGER, 10)

  @patch('openpilot.system.statsd.zmq.Context')
  @patch('openpilot.system.statsd.STATS_SOCKET', 'ipc:///tmp/test_stats')
  def test_connect_connects_to_socket(self, mock_ctx_class):
    """Test connect connects to STATS_SOCKET."""
    mock_ctx = MagicMock()
    mock_ctx_class.return_value = mock_ctx
    mock_sock = MagicMock()
    mock_ctx.socket.return_value = mock_sock

    sl = StatLog()
    sl.connect()

    mock_sock.connect.assert_called_once_with('ipc:///tmp/test_stats')

  @patch('openpilot.system.statsd.zmq.Context')
  def test_connect_stores_pid(self, mock_ctx_class):
    """Test connect stores current PID."""
    mock_ctx = MagicMock()
    mock_ctx_class.return_value = mock_ctx
    mock_sock = MagicMock()
    mock_ctx.socket.return_value = mock_sock

    sl = StatLog()
    sl.connect()

    self.assertEqual(sl.pid, os.getpid())


class TestStatLogGauge(unittest.TestCase):
  """Test StatLog.gauge method."""

  @patch('openpilot.system.statsd.zmq.Context')
  def test_gauge_sends_correct_format(self, mock_ctx_class):
    """Test gauge sends metric in correct format."""
    import zmq
    mock_ctx = MagicMock()
    mock_ctx_class.return_value = mock_ctx
    mock_sock = MagicMock()
    mock_ctx.socket.return_value = mock_sock

    sl = StatLog()
    sl.connect()
    sl.gauge('test_metric', 42.5)

    mock_sock.send_string.assert_called_once_with(
      'test_metric:42.5|g', zmq.NOBLOCK
    )


class TestStatLogSample(unittest.TestCase):
  """Test StatLog.sample method."""

  @patch('openpilot.system.statsd.zmq.Context')
  def test_sample_sends_correct_format(self, mock_ctx_class):
    """Test sample sends metric in correct format."""
    import zmq
    mock_ctx = MagicMock()
    mock_ctx_class.return_value = mock_ctx
    mock_sock = MagicMock()
    mock_ctx.socket.return_value = mock_sock

    sl = StatLog()
    sl.connect()
    sl.sample('timing', 0.123)

    mock_sock.send_string.assert_called_once_with(
      'timing:0.123|sa', zmq.NOBLOCK
    )


class TestStatLogReconnect(unittest.TestCase):
  """Test StatLog reconnection on fork."""

  @patch('openpilot.system.statsd.zmq.Context')
  @patch('openpilot.system.statsd.os.getpid')
  def test_reconnects_on_fork(self, mock_getpid, mock_ctx_class):
    """Test _send reconnects when PID changes."""
    import zmq
    mock_ctx = MagicMock()
    mock_ctx_class.return_value = mock_ctx
    mock_sock = MagicMock()
    mock_ctx.socket.return_value = mock_sock

    # First getpid call during connect
    mock_getpid.return_value = 1000

    sl = StatLog()
    sl.connect()

    # Verify initial connection
    self.assertEqual(sl.pid, 1000)
    connect_count = mock_sock.connect.call_count

    # Simulate fork - new PID
    mock_getpid.return_value = 2000

    # This should trigger reconnection
    sl.gauge('test', 1.0)

    # Should have reconnected (connect called again)
    self.assertGreater(mock_sock.connect.call_count, connect_count)


class TestStatLogErrorHandling(unittest.TestCase):
  """Test StatLog error handling."""

  @patch('openpilot.system.statsd.zmq.Context')
  def test_drops_on_zmq_again(self, mock_ctx_class):
    """Test _send drops message on zmq.Again error."""
    import zmq
    mock_ctx = MagicMock()
    mock_ctx_class.return_value = mock_ctx
    mock_sock = MagicMock()
    mock_ctx.socket.return_value = mock_sock
    mock_sock.send_string.side_effect = zmq.error.Again()

    sl = StatLog()
    sl.connect()

    # Should not raise
    sl.gauge('test', 1.0)


if __name__ == '__main__':
  unittest.main()
