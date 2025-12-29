"""Tests for system/statsd.py - statistics logging."""

import os

from openpilot.system.statsd import METRIC_TYPE, StatLog


class TestMetricType:
  """Test METRIC_TYPE class."""

  def test_gauge_value(self):
    """Test GAUGE is 'g'."""
    assert METRIC_TYPE.GAUGE == 'g'

  def test_sample_value(self):
    """Test SAMPLE is 'sa'."""
    assert METRIC_TYPE.SAMPLE == 'sa'


class TestStatLogInit:
  """Test StatLog initialization."""

  def test_init_defaults(self):
    """Test StatLog initializes with None values."""
    sl = StatLog()

    assert sl.pid is None
    assert sl.zctx is None
    assert sl.sock is None


class TestStatLogConnect:
  """Test StatLog.connect method."""

  def test_connect_creates_context(self, mocker):
    """Test connect creates zmq Context."""
    mock_ctx_class = mocker.patch('openpilot.system.statsd.zmq.Context')
    mock_ctx = mocker.MagicMock()
    mock_ctx_class.return_value = mock_ctx
    mock_sock = mocker.MagicMock()
    mock_ctx.socket.return_value = mock_sock

    sl = StatLog()
    sl.connect()

    mock_ctx_class.assert_called_once()
    assert sl.zctx == mock_ctx

  def test_connect_creates_push_socket(self, mocker):
    """Test connect creates PUSH socket."""
    import zmq

    mock_ctx_class = mocker.patch('openpilot.system.statsd.zmq.Context')
    mock_ctx = mocker.MagicMock()
    mock_ctx_class.return_value = mock_ctx
    mock_sock = mocker.MagicMock()
    mock_ctx.socket.return_value = mock_sock

    sl = StatLog()
    sl.connect()

    mock_ctx.socket.assert_called_once_with(zmq.PUSH)

  def test_connect_sets_linger(self, mocker):
    """Test connect sets LINGER socket option."""
    import zmq

    mock_ctx_class = mocker.patch('openpilot.system.statsd.zmq.Context')
    mock_ctx = mocker.MagicMock()
    mock_ctx_class.return_value = mock_ctx
    mock_sock = mocker.MagicMock()
    mock_ctx.socket.return_value = mock_sock

    sl = StatLog()
    sl.connect()

    mock_sock.setsockopt.assert_called_once_with(zmq.LINGER, 10)

  def test_connect_connects_to_socket(self, mocker):
    """Test connect connects to STATS_SOCKET."""
    mocker.patch('openpilot.system.statsd.STATS_SOCKET', 'ipc:///tmp/test_stats')
    mock_ctx_class = mocker.patch('openpilot.system.statsd.zmq.Context')
    mock_ctx = mocker.MagicMock()
    mock_ctx_class.return_value = mock_ctx
    mock_sock = mocker.MagicMock()
    mock_ctx.socket.return_value = mock_sock

    sl = StatLog()
    sl.connect()

    mock_sock.connect.assert_called_once_with('ipc:///tmp/test_stats')

  def test_connect_stores_pid(self, mocker):
    """Test connect stores current PID."""
    mock_ctx_class = mocker.patch('openpilot.system.statsd.zmq.Context')
    mock_ctx = mocker.MagicMock()
    mock_ctx_class.return_value = mock_ctx
    mock_sock = mocker.MagicMock()
    mock_ctx.socket.return_value = mock_sock

    sl = StatLog()
    sl.connect()

    assert sl.pid == os.getpid()


class TestStatLogGauge:
  """Test StatLog.gauge method."""

  def test_gauge_sends_correct_format(self, mocker):
    """Test gauge sends metric in correct format."""
    import zmq

    mock_ctx_class = mocker.patch('openpilot.system.statsd.zmq.Context')
    mock_ctx = mocker.MagicMock()
    mock_ctx_class.return_value = mock_ctx
    mock_sock = mocker.MagicMock()
    mock_ctx.socket.return_value = mock_sock

    sl = StatLog()
    sl.connect()
    sl.gauge('test_metric', 42.5)

    mock_sock.send_string.assert_called_once_with('test_metric:42.5|g', zmq.NOBLOCK)


class TestStatLogSample:
  """Test StatLog.sample method."""

  def test_sample_sends_correct_format(self, mocker):
    """Test sample sends metric in correct format."""
    import zmq

    mock_ctx_class = mocker.patch('openpilot.system.statsd.zmq.Context')
    mock_ctx = mocker.MagicMock()
    mock_ctx_class.return_value = mock_ctx
    mock_sock = mocker.MagicMock()
    mock_ctx.socket.return_value = mock_sock

    sl = StatLog()
    sl.connect()
    sl.sample('timing', 0.123)

    mock_sock.send_string.assert_called_once_with('timing:0.123|sa', zmq.NOBLOCK)


class TestStatLogReconnect:
  """Test StatLog reconnection on fork."""

  def test_reconnects_on_fork(self, mocker):
    """Test _send reconnects when PID changes."""
    mock_ctx_class = mocker.patch('openpilot.system.statsd.zmq.Context')
    mock_ctx = mocker.MagicMock()
    mock_ctx_class.return_value = mock_ctx
    mock_sock = mocker.MagicMock()
    mock_ctx.socket.return_value = mock_sock

    mock_getpid = mocker.patch('openpilot.system.statsd.os.getpid')
    # First getpid call during connect
    mock_getpid.return_value = 1000

    sl = StatLog()
    sl.connect()

    # Verify initial connection
    assert sl.pid == 1000
    connect_count = mock_sock.connect.call_count

    # Simulate fork - new PID
    mock_getpid.return_value = 2000

    # This should trigger reconnection
    sl.gauge('test', 1.0)

    # Should have reconnected (connect called again)
    assert mock_sock.connect.call_count > connect_count


class TestStatLogErrorHandling:
  """Test StatLog error handling."""

  def test_drops_on_zmq_again(self, mocker):
    """Test _send drops message on zmq.Again error."""
    import zmq

    mock_ctx_class = mocker.patch('openpilot.system.statsd.zmq.Context')
    mock_ctx = mocker.MagicMock()
    mock_ctx_class.return_value = mock_ctx
    mock_sock = mocker.MagicMock()
    mock_ctx.socket.return_value = mock_sock
    mock_sock.send_string.side_effect = zmq.error.Again()

    sl = StatLog()
    sl.connect()

    # Should not raise
    sl.gauge('test', 1.0)
