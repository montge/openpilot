"""Tests for system/manager/process.py - process management utilities."""
import signal
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from multiprocessing import Process

from cereal import car
from openpilot.common.params import Params
from openpilot.system.manager.process import (
  join_process, ManagerProcess, NativeProcess, PythonProcess, DaemonProcess,
  ensure_running,
)


class TestJoinProcess(unittest.TestCase):
  """Test join_process function."""

  def test_join_process_exits_immediately_if_done(self):
    """Test join_process returns quickly if process already exited."""
    mock_proc = MagicMock(spec=Process)
    mock_proc.exitcode = 0

    start = time.monotonic()
    join_process(mock_proc, timeout=5.0)
    elapsed = time.monotonic() - start

    self.assertLess(elapsed, 1.0)

  def test_join_process_waits_until_timeout(self):
    """Test join_process waits up to timeout for exit."""
    mock_proc = MagicMock(spec=Process)
    # exitcode stays None (still running)
    type(mock_proc).exitcode = PropertyMock(return_value=None)

    start = time.monotonic()
    join_process(mock_proc, timeout=0.1)
    elapsed = time.monotonic() - start

    self.assertGreaterEqual(elapsed, 0.1)
    self.assertLess(elapsed, 0.5)


class TestManagerProcess(unittest.TestCase):
  """Test ManagerProcess base class."""

  def _create_manager_process(self):
    """Create a concrete ManagerProcess for testing."""
    class ConcreteProcess(ManagerProcess):
      def prepare(self):
        pass

      def start(self):
        pass

    proc = ConcreteProcess()
    proc.name = "test_process"
    return proc

  def test_get_process_state_msg_no_proc(self):
    """Test get_process_state_msg when no process is running."""
    mp = self._create_manager_process()
    state = mp.get_process_state_msg()

    self.assertEqual(state.name, "test_process")
    self.assertFalse(state.running)

  def test_get_process_state_msg_with_proc(self):
    """Test get_process_state_msg with a running process."""
    mp = self._create_manager_process()
    mp.proc = MagicMock(spec=Process)
    mp.proc.is_alive.return_value = True
    mp.proc.pid = 12345
    mp.proc.exitcode = None

    state = mp.get_process_state_msg()

    self.assertEqual(state.name, "test_process")
    self.assertTrue(state.running)
    self.assertEqual(state.pid, 12345)

  def test_signal_no_proc(self):
    """Test signal does nothing when no process."""
    mp = self._create_manager_process()
    # Should not raise
    mp.signal(signal.SIGTERM)

  def test_signal_exited_proc(self):
    """Test signal does nothing when process already exited."""
    mp = self._create_manager_process()
    mp.proc = MagicMock(spec=Process)
    mp.proc.exitcode = 0
    mp.proc.pid = 12345

    with patch('os.kill') as mock_kill:
      mp.signal(signal.SIGTERM)
      mock_kill.assert_not_called()

  def test_signal_no_pid(self):
    """Test signal does nothing when no PID."""
    mp = self._create_manager_process()
    mp.proc = MagicMock(spec=Process)
    mp.proc.exitcode = None
    mp.proc.pid = None

    with patch('os.kill') as mock_kill:
      mp.signal(signal.SIGTERM)
      mock_kill.assert_not_called()

  @patch('os.kill')
  def test_signal_sends_signal(self, mock_kill):
    """Test signal sends signal to running process."""
    mp = self._create_manager_process()
    mp.proc = MagicMock(spec=Process)
    mp.proc.exitcode = None
    mp.proc.pid = 12345

    mp.signal(signal.SIGTERM)

    mock_kill.assert_called_once_with(12345, signal.SIGTERM)

  def test_stop_no_proc(self):
    """Test stop returns None when no process."""
    mp = self._create_manager_process()
    result = mp.stop()
    self.assertIsNone(result)

  def test_restart_stops_and_starts(self):
    """Test restart calls stop and start."""
    mp = self._create_manager_process()
    mp.stop = MagicMock(return_value=0)
    mp.start = MagicMock()

    mp.restart()

    mp.stop.assert_called_once_with(sig=signal.SIGKILL)
    mp.start.assert_called_once()


class TestNativeProcess(unittest.TestCase):
  """Test NativeProcess class."""

  def test_init(self):
    """Test NativeProcess initialization."""
    should_run = lambda started, params, CP: True
    proc = NativeProcess(
      name="test_native",
      cwd="test_cwd",
      cmdline=["./test"],
      should_run=should_run,
      enabled=True,
      sigkill=False,
    )

    self.assertEqual(proc.name, "test_native")
    self.assertEqual(proc.cwd, "test_cwd")
    self.assertEqual(proc.cmdline, ["./test"])
    self.assertTrue(proc.enabled)
    self.assertFalse(proc.sigkill)

  def test_prepare_does_nothing(self):
    """Test prepare() does nothing for native processes."""
    proc = NativeProcess(
      name="test",
      cwd=".",
      cmdline=["./test"],
      should_run=lambda s, p, c: True,
    )
    # Should not raise
    proc.prepare()

  def test_start_when_already_running(self):
    """Test start() does nothing if process is already running."""
    proc = NativeProcess(
      name="test",
      cwd=".",
      cmdline=["./test"],
      should_run=lambda s, p, c: True,
    )
    proc.proc = MagicMock()

    with patch('multiprocessing.Process') as mock_process:
      proc.start()
      mock_process.assert_not_called()


class TestPythonProcess(unittest.TestCase):
  """Test PythonProcess class."""

  def test_init(self):
    """Test PythonProcess initialization."""
    should_run = lambda started, params, CP: True
    proc = PythonProcess(
      name="test_python",
      module="openpilot.system.test",
      should_run=should_run,
      enabled=True,
      sigkill=False,
      restart_if_crash=True,
    )

    self.assertEqual(proc.name, "test_python")
    self.assertEqual(proc.module, "openpilot.system.test")
    self.assertTrue(proc.enabled)
    self.assertFalse(proc.sigkill)
    self.assertTrue(proc.restart_if_crash)

  def test_start_when_already_running(self):
    """Test start() does nothing if process is already running."""
    proc = PythonProcess(
      name="test",
      module="test_module",
      should_run=lambda s, p, c: True,
    )
    proc.proc = MagicMock()

    with patch('multiprocessing.Process') as mock_process:
      proc.start()
      mock_process.assert_not_called()


class TestDaemonProcess(unittest.TestCase):
  """Test DaemonProcess class."""

  def test_init(self):
    """Test DaemonProcess initialization."""
    proc = DaemonProcess(
      name="test_daemon",
      module="openpilot.system.test",
      param_name="TestPid",
      enabled=True,
    )

    self.assertEqual(proc.name, "test_daemon")
    self.assertEqual(proc.module, "openpilot.system.test")
    self.assertEqual(proc.param_name, "TestPid")
    self.assertTrue(proc.enabled)

  def test_should_run_always_true(self):
    """Test should_run always returns True."""
    result = DaemonProcess.should_run(True, None, None)
    self.assertTrue(result)

    result = DaemonProcess.should_run(False, None, None)
    self.assertTrue(result)

  def test_prepare_does_nothing(self):
    """Test prepare() does nothing for daemon processes."""
    proc = DaemonProcess(
      name="test",
      module="test_module",
      param_name="TestPid",
    )
    # Should not raise
    proc.prepare()

  def test_stop_does_nothing(self):
    """Test stop() does nothing for daemon processes."""
    proc = DaemonProcess(
      name="test",
      module="test_module",
      param_name="TestPid",
    )
    # Should not raise and return None
    result = proc.stop()
    self.assertIsNone(result)


class TestEnsureRunning(unittest.TestCase):
  """Test ensure_running function."""

  def _create_mock_process(self, name, enabled=True, should_run_val=True):
    """Create a mock ManagerProcess."""
    mock_proc = MagicMock(spec=ManagerProcess)
    mock_proc.name = name
    mock_proc.enabled = enabled
    mock_proc.should_run = MagicMock(return_value=should_run_val)
    mock_proc.restart_if_crash = False
    mock_proc.proc = None
    return mock_proc

  def test_ensure_running_starts_enabled_procs(self):
    """Test ensure_running starts enabled processes."""
    proc1 = self._create_mock_process("proc1")
    proc2 = self._create_mock_process("proc2")

    running = ensure_running([proc1, proc2], started=True)

    self.assertEqual(len(running), 2)
    proc1.start.assert_called_once()
    proc2.start.assert_called_once()

  def test_ensure_running_skips_disabled_procs(self):
    """Test ensure_running skips disabled processes."""
    proc1 = self._create_mock_process("proc1", enabled=True)
    proc2 = self._create_mock_process("proc2", enabled=False)

    running = ensure_running([proc1, proc2], started=True)

    self.assertEqual(len(running), 1)
    proc1.start.assert_called_once()
    proc2.start.assert_not_called()

  def test_ensure_running_skips_not_run_procs(self):
    """Test ensure_running skips processes in not_run list."""
    proc1 = self._create_mock_process("proc1")
    proc2 = self._create_mock_process("proc2")

    running = ensure_running([proc1, proc2], started=True, not_run=["proc2"])

    self.assertEqual(len(running), 1)
    proc1.start.assert_called_once()
    proc2.stop.assert_called_once_with(block=False)

  def test_ensure_running_stops_procs_that_should_not_run(self):
    """Test ensure_running stops processes that shouldn't run."""
    proc1 = self._create_mock_process("proc1", should_run_val=True)
    proc2 = self._create_mock_process("proc2", should_run_val=False)

    running = ensure_running([proc1, proc2], started=True)

    self.assertEqual(len(running), 1)
    proc2.stop.assert_called_once_with(block=False)

  def test_ensure_running_restarts_crashed_procs(self):
    """Test ensure_running restarts crashed processes with restart_if_crash."""
    proc = self._create_mock_process("proc1")
    proc.restart_if_crash = True
    proc.proc = MagicMock()
    proc.proc.is_alive.return_value = False
    proc.proc.exitcode = 1

    running = ensure_running([proc], started=True)

    self.assertEqual(len(running), 1)
    proc.restart.assert_called_once()


if __name__ == '__main__':
  unittest.main()
