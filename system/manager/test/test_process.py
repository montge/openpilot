"""Tests for system/manager/process.py - process management utilities."""

import signal
import time
from multiprocessing import Process

from openpilot.system.manager.process import (
  join_process,
  ManagerProcess,
  NativeProcess,
  PythonProcess,
  DaemonProcess,
  ensure_running,
)


class TestJoinProcess:
  """Test join_process function."""

  def test_join_process_exits_immediately_if_done(self, mocker):
    """Test join_process returns quickly if process already exited."""
    mock_proc = mocker.MagicMock(spec=Process)
    mock_proc.exitcode = 0

    start = time.monotonic()
    join_process(mock_proc, timeout=5.0)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0

  def test_join_process_waits_until_timeout(self, mocker):
    """Test join_process waits up to timeout for exit."""
    mock_proc = mocker.MagicMock(spec=Process)
    # exitcode stays None (still running)
    type(mock_proc).exitcode = mocker.PropertyMock(return_value=None)

    start = time.monotonic()
    join_process(mock_proc, timeout=0.1)
    elapsed = time.monotonic() - start

    assert elapsed >= 0.1
    assert elapsed < 0.5


class TestManagerProcess:
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

    assert state.name == "test_process"
    assert state.running is False

  def test_get_process_state_msg_with_proc(self, mocker):
    """Test get_process_state_msg with a running process."""
    mp = self._create_manager_process()
    mp.proc = mocker.MagicMock(spec=Process)
    mp.proc.is_alive.return_value = True
    mp.proc.pid = 12345
    mp.proc.exitcode = None

    state = mp.get_process_state_msg()

    assert state.name == "test_process"
    assert state.running is True
    assert state.pid == 12345

  def test_signal_no_proc(self):
    """Test signal does nothing when no process."""
    mp = self._create_manager_process()
    # Should not raise
    mp.signal(signal.SIGTERM)

  def test_signal_exited_proc(self, mocker):
    """Test signal does nothing when process already exited."""
    mp = self._create_manager_process()
    mp.proc = mocker.MagicMock(spec=Process)
    mp.proc.exitcode = 0
    mp.proc.pid = 12345

    mock_kill = mocker.patch('os.kill')
    mp.signal(signal.SIGTERM)
    mock_kill.assert_not_called()

  def test_signal_no_pid(self, mocker):
    """Test signal does nothing when no PID."""
    mp = self._create_manager_process()
    mp.proc = mocker.MagicMock(spec=Process)
    mp.proc.exitcode = None
    mp.proc.pid = None

    mock_kill = mocker.patch('os.kill')
    mp.signal(signal.SIGTERM)
    mock_kill.assert_not_called()

  def test_signal_sends_signal(self, mocker):
    """Test signal sends signal to running process."""
    mock_kill = mocker.patch('os.kill')
    mp = self._create_manager_process()
    mp.proc = mocker.MagicMock(spec=Process)
    mp.proc.exitcode = None
    mp.proc.pid = 12345

    mp.signal(signal.SIGTERM)

    mock_kill.assert_called_once_with(12345, signal.SIGTERM)

  def test_stop_no_proc(self):
    """Test stop returns None when no process."""
    mp = self._create_manager_process()
    result = mp.stop()
    assert result is None

  def test_restart_stops_and_starts(self, mocker):
    """Test restart calls stop and start."""
    mp = self._create_manager_process()
    mp.stop = mocker.MagicMock(return_value=0)
    mp.start = mocker.MagicMock()

    mp.restart()

    mp.stop.assert_called_once_with(sig=signal.SIGKILL)
    mp.start.assert_called_once()


class TestNativeProcess:
  """Test NativeProcess class."""

  def test_init(self):
    """Test NativeProcess initialization."""

    def should_run(started, params, CP):
      return True

    proc = NativeProcess(
      name="test_native",
      cwd="test_cwd",
      cmdline=["./test"],
      should_run=should_run,
      enabled=True,
      sigkill=False,
    )

    assert proc.name == "test_native"
    assert proc.cwd == "test_cwd"
    assert proc.cmdline == ["./test"]
    assert proc.enabled is True
    assert proc.sigkill is False

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

  def test_start_when_already_running(self, mocker):
    """Test start() does nothing if process is already running."""
    proc = NativeProcess(
      name="test",
      cwd=".",
      cmdline=["./test"],
      should_run=lambda s, p, c: True,
    )
    proc.proc = mocker.MagicMock()

    mock_process = mocker.patch('multiprocessing.Process')
    proc.start()
    mock_process.assert_not_called()


class TestPythonProcess:
  """Test PythonProcess class."""

  def test_init(self):
    """Test PythonProcess initialization."""

    def should_run(started, params, CP):
      return True

    proc = PythonProcess(
      name="test_python",
      module="openpilot.system.test",
      should_run=should_run,
      enabled=True,
      sigkill=False,
      restart_if_crash=True,
    )

    assert proc.name == "test_python"
    assert proc.module == "openpilot.system.test"
    assert proc.enabled is True
    assert proc.sigkill is False
    assert proc.restart_if_crash is True

  def test_start_when_already_running(self, mocker):
    """Test start() does nothing if process is already running."""
    proc = PythonProcess(
      name="test",
      module="test_module",
      should_run=lambda s, p, c: True,
    )
    proc.proc = mocker.MagicMock()

    mock_process = mocker.patch('multiprocessing.Process')
    proc.start()
    mock_process.assert_not_called()


class TestDaemonProcess:
  """Test DaemonProcess class."""

  def test_init(self):
    """Test DaemonProcess initialization."""
    proc = DaemonProcess(
      name="test_daemon",
      module="openpilot.system.test",
      param_name="TestPid",
      enabled=True,
    )

    assert proc.name == "test_daemon"
    assert proc.module == "openpilot.system.test"
    assert proc.param_name == "TestPid"
    assert proc.enabled is True

  def test_should_run_always_true(self):
    """Test should_run always returns True."""
    result = DaemonProcess.should_run(True, None, None)
    assert result is True

    result = DaemonProcess.should_run(False, None, None)
    assert result is True

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
    assert result is None


class TestEnsureRunning:
  """Test ensure_running function."""

  def _create_mock_process(self, mocker, name, enabled=True, should_run_val=True):
    """Create a mock ManagerProcess."""
    mock_proc = mocker.MagicMock(spec=ManagerProcess)
    mock_proc.name = name
    mock_proc.enabled = enabled
    mock_proc.should_run = mocker.MagicMock(return_value=should_run_val)
    mock_proc.restart_if_crash = False
    mock_proc.proc = None
    return mock_proc

  def test_ensure_running_starts_enabled_procs(self, mocker):
    """Test ensure_running starts enabled processes."""
    proc1 = self._create_mock_process(mocker, "proc1")
    proc2 = self._create_mock_process(mocker, "proc2")

    running = ensure_running([proc1, proc2], started=True)

    assert len(running) == 2
    proc1.start.assert_called_once()
    proc2.start.assert_called_once()

  def test_ensure_running_skips_disabled_procs(self, mocker):
    """Test ensure_running skips disabled processes."""
    proc1 = self._create_mock_process(mocker, "proc1", enabled=True)
    proc2 = self._create_mock_process(mocker, "proc2", enabled=False)

    running = ensure_running([proc1, proc2], started=True)

    assert len(running) == 1
    proc1.start.assert_called_once()
    proc2.start.assert_not_called()

  def test_ensure_running_skips_not_run_procs(self, mocker):
    """Test ensure_running skips processes in not_run list."""
    proc1 = self._create_mock_process(mocker, "proc1")
    proc2 = self._create_mock_process(mocker, "proc2")

    running = ensure_running([proc1, proc2], started=True, not_run=["proc2"])

    assert len(running) == 1
    proc1.start.assert_called_once()
    proc2.stop.assert_called_once_with(block=False)

  def test_ensure_running_stops_procs_that_should_not_run(self, mocker):
    """Test ensure_running stops processes that shouldn't run."""
    proc1 = self._create_mock_process(mocker, "proc1", should_run_val=True)
    proc2 = self._create_mock_process(mocker, "proc2", should_run_val=False)

    running = ensure_running([proc1, proc2], started=True)

    assert len(running) == 1
    proc2.stop.assert_called_once_with(block=False)

  def test_ensure_running_restarts_crashed_procs(self, mocker):
    """Test ensure_running restarts crashed processes with restart_if_crash."""
    proc = self._create_mock_process(mocker, "proc1")
    proc.restart_if_crash = True
    proc.proc = mocker.MagicMock()
    proc.proc.is_alive.return_value = False
    proc.proc.exitcode = 1

    running = ensure_running([proc], started=True)

    assert len(running) == 1
    proc.restart.assert_called_once()
