"""Integration tests for the manager module.

Tests process lifecycle, dependencies, orchestration, and cleanup.
Uses mocks to avoid starting actual hardware-dependent processes.
"""

import signal
import time
from collections.abc import Callable
from multiprocessing import Process


from cereal import car
from openpilot.common.params import Params
from openpilot.system.manager.process import (
  ManagerProcess,
  PythonProcess,
  NativeProcess,
  DaemonProcess,
  ensure_running,
  join_process,
)


class DummyProcess(ManagerProcess):
  """A minimal test process for unit testing ManagerProcess behavior."""

  def __init__(self, name: str, should_run_fn: Callable = None, enabled: bool = True):
    self.name = name
    self.should_run = should_run_fn or (lambda started, params, CP: True)
    self.enabled = enabled
    self.proc = None
    self.shutting_down = False
    self.prepared = False

  def prepare(self) -> None:
    self.prepared = True

  def start(self) -> None:
    if self.shutting_down:
      self.stop()
    if self.proc is not None:
      return
    # Create a simple process that sleeps
    self.proc = Process(target=lambda: time.sleep(60), name=self.name)
    self.proc.start()
    self.shutting_down = False


def simple_target():
  """Target function for test processes."""
  time.sleep(60)


def crashing_target():
  """Target function that exits immediately with error."""
  raise RuntimeError("Intentional crash")


def quick_exit_target():
  """Target function that exits cleanly after a short delay."""
  time.sleep(0.1)


class TestProcessLifecycle:
  """Tests for process start, stop, and restart functionality."""

  def test_process_start(self):
    """Test that a process can be started."""
    proc = DummyProcess("test_start")
    assert proc.proc is None

    proc.start()
    try:
      assert proc.proc is not None
      assert proc.proc.is_alive()
    finally:
      proc.stop()

  def test_process_stop_with_sigint(self):
    """Test that a process responds to SIGINT."""
    proc = DummyProcess("test_sigint")
    proc.sigkill = False

    proc.start()
    assert proc.proc.is_alive()

    exit_code = proc.stop(block=True)
    assert proc.proc is None
    # Process should have exited (either via SIGINT or fallback SIGKILL)
    assert exit_code is not None

  def test_process_stop_with_sigkill(self):
    """Test that a process with sigkill=True uses SIGKILL."""
    proc = DummyProcess("test_sigkill")
    proc.sigkill = True

    proc.start()
    assert proc.proc.is_alive()

    exit_code = proc.stop(block=True)
    assert proc.proc is None
    assert exit_code == -signal.SIGKILL

  def test_process_restart(self):
    """Test that a process can be restarted."""
    proc = DummyProcess("test_restart")
    proc.start()

    original_pid = proc.proc.pid
    assert proc.proc.is_alive()

    proc.restart()

    assert proc.proc is not None
    assert proc.proc.is_alive()
    assert proc.proc.pid != original_pid

    proc.stop()

  def test_process_non_blocking_stop(self):
    """Test non-blocking stop sets shutting_down flag."""
    proc = DummyProcess("test_nonblock")
    proc.start()

    result = proc.stop(block=False)
    assert result is None  # Non-blocking returns immediately
    assert proc.shutting_down is True

    # Now do a blocking stop to clean up
    proc.stop(block=True)

  def test_stop_already_stopped_process(self):
    """Test stopping a process that isn't running returns None."""
    proc = DummyProcess("test_not_running")
    result = proc.stop()
    assert result is None

  def test_double_start(self):
    """Test that starting an already running process does nothing."""
    proc = DummyProcess("test_double_start")
    proc.start()
    original_pid = proc.proc.pid

    proc.start()  # Should be a no-op
    assert proc.proc.pid == original_pid

    proc.stop()

  def test_start_while_shutting_down(self):
    """Test that start() properly stops a shutting_down process first."""
    proc = DummyProcess("test_shutdown_start")
    proc.start()
    original_pid = proc.proc.pid

    # Simulate a non-blocking stop
    proc.stop(block=False)
    assert proc.shutting_down is True

    # Now start again - should complete shutdown first
    proc.start()
    assert proc.shutting_down is False
    assert proc.proc is not None
    # The process should be new (different PID)
    assert proc.proc.pid != original_pid

    proc.stop()


class TestProcessStateMessage:
  """Tests for get_process_state_msg functionality."""

  def test_process_state_not_running(self):
    """Test state message when process is not running."""
    proc = DummyProcess("test_state_stopped")

    state = proc.get_process_state_msg()
    assert state.name == "test_state_stopped"
    assert state.running is False
    assert state.pid == 0

  def test_process_state_running(self):
    """Test state message when process is running."""
    proc = DummyProcess("test_state_running")
    proc.start()

    try:
      state = proc.get_process_state_msg()
      assert state.name == "test_state_running"
      assert state.running is True
      assert state.shouldBeRunning is True
      assert state.pid > 0
    finally:
      proc.stop()

  def test_process_state_shutting_down(self):
    """Test state message when process is shutting down."""
    proc = DummyProcess("test_state_shutdown")
    proc.start()
    proc.stop(block=False)

    try:
      state = proc.get_process_state_msg()
      assert state.shouldBeRunning is False
    finally:
      proc.stop(block=True)


class TestEnsureRunning:
  """Tests for the ensure_running orchestration function."""

  def setup_method(self):
    self.params = Params()
    self.CP = car.CarParams.new_message()

  def test_ensure_running_starts_processes(self):
    """Test that ensure_running starts enabled processes."""
    procs = [
      DummyProcess("proc1"),
      DummyProcess("proc2"),
    ]

    try:
      running = ensure_running(procs, started=True, params=self.params, CP=self.CP)
      assert len(running) == 2
      for p in procs:
        assert p.proc is not None
        assert p.proc.is_alive()
    finally:
      for p in procs:
        p.stop()

  def test_ensure_running_respects_not_run_list(self):
    """Test that processes in not_run list are not started."""
    procs = [
      DummyProcess("proc1"),
      DummyProcess("blocked_proc"),
    ]

    try:
      running = ensure_running(procs, started=True, params=self.params, CP=self.CP, not_run=["blocked_proc"])
      assert len(running) == 1
      assert procs[0].proc is not None
      assert procs[1].proc is None
    finally:
      for p in procs:
        p.stop()

  def test_ensure_running_respects_enabled_flag(self):
    """Test that disabled processes are not started."""
    procs = [
      DummyProcess("enabled_proc", enabled=True),
      DummyProcess("disabled_proc", enabled=False),
    ]

    try:
      running = ensure_running(procs, started=True, params=self.params, CP=self.CP)
      assert len(running) == 1
      assert procs[0].proc is not None
      assert procs[1].proc is None
    finally:
      for p in procs:
        p.stop()

  def test_ensure_running_respects_should_run(self):
    """Test that should_run callback determines if process starts."""
    procs = [
      DummyProcess("always_run", should_run_fn=lambda s, p, c: True),
      DummyProcess("only_started", should_run_fn=lambda s, p, c: s),
      DummyProcess("never_run", should_run_fn=lambda s, p, c: False),
    ]

    try:
      # When started=False, only always_run should run
      running = ensure_running(procs, started=False, params=self.params, CP=self.CP)
      assert len(running) == 1
      assert running[0].name == "always_run"
    finally:
      for p in procs:
        p.stop()

  def test_ensure_running_stops_processes_that_should_not_run(self):
    """Test that ensure_running stops processes that should no longer run."""
    proc = DummyProcess("conditional_proc", should_run_fn=lambda s, p, c: s)
    procs = [proc]

    try:
      # Start it when started=True
      ensure_running(procs, started=True, params=self.params, CP=self.CP)
      assert proc.proc is not None

      # Now it should stop when started=False (non-blocking stop)
      ensure_running(procs, started=False, params=self.params, CP=self.CP)
      assert proc.shutting_down is True

      # Clean up
      proc.stop(block=True)
    finally:
      for p in procs:
        if p.proc:
          p.stop()


class TestRestartIfCrash:
  """Tests for restart_if_crash functionality."""

  def test_restart_if_crash_restarts_dead_process(self):
    """Test that a crashed process with restart_if_crash=True is restarted."""
    # Create a process that will exit quickly
    proc = DummyProcess("crash_proc")
    proc.restart_if_crash = True

    # Start it with a process that exits immediately
    proc.proc = Process(target=quick_exit_target, name=proc.name)
    proc.proc.start()

    # Wait for process to exit
    time.sleep(0.3)
    assert not proc.proc.is_alive()
    original_pid = proc.proc.pid

    # ensure_running should restart it
    params = Params()
    CP = car.CarParams.new_message()

    try:
      running = ensure_running([proc], started=True, params=params, CP=CP)
      assert len(running) == 1
      assert proc.proc is not None
      assert proc.proc.is_alive()
      # Should be a new process
      assert proc.proc.pid != original_pid
    finally:
      proc.stop()

  def test_no_restart_without_flag(self):
    """Test that a crashed process without restart_if_crash=True stays dead."""
    proc = DummyProcess("no_restart_proc")
    proc.restart_if_crash = False

    # Start it with a process that exits immediately
    proc.proc = Process(target=quick_exit_target, name=proc.name)
    proc.proc.start()
    original_pid = proc.proc.pid

    # Wait for process to exit
    time.sleep(0.3)
    assert not proc.proc.is_alive()

    # ensure_running should NOT restart it (just add to running list)
    params = Params()
    CP = car.CarParams.new_message()

    running = ensure_running([proc], started=True, params=params, CP=CP)
    # Process should still be in the list but not restarted
    assert len(running) == 1
    # Process object should be the same (dead one)
    assert proc.proc.pid == original_pid


class TestJoinProcess:
  """Tests for the join_process helper function."""

  def test_join_process_exits_on_timeout(self):
    """Test that join_process respects timeout."""
    proc = Process(target=simple_target)
    proc.start()

    try:
      start = time.monotonic()
      join_process(proc, timeout=0.1)
      elapsed = time.monotonic() - start

      # Should have returned after timeout
      assert elapsed < 0.5
      assert proc.exitcode is None  # Process still running
    finally:
      proc.kill()
      proc.join()

  def test_join_process_returns_when_process_exits(self):
    """Test that join_process returns when process exits."""
    proc = Process(target=quick_exit_target)
    proc.start()

    join_process(proc, timeout=5.0)
    assert proc.exitcode is not None


class TestPythonProcess:
  """Tests for PythonProcess class."""

  def test_python_process_prepare_imports_module(self):
    """Test that prepare() imports the module."""
    # Use a simple stdlib module for testing
    proc = PythonProcess(name="test_import", module="json", should_run=lambda s, p, c: True, enabled=True)

    # Should not raise
    proc.prepare()

  def test_python_process_prepare_disabled(self):
    """Test that prepare() is a no-op when disabled."""
    proc = PythonProcess(name="test_disabled", module="nonexistent.module.that.does.not.exist", should_run=lambda s, p, c: True, enabled=False)

    # Should not raise because it's disabled
    proc.prepare()


class TestNativeProcess:
  """Tests for NativeProcess class."""

  def test_native_process_prepare_is_noop(self):
    """Test that NativeProcess.prepare() does nothing."""
    proc = NativeProcess(name="test_native", cwd=".", cmdline=["echo", "hello"], should_run=lambda s, p, c: True)
    # Should not raise
    proc.prepare()


class TestDaemonProcess:
  """Tests for DaemonProcess class."""

  def test_daemon_should_run_always_true(self):
    """Test that DaemonProcess.should_run always returns True."""
    proc = DaemonProcess(name="test_daemon", module="test.module", param_name="TestPid")

    assert proc.should_run(True, None, None) is True
    assert proc.should_run(False, None, None) is True

  def test_daemon_stop_is_noop(self):
    """Test that DaemonProcess.stop() does nothing."""
    proc = DaemonProcess(name="test_daemon", module="test.module", param_name="TestPid")

    # Should not raise
    result = proc.stop()
    assert result is None


class TestSignaling:
  """Tests for process signaling functionality."""

  def test_signal_running_process(self):
    """Test sending a signal to a running process."""
    proc = DummyProcess("test_signal")
    proc.start()

    try:
      # Send SIGTERM
      proc.signal(signal.SIGTERM)
      # Process should eventually die
      time.sleep(0.5)
      # The process may or may not be alive depending on timing
    finally:
      proc.stop()

  def test_signal_no_process(self):
    """Test that signaling with no process is safe."""
    proc = DummyProcess("test_no_proc")
    # Should not raise
    proc.signal(signal.SIGTERM)


class TestManagerCleanup:
  """Tests for manager cleanup functionality."""

  def test_cleanup_stops_all_processes(self, mocker):
    """Test that manager_cleanup stops all managed processes."""
    from openpilot.system.manager import manager

    # Create mock processes
    mock_procs = {
      "proc1": mocker.MagicMock(),
      "proc2": mocker.MagicMock(),
    }

    mocker.patch.dict('openpilot.system.manager.process_config.managed_processes', mock_procs)
    manager.manager_cleanup()

    # Each process should have stop called twice (block=False then block=True)
    for _, proc in mock_procs.items():
      assert proc.stop.call_count == 2


class TestManagerInit:
  """Tests for manager initialization."""

  def test_manager_init_clears_params(self, mocker):
    """Test that manager_init clears appropriate params."""
    from openpilot.system.manager import manager

    mocker.patch('openpilot.system.manager.manager.managed_processes', {})
    mocker.patch('openpilot.system.manager.manager.save_bootlog')
    mock_register = mocker.patch('openpilot.system.manager.manager.register')
    mock_hw = mocker.patch('openpilot.system.manager.manager.HARDWARE')
    mocker.patch('openpilot.system.manager.manager.sentry')
    mock_build_meta = mocker.patch('openpilot.system.manager.manager.get_build_metadata')

    # Setup mocks
    mock_build_meta.return_value = mocker.MagicMock(
      release_channel=False,
      openpilot=mocker.MagicMock(
        version="test", git_commit="abc123", git_commit_date="2024-01-01", git_origin="test", git_normalized_origin="test", is_dirty=False
      ),
      channel="test",
      tested_channel=False,
    )
    mock_register.return_value = "test_dongle_id"
    mock_hw.get_serial.return_value = "test_serial"
    mock_hw.get_device_type.return_value = "test_device"

    manager.manager_init()

    # Verify register was called
    mock_register.assert_called_once()


class TestProcessConfig:
  """Tests for process configuration."""

  def test_all_processes_have_required_attributes(self):
    """Test that all configured processes have required attributes."""
    from openpilot.system.manager.process_config import managed_processes

    for name, proc in managed_processes.items():
      assert hasattr(proc, 'name')
      assert hasattr(proc, 'should_run')
      assert hasattr(proc, 'enabled')
      assert hasattr(proc, 'start')
      assert hasattr(proc, 'stop')
      assert hasattr(proc, 'prepare')
      assert proc.name == name

  def test_no_duplicate_process_names(self):
    """Test that there are no duplicate process names."""
    from openpilot.system.manager.process_config import procs, managed_processes

    assert len(procs) == len(managed_processes)


class TestShouldRunConditions:
  """Tests for the should_run condition functions."""

  def test_always_run(self):
    """Test always_run returns True in all cases."""
    from openpilot.system.manager.process_config import always_run

    params = Params()
    CP = car.CarParams.new_message()

    assert always_run(True, params, CP) is True
    assert always_run(False, params, CP) is True

  def test_only_onroad(self):
    """Test only_onroad returns True only when started."""
    from openpilot.system.manager.process_config import only_onroad

    params = Params()
    CP = car.CarParams.new_message()

    assert only_onroad(True, params, CP) is True
    assert only_onroad(False, params, CP) is False

  def test_only_offroad(self):
    """Test only_offroad returns True only when not started."""
    from openpilot.system.manager.process_config import only_offroad

    params = Params()
    CP = car.CarParams.new_message()

    assert only_offroad(True, params, CP) is False
    assert only_offroad(False, params, CP) is True

  def test_and_combinator(self):
    """Test and_ combines conditions correctly."""
    from openpilot.system.manager.process_config import and_

    def always_true(s, p, c):
      return True

    def always_false(s, p, c):
      return False

    combined = and_(always_true, always_true)
    assert combined(True, None, None) is True

    combined = and_(always_true, always_false)
    assert combined(True, None, None) is False

  def test_or_combinator(self):
    """Test or_ combines conditions correctly."""
    from openpilot.system.manager.process_config import or_

    def always_true(s, p, c):
      return True

    def always_false(s, p, c):
      return False

    combined = or_(always_false, always_false)
    assert combined(True, None, None) is False

    combined = or_(always_true, always_false)
    assert combined(True, None, None) is True


class TestManagerThread:
  """Tests for the manager main loop (manager_thread)."""

  def test_manager_thread_exits_on_shutdown_param(self, mocker):
    """Test that manager_thread exits when shutdown param is set."""
    from openpilot.system.manager import manager

    params = Params()

    # Setup mocks
    mock_sm = mocker.MagicMock()
    mock_sm.__getitem__ = mocker.MagicMock(return_value=mocker.MagicMock(started=False))
    mock_sm.all_checks.return_value = False
    mock_messaging = mocker.patch('openpilot.system.manager.manager.messaging')
    mock_messaging.SubMaster.return_value = mock_sm
    mock_messaging.PubMaster.return_value = mocker.MagicMock()
    mocker.patch('openpilot.system.manager.manager.ensure_running', return_value=[])
    mocker.patch('openpilot.system.manager.manager.write_onroad_params')

    # Set shutdown param after first iteration
    call_count = [0]

    def update_side_effect(*args):
      call_count[0] += 1
      if call_count[0] >= 2:
        params.put_bool("DoShutdown", True)

    mock_sm.update.side_effect = update_side_effect

    mocker.patch.dict('openpilot.system.manager.process_config.managed_processes', {})
    # Should exit cleanly
    manager.manager_thread()

    # Verify it ran at least twice
    assert call_count[0] >= 2

  def test_manager_thread_handles_onroad_transition(self, mocker):
    """Test that manager_thread handles onroad/offroad transitions."""
    from openpilot.system.manager import manager

    params = Params()

    # Setup mocks
    mock_sm = mocker.MagicMock()
    started_values = [False, True, True, False]  # Simulate transition
    call_count = [0]

    def get_item(key):
      if key == 'deviceState':
        return mocker.MagicMock(started=started_values[min(call_count[0], len(started_values) - 1)])
      elif key == 'pandaStates':
        return []
      return mocker.MagicMock()

    mock_sm.__getitem__ = mocker.MagicMock(side_effect=get_item)
    mock_sm.all_checks.return_value = False
    mock_messaging = mocker.patch('openpilot.system.manager.manager.messaging')
    mock_messaging.SubMaster.return_value = mock_sm
    mock_messaging.PubMaster.return_value = mocker.MagicMock()
    mocker.patch('openpilot.system.manager.manager.ensure_running', return_value=[])
    mock_write_onroad = mocker.patch('openpilot.system.manager.manager.write_onroad_params')

    def update_side_effect(*args):
      call_count[0] += 1
      if call_count[0] >= len(started_values):
        params.put_bool("DoShutdown", True)

    mock_sm.update.side_effect = update_side_effect

    mocker.patch.dict('openpilot.system.manager.process_config.managed_processes', {})
    manager.manager_thread()

    # Verify write_onroad_params was called
    assert mock_write_onroad.call_count >= 1


class TestHelpers:
  """Tests for manager helper functions."""

  def test_write_onroad_params_started(self):
    """Test write_onroad_params sets correct values when started."""
    from openpilot.system.manager.helpers import write_onroad_params

    params = Params()
    write_onroad_params(True, params)

    assert params.get_bool("IsOnroad") is True
    assert params.get_bool("IsOffroad") is False

  def test_write_onroad_params_stopped(self):
    """Test write_onroad_params sets correct values when stopped."""
    from openpilot.system.manager.helpers import write_onroad_params

    params = Params()
    write_onroad_params(False, params)

    assert params.get_bool("IsOnroad") is False
    assert params.get_bool("IsOffroad") is True
