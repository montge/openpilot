import subprocess
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from openpilot.common.spinner import Spinner


class TestSpinner:
  @patch('openpilot.common.spinner.subprocess.Popen')
  def test_init_success(self, mock_popen):
    """Test Spinner initialization when subprocess starts successfully."""
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc

    spinner = Spinner()

    assert spinner.spinner_proc is mock_proc
    mock_popen.assert_called_once()

  @patch('openpilot.common.spinner.subprocess.Popen')
  def test_init_oserror(self, mock_popen):
    """Test Spinner initialization when subprocess fails with OSError."""
    mock_popen.side_effect = OSError("No such file")

    spinner = Spinner()

    assert spinner.spinner_proc is None

  @patch('openpilot.common.spinner.subprocess.Popen')
  def test_context_manager_enter(self, mock_popen):
    """Test Spinner as context manager - __enter__."""
    mock_popen.return_value = MagicMock()

    with Spinner() as spinner:
      assert spinner is not None
      assert hasattr(spinner, 'spinner_proc')

  @patch('openpilot.common.spinner.subprocess.Popen')
  def test_context_manager_exit(self, mock_popen):
    """Test Spinner as context manager - __exit__ closes spinner."""
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc

    with Spinner():
      pass

    mock_proc.kill.assert_called_once()
    mock_proc.communicate.assert_called_once_with(timeout=2.)

  @patch('openpilot.common.spinner.subprocess.Popen')
  def test_update_with_proc(self, mock_popen):
    """Test update method when spinner_proc is available."""
    mock_proc = MagicMock()
    mock_stdin = MagicMock()
    mock_proc.stdin = mock_stdin
    mock_popen.return_value = mock_proc

    spinner = Spinner()
    spinner.update("Loading...")

    mock_stdin.write.assert_called_once_with(b"Loading...\n")
    mock_stdin.flush.assert_called_once()

  @patch('openpilot.common.spinner.subprocess.Popen')
  def test_update_without_proc(self, mock_popen):
    """Test update method when spinner_proc is None."""
    mock_popen.side_effect = OSError("No such file")

    spinner = Spinner()
    # Should not raise an error
    spinner.update("Loading...")

  @patch('openpilot.common.spinner.subprocess.Popen')
  def test_update_broken_pipe(self, mock_popen):
    """Test update method handles BrokenPipeError on flush."""
    mock_proc = MagicMock()
    mock_stdin = MagicMock()
    mock_stdin.flush.side_effect = BrokenPipeError()
    mock_proc.stdin = mock_stdin
    mock_popen.return_value = mock_proc

    spinner = Spinner()
    # Should not raise an error
    spinner.update("Loading...")

    mock_stdin.write.assert_called_once()

  @patch('openpilot.common.spinner.subprocess.Popen')
  def test_update_progress(self, mock_popen):
    """Test update_progress method calculates percentage correctly."""
    mock_proc = MagicMock()
    mock_stdin = MagicMock()
    mock_proc.stdin = mock_stdin
    mock_popen.return_value = mock_proc

    spinner = Spinner()

    # 50% progress
    spinner.update_progress(cur=50.0, total=100.0)
    mock_stdin.write.assert_called_with(b"50\n")

    # 25% progress
    spinner.update_progress(cur=25.0, total=100.0)
    mock_stdin.write.assert_called_with(b"25\n")

    # 75% progress
    spinner.update_progress(cur=3.0, total=4.0)
    mock_stdin.write.assert_called_with(b"75\n")

  @patch('openpilot.common.spinner.subprocess.Popen')
  def test_update_progress_rounding(self, mock_popen):
    """Test update_progress rounds to nearest integer."""
    mock_proc = MagicMock()
    mock_stdin = MagicMock()
    mock_proc.stdin = mock_stdin
    mock_popen.return_value = mock_proc

    spinner = Spinner()

    # 33.33...% should round to 33
    spinner.update_progress(cur=1.0, total=3.0)
    mock_stdin.write.assert_called_with(b"33\n")

    # 66.66...% should round to 67
    spinner.update_progress(cur=2.0, total=3.0)
    mock_stdin.write.assert_called_with(b"67\n")

  @patch('openpilot.common.spinner.subprocess.Popen')
  def test_close_with_proc(self, mock_popen):
    """Test close method when spinner_proc is available."""
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc

    spinner = Spinner()
    spinner.close()

    mock_proc.kill.assert_called_once()
    mock_proc.communicate.assert_called_once_with(timeout=2.)
    assert spinner.spinner_proc is None

  @patch('openpilot.common.spinner.subprocess.Popen')
  def test_close_without_proc(self, mock_popen):
    """Test close method when spinner_proc is None."""
    mock_popen.side_effect = OSError("No such file")

    spinner = Spinner()
    # Should not raise an error
    spinner.close()

  @patch('openpilot.common.spinner.subprocess.Popen')
  def test_close_timeout_expired(self, mock_popen, capsys):
    """Test close method handles TimeoutExpired."""
    mock_proc = MagicMock()
    mock_proc.communicate.side_effect = subprocess.TimeoutExpired(cmd="spinner", timeout=2.)
    mock_popen.return_value = mock_proc

    spinner = Spinner()
    spinner.close()

    captured = capsys.readouterr()
    assert "WARNING: failed to kill spinner" in captured.out
    assert spinner.spinner_proc is None

  @patch('openpilot.common.spinner.subprocess.Popen')
  def test_close_idempotent(self, mock_popen):
    """Test that close can be called multiple times safely."""
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc

    spinner = Spinner()
    spinner.close()
    spinner.close()  # Second call should not error

    # Kill should only be called once (first close sets spinner_proc to None)
    mock_proc.kill.assert_called_once()

  @patch('openpilot.common.spinner.subprocess.Popen')
  def test_del_calls_close(self, mock_popen):
    """Test that __del__ calls close."""
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc

    spinner = Spinner()
    spinner.__del__()

    mock_proc.kill.assert_called_once()

  @patch('openpilot.common.spinner.subprocess.Popen')
  def test_exit_calls_close(self, mock_popen):
    """Test that __exit__ calls close with exception info."""
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc

    spinner = Spinner()
    spinner.__exit__(None, None, None)

    mock_proc.kill.assert_called_once()

  @patch('openpilot.common.spinner.subprocess.Popen')
  def test_update_unicode(self, mock_popen):
    """Test update method with unicode characters."""
    mock_proc = MagicMock()
    mock_stdin = MagicMock()
    mock_proc.stdin = mock_stdin
    mock_popen.return_value = mock_proc

    spinner = Spinner()
    spinner.update("Loading... \u2764")

    mock_stdin.write.assert_called_once_with("Loading... \u2764\n".encode('utf8'))

  @patch('openpilot.common.spinner.subprocess.Popen')
  def test_popen_arguments(self, mock_popen):
    """Test that Popen is called with correct arguments."""
    mock_popen.return_value = MagicMock()

    Spinner()

    call_args = mock_popen.call_args
    assert call_args[0][0] == ["./spinner.py"]
    assert call_args[1]['stdin'] == subprocess.PIPE
    assert call_args[1]['close_fds'] is True
    assert 'cwd' in call_args[1]
