"""Tests for system/manager/helpers.py - manager helper functions.

write_onroad_params was removed upstream (single IsOffroad param); the IsOffroad
param is now managed directly by manager_thread and covered in
test_manager_integration.py. This file covers the remaining helpers.
"""

from openpilot.system.manager.helpers import save_bootlog


class TestSaveBootlog:
  """Test save_bootlog function."""

  def test_save_bootlog_copies_params_and_spawns_thread(self, mocker):
    """Test save_bootlog snapshots params and starts a daemon bootlog thread."""
    mocker.patch('openpilot.system.manager.helpers.tempfile.mkdtemp', return_value='/tmp/bootlog_params')
    mock_copytree = mocker.patch('openpilot.system.manager.helpers.shutil.copytree')
    mock_thread = mocker.patch('openpilot.system.manager.helpers.threading.Thread')

    save_bootlog()

    # params dir is copied into the temp dir
    mock_copytree.assert_called_once()
    dst = mock_copytree.call_args[0][1]
    assert dst.startswith('/tmp/bootlog_params')

    # bootlog runs on a daemon thread pointed at the params copy
    mock_thread.assert_called_once()
    assert mock_thread.call_args.kwargs['args'] == ('/tmp/bootlog_params',)
    assert mock_thread.return_value.daemon is True
    mock_thread.return_value.start.assert_called_once()

  def test_save_bootlog_thread_runs_bootlog_and_cleans_up(self, mocker):
    """Test the spawned thread target invokes ./bootlog with PARAMS_COPY_PATH and removes the temp dir."""
    mocker.patch('openpilot.system.manager.helpers.tempfile.mkdtemp', return_value='/tmp/bootlog_params')
    mocker.patch('openpilot.system.manager.helpers.shutil.copytree')
    mock_rmtree = mocker.patch('openpilot.system.manager.helpers.shutil.rmtree')
    mock_call = mocker.patch('openpilot.system.manager.helpers.subprocess.call')
    mock_thread = mocker.patch('openpilot.system.manager.helpers.threading.Thread')

    save_bootlog()

    # run the thread target synchronously
    fn = mock_thread.call_args.kwargs['target']
    args = mock_thread.call_args.kwargs['args']
    fn(*args)

    mock_call.assert_called_once()
    assert mock_call.call_args[0][0] == "./bootlog"
    assert mock_call.call_args.kwargs['env']['PARAMS_COPY_PATH'] == '/tmp/bootlog_params'
    mock_rmtree.assert_called_once_with('/tmp/bootlog_params')
