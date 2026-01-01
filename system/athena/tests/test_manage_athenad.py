"""Tests for system/athena/manage_athenad.py - athenad process manager."""

from openpilot.system.athena.manage_athenad import ATHENA_MGR_PID_PARAM, main


class TestManageAthenadConstants:
  """Test manage_athenad module constants."""

  def test_athena_mgr_pid_param(self):
    """Test ATHENA_MGR_PID_PARAM constant."""
    assert ATHENA_MGR_PID_PARAM == "AthenadPid"


class TestManageAthenadMain:
  """Test manage_athenad main function."""

  def test_main_starts_athenad_process(self, mocker):
    """Test main starts athenad process and handles exit."""
    mock_params = mocker.patch('openpilot.system.athena.manage_athenad.Params')
    mock_params.return_value.get.return_value = "test_dongle_id"
    mock_params.return_value.remove = mocker.MagicMock()

    mock_build_metadata = mocker.patch('openpilot.system.athena.manage_athenad.get_build_metadata')
    mock_build_metadata.return_value.openpilot.version = "1.0.0"
    mock_build_metadata.return_value.openpilot.git_normalized_origin = "origin"
    mock_build_metadata.return_value.channel = "master"
    mock_build_metadata.return_value.openpilot.git_commit = "abc123"
    mock_build_metadata.return_value.openpilot.is_dirty = False

    mock_hardware = mocker.patch('openpilot.system.athena.manage_athenad.HARDWARE')
    mock_hardware.get_device_type.return_value = "tici"

    mocker.patch('openpilot.system.athena.manage_athenad.cloudlog')

    # Mock Process to simulate one iteration then exception
    mock_process = mocker.MagicMock()
    mock_process.exitcode = 0
    mock_process_class = mocker.patch('openpilot.system.athena.manage_athenad.Process')
    mock_process_class.return_value = mock_process

    # Make join raise a generic exception to exit the loop
    mock_process.join.side_effect = Exception("Test exit")

    mocker.patch('openpilot.system.athena.manage_athenad.time.sleep')

    # Run main and expect it to clean up
    main()

    # Verify process was started
    mock_process.start.assert_called_once()
    # Verify cleanup happened
    mock_params.return_value.remove.assert_called_once_with(ATHENA_MGR_PID_PARAM)

  def test_main_removes_param_on_exception(self, mocker):
    """Test main removes param on exception."""
    mock_params = mocker.patch('openpilot.system.athena.manage_athenad.Params')
    mock_params.return_value.get.return_value = "test_dongle_id"
    mock_params.return_value.remove = mocker.MagicMock()

    mock_build_metadata = mocker.patch('openpilot.system.athena.manage_athenad.get_build_metadata')
    mock_build_metadata.return_value.openpilot.version = "1.0.0"
    mock_build_metadata.return_value.openpilot.git_normalized_origin = "origin"
    mock_build_metadata.return_value.channel = "master"
    mock_build_metadata.return_value.openpilot.git_commit = "abc123"
    mock_build_metadata.return_value.openpilot.is_dirty = False

    mock_hardware = mocker.patch('openpilot.system.athena.manage_athenad.HARDWARE')
    mock_hardware.get_device_type.return_value = "tici"

    mocker.patch('openpilot.system.athena.manage_athenad.cloudlog')

    # Mock Process to raise exception immediately
    mock_process_class = mocker.patch('openpilot.system.athena.manage_athenad.Process')
    mock_process_class.side_effect = RuntimeError("Test exception")

    mocker.patch('openpilot.system.athena.manage_athenad.time.sleep')

    # Run main
    main()

    # Verify cleanup happened despite exception
    mock_params.return_value.remove.assert_called_once_with(ATHENA_MGR_PID_PARAM)
