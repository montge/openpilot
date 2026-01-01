"""Tests for common/prefix.py - OpenpilotPrefix context manager."""

from openpilot.common.prefix import OpenpilotPrefix


class TestOpenpilotPrefixInit:
  """Test OpenpilotPrefix initialization."""

  def test_init_with_custom_prefix(self):
    """Test initialization with custom prefix."""
    prefix = OpenpilotPrefix(prefix="test_prefix", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    assert prefix.prefix == "test_prefix"

  def test_init_generates_uuid_prefix(self):
    """Test initialization generates UUID prefix when not specified."""
    prefix = OpenpilotPrefix(create_dirs_on_enter=False, clean_dirs_on_exit=False)
    assert len(prefix.prefix) == 15

  def test_init_uuid_is_hex(self):
    """Test generated prefix is valid hex."""
    prefix = OpenpilotPrefix(create_dirs_on_enter=False, clean_dirs_on_exit=False)
    # UUID hex is 0-9a-f
    for c in prefix.prefix:
      assert c in '0123456789abcdef'

  def test_init_default_create_dirs(self):
    """Test create_dirs_on_enter defaults to True."""
    prefix = OpenpilotPrefix.__new__(OpenpilotPrefix)
    prefix.__init__()
    assert prefix.create_dirs_on_enter is True

  def test_init_default_clean_dirs(self):
    """Test clean_dirs_on_exit defaults to True."""
    prefix = OpenpilotPrefix.__new__(OpenpilotPrefix)
    prefix.__init__()
    assert prefix.clean_dirs_on_exit is True

  def test_init_default_shared_cache(self):
    """Test shared_download_cache defaults to False."""
    prefix = OpenpilotPrefix.__new__(OpenpilotPrefix)
    prefix.__init__()
    assert prefix.shared_download_cache is False

  def test_init_custom_flags(self):
    """Test initialization with custom flags."""
    prefix = OpenpilotPrefix(prefix="test", create_dirs_on_enter=False, clean_dirs_on_exit=False, shared_download_cache=True)
    assert prefix.create_dirs_on_enter is False
    assert prefix.clean_dirs_on_exit is False
    assert prefix.shared_download_cache is True

  def test_msgq_path_set(self, mocker):
    """Test msgq_path is set correctly."""
    mock_shm_path = mocker.patch('openpilot.common.prefix.Paths.shm_path')
    mock_shm_path.return_value = "/dev/shm"
    prefix = OpenpilotPrefix(prefix="myprefix", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    assert prefix.msgq_path == "/dev/shm/msgq_myprefix"

  def test_different_prefixes_generate_different_paths(self):
    """Test different prefixes generate different msgq paths."""
    p1 = OpenpilotPrefix(prefix="prefix1", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    p2 = OpenpilotPrefix(prefix="prefix2", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    assert p1.msgq_path != p2.msgq_path

  def test_multiple_init_generate_unique_prefixes(self):
    """Test multiple initializations generate unique prefixes."""
    prefixes = set()
    for _ in range(10):
      p = OpenpilotPrefix(create_dirs_on_enter=False, clean_dirs_on_exit=False)
      prefixes.add(p.prefix)
    assert len(prefixes) == 10


class TestOpenpilotPrefixCreateDirs:
  """Test OpenpilotPrefix.create_dirs method."""

  def test_creates_msgq_dir(self, mocker):
    """Test create_dirs creates msgq directory."""
    mock_shm = mocker.patch('openpilot.common.prefix.Paths.shm_path')
    mock_log = mocker.patch('openpilot.common.prefix.Paths.log_root')
    mock_mkdir = mocker.patch('openpilot.common.prefix.os.mkdir')
    mocker.patch('openpilot.common.prefix.os.makedirs')
    mock_shm.return_value = "/dev/shm"
    mock_log.return_value = "/data/realdata"

    prefix = OpenpilotPrefix(prefix="test", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    prefix.create_dirs()

    mock_mkdir.assert_called_once_with("/dev/shm/msgq_test")

  def test_creates_log_root(self, mocker):
    """Test create_dirs creates log root directory."""
    mock_shm = mocker.patch('openpilot.common.prefix.Paths.shm_path')
    mock_log = mocker.patch('openpilot.common.prefix.Paths.log_root')
    mocker.patch('openpilot.common.prefix.os.mkdir')
    mock_makedirs = mocker.patch('openpilot.common.prefix.os.makedirs')
    mock_shm.return_value = "/dev/shm"
    mock_log.return_value = "/data/realdata"

    prefix = OpenpilotPrefix(prefix="test", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    prefix.create_dirs()

    mock_makedirs.assert_called_once_with("/data/realdata", exist_ok=True)

  def test_handles_existing_msgq_dir(self, mocker):
    """Test create_dirs handles FileExistsError for msgq dir."""
    mock_shm = mocker.patch('openpilot.common.prefix.Paths.shm_path')
    mock_log = mocker.patch('openpilot.common.prefix.Paths.log_root')
    mock_mkdir = mocker.patch('openpilot.common.prefix.os.mkdir')
    mock_makedirs = mocker.patch('openpilot.common.prefix.os.makedirs')
    mock_shm.return_value = "/dev/shm"
    mock_log.return_value = "/data/realdata"
    mock_mkdir.side_effect = FileExistsError()

    prefix = OpenpilotPrefix(prefix="test", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    # Should not raise
    prefix.create_dirs()
    mock_makedirs.assert_called_once()


class TestOpenpilotPrefixContextManager:
  """Test OpenpilotPrefix context manager behavior."""

  def test_enter_sets_shared_download_cache(self, mocker):
    """Test __enter__ sets COMMA_CACHE when shared_download_cache=True."""
    import os

    mocker.patch('openpilot.common.prefix.Paths.shm_path', return_value="/dev/shm")
    mocker.patch('openpilot.common.prefix.Paths.log_root', return_value="/tmp/log")
    mocker.patch('openpilot.common.prefix.os.mkdir')
    mocker.patch('openpilot.common.prefix.os.makedirs')

    # Ensure COMMA_CACHE is not set
    if "COMMA_CACHE" in os.environ:
      del os.environ["COMMA_CACHE"]

    prefix = OpenpilotPrefix(prefix="test", create_dirs_on_enter=False, clean_dirs_on_exit=False, shared_download_cache=True)
    prefix.__enter__()

    try:
      assert "COMMA_CACHE" in os.environ
    finally:
      prefix.__exit__(None, None, None)
      # Clean up
      if "COMMA_CACHE" in os.environ:
        del os.environ["COMMA_CACHE"]

  def test_exit_restores_original_prefix(self, mocker):
    """Test __exit__ restores original OPENPILOT_PREFIX."""
    import os

    mocker.patch('openpilot.common.prefix.Paths.shm_path', return_value="/dev/shm")
    mocker.patch('openpilot.common.prefix.Paths.log_root', return_value="/tmp/log")
    mocker.patch('openpilot.common.prefix.os.mkdir')
    mocker.patch('openpilot.common.prefix.os.makedirs')
    mocker.patch('openpilot.common.prefix.Params')
    mocker.patch('openpilot.common.prefix.os.path.exists', return_value=False)
    mocker.patch('openpilot.common.prefix.shutil.rmtree')

    # Save current state
    saved_prefix = os.environ.get("OPENPILOT_PREFIX")

    # Set original prefix
    original = "original_prefix_value"
    os.environ["OPENPILOT_PREFIX"] = original

    try:
      prefix = OpenpilotPrefix(prefix="test", create_dirs_on_enter=False, clean_dirs_on_exit=False)
      prefix.__enter__()
      assert os.environ["OPENPILOT_PREFIX"] == "test"

      prefix.__exit__(None, None, None)
      assert os.environ["OPENPILOT_PREFIX"] == original
    finally:
      # Restore original state
      if saved_prefix is not None:
        os.environ["OPENPILOT_PREFIX"] = saved_prefix
      elif "OPENPILOT_PREFIX" in os.environ:
        del os.environ["OPENPILOT_PREFIX"]

  def test_enter_skips_create_dirs_when_disabled(self, mocker):
    """Test __enter__ skips create_dirs when create_dirs_on_enter=False."""
    mocker.patch('openpilot.common.prefix.Paths.shm_path', return_value="/dev/shm")

    mock_mkdir = mocker.patch('openpilot.common.prefix.os.mkdir')
    mock_makedirs = mocker.patch('openpilot.common.prefix.os.makedirs')

    prefix = OpenpilotPrefix(prefix="test", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    prefix.__enter__()

    try:
      mock_mkdir.assert_not_called()
      mock_makedirs.assert_not_called()
    finally:
      prefix.__exit__(None, None, None)

  def test_exit_handles_missing_prefix_env(self, mocker):
    """Test __exit__ handles KeyError when OPENPILOT_PREFIX not in environ."""
    import os

    mocker.patch('openpilot.common.prefix.Paths.shm_path', return_value="/dev/shm")

    # Save the current prefix to restore later
    saved_prefix = os.environ.get("OPENPILOT_PREFIX")

    try:
      # Remove OPENPILOT_PREFIX to trigger the KeyError path
      if "OPENPILOT_PREFIX" in os.environ:
        del os.environ["OPENPILOT_PREFIX"]

      prefix = OpenpilotPrefix(prefix="test", create_dirs_on_enter=False, clean_dirs_on_exit=False)
      # Manually set original_prefix to None to simulate no original prefix
      prefix.original_prefix = None

      # Should not raise KeyError - lines 38-39 handle this
      prefix.__exit__(None, None, None)
    finally:
      # Restore the prefix for the fixture
      if saved_prefix is not None:
        os.environ["OPENPILOT_PREFIX"] = saved_prefix


class TestOpenpilotPrefixCleanDirs:
  """Test OpenpilotPrefix.clean_dirs method."""

  def test_clean_dirs_removes_symlink_and_realpath(self, mocker):
    """Test clean_dirs removes symlink and its target."""
    mocker.patch('openpilot.common.prefix.Paths.shm_path', return_value="/dev/shm")
    mocker.patch('openpilot.common.prefix.Paths.log_root', return_value="/tmp/log")
    mocker.patch('openpilot.common.prefix.Paths.download_cache_root', return_value="/tmp/cache")
    mocker.patch('openpilot.common.prefix.Paths.comma_home', return_value="/tmp/.comma")

    mock_params = mocker.patch('openpilot.common.prefix.Params')
    mock_params.return_value.get_param_path.return_value = "/tmp/params"

    mock_exists = mocker.patch('openpilot.common.prefix.os.path.exists')
    mock_exists.return_value = True

    mock_realpath = mocker.patch('openpilot.common.prefix.os.path.realpath')
    mock_realpath.return_value = "/tmp/params_real"

    mock_rmtree = mocker.patch('openpilot.common.prefix.shutil.rmtree')
    mock_remove = mocker.patch('openpilot.common.prefix.os.remove')

    prefix = OpenpilotPrefix(prefix="test", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    prefix.clean_dirs()

    mock_rmtree.assert_any_call("/tmp/params_real", ignore_errors=True)
    mock_remove.assert_called_once_with("/tmp/params")

  def test_clean_dirs_skips_symlink_when_not_exists(self, mocker):
    """Test clean_dirs skips symlink removal when path doesn't exist."""
    mocker.patch('openpilot.common.prefix.Paths.shm_path', return_value="/dev/shm")
    mocker.patch('openpilot.common.prefix.Paths.log_root', return_value="/tmp/log")
    mocker.patch('openpilot.common.prefix.Paths.download_cache_root', return_value="/tmp/cache")
    mocker.patch('openpilot.common.prefix.Paths.comma_home', return_value="/tmp/.comma")

    mock_params = mocker.patch('openpilot.common.prefix.Params')
    mock_params.return_value.get_param_path.return_value = "/tmp/params"

    mock_exists = mocker.patch('openpilot.common.prefix.os.path.exists')
    mock_exists.return_value = False  # Symlink doesn't exist

    mocker.patch('openpilot.common.prefix.shutil.rmtree')
    mock_remove = mocker.patch('openpilot.common.prefix.os.remove')

    prefix = OpenpilotPrefix(prefix="test", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    prefix.clean_dirs()

    # os.remove should not be called since symlink doesn't exist
    mock_remove.assert_not_called()

  def test_clean_dirs_removes_log_root_on_pc(self, mocker):
    """Test clean_dirs removes log_root when PC=True."""
    mocker.patch('openpilot.common.prefix.Paths.shm_path', return_value="/dev/shm")
    mocker.patch('openpilot.common.prefix.Paths.log_root', return_value="/tmp/log")
    mocker.patch('openpilot.common.prefix.Paths.download_cache_root', return_value="/tmp/cache")
    mocker.patch('openpilot.common.prefix.Paths.comma_home', return_value="/tmp/.comma")
    mocker.patch('openpilot.common.prefix.PC', True)

    mock_params = mocker.patch('openpilot.common.prefix.Params')
    mock_params.return_value.get_param_path.return_value = "/tmp/params"

    mock_exists = mocker.patch('openpilot.common.prefix.os.path.exists')
    mock_exists.return_value = False

    mock_rmtree = mocker.patch('openpilot.common.prefix.shutil.rmtree')

    prefix = OpenpilotPrefix(prefix="test", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    prefix.clean_dirs()

    mock_rmtree.assert_any_call("/tmp/log", ignore_errors=True)

  def test_clean_dirs_skips_log_root_on_device(self, mocker):
    """Test clean_dirs skips log_root removal when PC=False."""
    mocker.patch('openpilot.common.prefix.Paths.shm_path', return_value="/dev/shm")
    mocker.patch('openpilot.common.prefix.Paths.log_root', return_value="/data/realdata")
    mocker.patch('openpilot.common.prefix.Paths.download_cache_root', return_value="/tmp/cache")
    mocker.patch('openpilot.common.prefix.Paths.comma_home', return_value="/tmp/.comma")
    mocker.patch('openpilot.common.prefix.PC', False)

    mock_params = mocker.patch('openpilot.common.prefix.Params')
    mock_params.return_value.get_param_path.return_value = "/tmp/params"

    mock_exists = mocker.patch('openpilot.common.prefix.os.path.exists')
    mock_exists.return_value = False

    mock_rmtree = mocker.patch('openpilot.common.prefix.shutil.rmtree')

    prefix = OpenpilotPrefix(prefix="test", create_dirs_on_enter=False, clean_dirs_on_exit=False)
    prefix.clean_dirs()

    # log_root should NOT be removed on device
    rmtree_calls = [str(call) for call in mock_rmtree.call_args_list]
    assert not any("/data/realdata" in call for call in rmtree_calls)

  def test_clean_dirs_skips_cache_when_comma_cache_set(self, mocker):
    """Test clean_dirs skips download_cache_root when COMMA_CACHE is set."""
    import os

    mocker.patch('openpilot.common.prefix.Paths.shm_path', return_value="/dev/shm")
    mocker.patch('openpilot.common.prefix.Paths.log_root', return_value="/tmp/log")
    mocker.patch('openpilot.common.prefix.Paths.download_cache_root', return_value="/tmp/cache")
    mocker.patch('openpilot.common.prefix.Paths.comma_home', return_value="/tmp/.comma")
    mocker.patch('openpilot.common.prefix.PC', False)

    mock_params = mocker.patch('openpilot.common.prefix.Params')
    mock_params.return_value.get_param_path.return_value = "/tmp/params"

    mock_exists = mocker.patch('openpilot.common.prefix.os.path.exists')
    mock_exists.return_value = False

    mock_rmtree = mocker.patch('openpilot.common.prefix.shutil.rmtree')

    # Set COMMA_CACHE
    os.environ["COMMA_CACHE"] = "/shared/cache"

    try:
      prefix = OpenpilotPrefix(prefix="test", create_dirs_on_enter=False, clean_dirs_on_exit=False)
      prefix.clean_dirs()

      # download_cache_root should NOT be removed when COMMA_CACHE is set
      rmtree_calls = [str(call) for call in mock_rmtree.call_args_list]
      assert not any("/tmp/cache" in call for call in rmtree_calls)
    finally:
      del os.environ["COMMA_CACHE"]
