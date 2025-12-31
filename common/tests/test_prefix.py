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
