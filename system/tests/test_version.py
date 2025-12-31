"""Tests for system/version.py - version and build metadata utilities."""

import json
import tempfile
from pathlib import Path

import pytest

from openpilot.system.version import (
  OpenpilotMetadata,
  BuildMetadata,
  get_version,
  get_release_notes,
  is_prebuilt,
  is_dirty,
  build_metadata_from_dict,
  get_build_metadata,
  RELEASE_BRANCHES,
  TESTED_BRANCHES,
  training_version,
  terms_version,
)


class TestConstants:
  """Test module constants."""

  def test_release_branches_not_empty(self):
    """Test RELEASE_BRANCHES contains expected branches."""
    assert 'release-tizi' in RELEASE_BRANCHES
    assert 'nightly' in RELEASE_BRANCHES

  def test_tested_branches_includes_release(self):
    """Test TESTED_BRANCHES includes all release branches."""
    for branch in RELEASE_BRANCHES:
      assert branch in TESTED_BRANCHES

  def test_tested_branches_includes_devel(self):
    """Test TESTED_BRANCHES includes devel branches."""
    assert 'devel-staging' in TESTED_BRANCHES

  def test_training_version_format(self):
    """Test training_version is a valid semver string."""
    parts = training_version.split('.')
    assert len(parts) == 3
    for part in parts:
      assert part.isdigit()

  def test_terms_version_is_digit(self):
    """Test terms_version is a digit string."""
    assert terms_version.isdigit()


class TestOpenpilotMetadata:
  """Test OpenpilotMetadata dataclass."""

  def _create_metadata(self, version="0.9.7", git_origin="https://github.com/commaai/openpilot.git"):
    return OpenpilotMetadata(
      version=version,
      release_notes="Test notes",
      git_commit="abc123def456",
      git_origin=git_origin,
      git_commit_date="2024-01-15",
      build_style="release",
      is_dirty=False,
    )

  def test_short_version(self):
    """Test short_version property extracts version before dash."""
    meta = self._create_metadata(version="0.9.7-release")
    assert meta.short_version == "0.9.7"

  def test_short_version_no_dash(self):
    """Test short_version when no dash in version."""
    meta = self._create_metadata(version="0.9.7")
    assert meta.short_version == "0.9.7"

  def test_comma_remote_true_for_commaai(self):
    """Test comma_remote is True for commaai origin."""
    meta = self._create_metadata(git_origin="https://github.com/commaai/openpilot.git")
    assert meta.comma_remote is True

  def test_comma_remote_false_for_fork(self):
    """Test comma_remote is False for fork."""
    meta = self._create_metadata(git_origin="https://github.com/user/openpilot.git")
    assert meta.comma_remote is False

  def test_comma_remote_ssh_format(self):
    """Test comma_remote works with SSH URL format."""
    meta = self._create_metadata(git_origin="git@github.com:commaai/openpilot.git")
    assert meta.comma_remote is True

  def test_git_normalized_origin_https(self):
    """Test git_normalized_origin removes https prefix."""
    meta = self._create_metadata(git_origin="https://github.com/user/repo.git")
    assert meta.git_normalized_origin == "github.com/user/repo"

  def test_git_normalized_origin_ssh(self):
    """Test git_normalized_origin normalizes SSH format."""
    meta = self._create_metadata(git_origin="git@github.com:user/repo.git")
    assert meta.git_normalized_origin == "github.com/user/repo"


class TestBuildMetadata:
  """Test BuildMetadata dataclass."""

  def _create_build_metadata(self, channel="release-tizi"):
    openpilot_meta = OpenpilotMetadata(
      version="0.9.7",
      release_notes="Notes",
      git_commit="abc123def456",
      git_origin="https://github.com/commaai/openpilot.git",
      git_commit_date="2024-01-15",
      build_style="release",
      is_dirty=False,
    )
    return BuildMetadata(channel=channel, openpilot=openpilot_meta)

  def test_tested_channel_true_for_release(self):
    """Test tested_channel is True for release branches."""
    meta = self._create_build_metadata(channel="release-tizi")
    assert meta.tested_channel is True

  def test_tested_channel_true_for_devel(self):
    """Test tested_channel is True for devel-staging."""
    meta = self._create_build_metadata(channel="devel-staging")
    assert meta.tested_channel is True

  def test_tested_channel_false_for_custom(self):
    """Test tested_channel is False for custom branches."""
    meta = self._create_build_metadata(channel="my-feature-branch")
    assert meta.tested_channel is False

  def test_release_channel_true(self):
    """Test release_channel is True for release branches."""
    meta = self._create_build_metadata(channel="release-tizi")
    assert meta.release_channel is True

  def test_release_channel_false_for_devel(self):
    """Test release_channel is False for devel branches."""
    meta = self._create_build_metadata(channel="devel-staging")
    assert meta.release_channel is False

  def test_canonical_format(self):
    """Test canonical property format."""
    meta = self._create_build_metadata()
    assert meta.canonical == "0.9.7-abc123def456-release"

  def test_ui_description_format(self):
    """Test ui_description property format."""
    meta = self._create_build_metadata(channel="release-tizi")
    assert meta.ui_description == "0.9.7 / abc123 / release-tizi"


class TestBuildMetadataFromDict:
  """Test build_metadata_from_dict function."""

  def test_full_dict(self):
    """Test parsing a complete metadata dict."""
    data = {
      "channel": "release-tizi",
      "openpilot": {
        "version": "0.9.7",
        "release_notes": "Test notes",
        "git_commit": "abc123",
        "git_origin": "https://github.com/commaai/openpilot.git",
        "git_commit_date": "2024-01-15",
        "build_style": "release",
      },
    }
    meta = build_metadata_from_dict(data)
    assert meta.channel == "release-tizi"
    assert meta.openpilot.version == "0.9.7"
    assert meta.openpilot.git_commit == "abc123"
    assert meta.openpilot.is_dirty is False

  def test_empty_dict_uses_defaults(self):
    """Test parsing an empty dict uses unknown defaults."""
    meta = build_metadata_from_dict({})
    assert meta.channel == "unknown"
    assert meta.openpilot.version == "unknown"
    assert meta.openpilot.git_commit == "unknown"

  def test_partial_dict(self):
    """Test parsing a partial dict."""
    data = {
      "channel": "nightly",
      "openpilot": {
        "version": "0.9.8",
      },
    }
    meta = build_metadata_from_dict(data)
    assert meta.channel == "nightly"
    assert meta.openpilot.version == "0.9.8"
    assert meta.openpilot.git_commit == "unknown"


class TestGetVersion:
  """Test get_version function."""

  def test_get_version_returns_string(self):
    """Test get_version returns a version string."""
    from openpilot.common.basedir import BASEDIR

    version = get_version(BASEDIR)
    assert isinstance(version, str)
    assert len(version) > 0

  def test_get_version_format(self):
    """Test version has expected format (contains dots)."""
    from openpilot.common.basedir import BASEDIR

    version = get_version(BASEDIR)
    assert '.' in version


class TestGetReleaseNotes:
  """Test get_release_notes function."""

  def test_get_release_notes_returns_string(self):
    """Test get_release_notes returns content."""
    from openpilot.common.basedir import BASEDIR

    notes = get_release_notes(BASEDIR)
    assert isinstance(notes, str)
    assert len(notes) > 0


class TestIsPrebuilt:
  """Test is_prebuilt function."""

  def test_is_prebuilt_returns_bool(self):
    """Test is_prebuilt returns boolean."""
    from openpilot.common.basedir import BASEDIR

    result = is_prebuilt(BASEDIR)
    assert isinstance(result, bool)

  def test_is_prebuilt_false_in_dev(self):
    """Test is_prebuilt is False in development environment."""
    from openpilot.common.basedir import BASEDIR

    # In a dev environment, there should be no 'prebuilt' file
    result = is_prebuilt(BASEDIR)
    assert result is False


class TestIsDirty:
  """Test is_dirty function."""

  def test_is_dirty_returns_bool(self):
    """Test is_dirty returns boolean."""
    from openpilot.common.basedir import BASEDIR

    # Clear cache to get fresh result
    is_dirty.cache_clear()
    result = is_dirty(BASEDIR)
    assert isinstance(result, bool)

  def test_is_dirty_true_when_no_origin(self, mocker):
    """Test is_dirty returns True when no origin."""
    is_dirty.cache_clear()
    mocker.patch('openpilot.system.version.get_origin', return_value='')
    mocker.patch('openpilot.system.version.get_short_branch', return_value='main')

    result = is_dirty()
    assert result is True

  def test_is_dirty_true_when_no_short_branch(self, mocker):
    """Test is_dirty returns True when no short branch."""
    is_dirty.cache_clear()
    mocker.patch('openpilot.system.version.get_origin', return_value='origin')
    mocker.patch('openpilot.system.version.get_short_branch', return_value='')

    result = is_dirty()
    assert result is True

  def test_is_dirty_true_when_no_branch(self, mocker):
    """Test is_dirty returns True when get_branch returns empty."""
    is_dirty.cache_clear()
    mocker.patch('openpilot.system.version.get_origin', return_value='origin')
    mocker.patch('openpilot.system.version.get_short_branch', return_value='main')
    mocker.patch('openpilot.system.version.is_prebuilt', return_value=False)
    mocker.patch('openpilot.system.version.get_branch', return_value='')

    result = is_dirty()
    assert result is True

  def test_is_dirty_handles_subprocess_error(self, mocker):
    """Test is_dirty handles subprocess errors gracefully."""
    import subprocess

    is_dirty.cache_clear()
    mocker.patch('openpilot.system.version.get_origin', return_value='origin')
    mocker.patch('openpilot.system.version.get_short_branch', return_value='main')
    mocker.patch('openpilot.system.version.is_prebuilt', return_value=False)
    mocker.patch('openpilot.system.version.get_branch', return_value='main')
    mocker.patch('subprocess.check_call')  # Mock update-index
    mocker.patch('subprocess.call', side_effect=subprocess.CalledProcessError(1, 'git'))

    result = is_dirty()
    assert result is True

  def test_is_dirty_skips_git_checks_when_prebuilt(self, mocker):
    """Test is_dirty skips git checks when is_prebuilt returns True."""
    is_dirty.cache_clear()
    mocker.patch('openpilot.system.version.get_origin', return_value='origin')
    mocker.patch('openpilot.system.version.get_short_branch', return_value='main')
    mocker.patch('openpilot.system.version.is_prebuilt', return_value=True)

    # These should NOT be called when is_prebuilt is True
    mock_check_call = mocker.patch('subprocess.check_call')
    mock_call = mocker.patch('subprocess.call')

    result = is_dirty()

    # Prebuilt should return dirty=False without calling git commands
    assert result is False
    mock_check_call.assert_not_called()
    mock_call.assert_not_called()


class TestGetBuildMetadata:
  """Test get_build_metadata function."""

  def test_get_build_metadata_from_git(self):
    """Test get_build_metadata works from git repo."""
    from openpilot.common.basedir import BASEDIR

    meta = get_build_metadata(BASEDIR)
    assert isinstance(meta, BuildMetadata)
    assert isinstance(meta.openpilot, OpenpilotMetadata)

  def test_get_build_metadata_has_version(self):
    """Test metadata includes version."""
    from openpilot.common.basedir import BASEDIR

    meta = get_build_metadata(BASEDIR)
    assert meta.openpilot.version is not None
    assert '.' in meta.openpilot.version

  def test_get_build_metadata_from_json(self):
    """Test get_build_metadata reads from build.json if present."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create build.json
      build_data = {
        "channel": "test-channel",
        "openpilot": {
          "version": "1.0.0",
          "release_notes": "Test",
          "git_commit": "abc123",
          "git_origin": "https://test.com/repo.git",
          "git_commit_date": "2024-01-01",
          "build_style": "test",
        },
      }
      build_json_path = Path(tmpdir) / "build.json"
      build_json_path.write_text(json.dumps(build_data))

      meta = get_build_metadata(tmpdir)
      assert meta.channel == "test-channel"
      assert meta.openpilot.version == "1.0.0"

  def test_get_build_metadata_raises_without_git_or_json(self):
    """Test get_build_metadata raises exception without git or build.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
      with pytest.raises(Exception) as exc_info:
        get_build_metadata(tmpdir)
      assert "invalid build metadata" in str(exc_info.value)
