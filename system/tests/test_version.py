"""Tests for system/version.py - version and build metadata utilities."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from openpilot.system.version import (
  OpenpilotMetadata, BuildMetadata,
  get_version, get_release_notes, is_prebuilt,
  build_metadata_from_dict, get_build_metadata,
  RELEASE_BRANCHES, TESTED_BRANCHES,
  training_version, terms_version,
)


class TestConstants(unittest.TestCase):
  """Test module constants."""

  def test_release_branches_not_empty(self):
    """Test RELEASE_BRANCHES contains expected branches."""
    self.assertIn('release-tizi', RELEASE_BRANCHES)
    self.assertIn('nightly', RELEASE_BRANCHES)

  def test_tested_branches_includes_release(self):
    """Test TESTED_BRANCHES includes all release branches."""
    for branch in RELEASE_BRANCHES:
      self.assertIn(branch, TESTED_BRANCHES)

  def test_tested_branches_includes_devel(self):
    """Test TESTED_BRANCHES includes devel branches."""
    self.assertIn('devel-staging', TESTED_BRANCHES)

  def test_training_version_format(self):
    """Test training_version is a valid semver string."""
    parts = training_version.split('.')
    self.assertEqual(len(parts), 3)
    for part in parts:
      self.assertTrue(part.isdigit())

  def test_terms_version_is_digit(self):
    """Test terms_version is a digit string."""
    self.assertTrue(terms_version.isdigit())


class TestOpenpilotMetadata(unittest.TestCase):
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
    self.assertEqual(meta.short_version, "0.9.7")

  def test_short_version_no_dash(self):
    """Test short_version when no dash in version."""
    meta = self._create_metadata(version="0.9.7")
    self.assertEqual(meta.short_version, "0.9.7")

  def test_comma_remote_true_for_commaai(self):
    """Test comma_remote is True for commaai origin."""
    meta = self._create_metadata(git_origin="https://github.com/commaai/openpilot.git")
    self.assertTrue(meta.comma_remote)

  def test_comma_remote_false_for_fork(self):
    """Test comma_remote is False for fork."""
    meta = self._create_metadata(git_origin="https://github.com/user/openpilot.git")
    self.assertFalse(meta.comma_remote)

  def test_comma_remote_ssh_format(self):
    """Test comma_remote works with SSH URL format."""
    meta = self._create_metadata(git_origin="git@github.com:commaai/openpilot.git")
    self.assertTrue(meta.comma_remote)

  def test_git_normalized_origin_https(self):
    """Test git_normalized_origin removes https prefix."""
    meta = self._create_metadata(git_origin="https://github.com/user/repo.git")
    self.assertEqual(meta.git_normalized_origin, "github.com/user/repo")

  def test_git_normalized_origin_ssh(self):
    """Test git_normalized_origin normalizes SSH format."""
    meta = self._create_metadata(git_origin="git@github.com:user/repo.git")
    self.assertEqual(meta.git_normalized_origin, "github.com/user/repo")


class TestBuildMetadata(unittest.TestCase):
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
    self.assertTrue(meta.tested_channel)

  def test_tested_channel_true_for_devel(self):
    """Test tested_channel is True for devel-staging."""
    meta = self._create_build_metadata(channel="devel-staging")
    self.assertTrue(meta.tested_channel)

  def test_tested_channel_false_for_custom(self):
    """Test tested_channel is False for custom branches."""
    meta = self._create_build_metadata(channel="my-feature-branch")
    self.assertFalse(meta.tested_channel)

  def test_release_channel_true(self):
    """Test release_channel is True for release branches."""
    meta = self._create_build_metadata(channel="release-tizi")
    self.assertTrue(meta.release_channel)

  def test_release_channel_false_for_devel(self):
    """Test release_channel is False for devel branches."""
    meta = self._create_build_metadata(channel="devel-staging")
    self.assertFalse(meta.release_channel)

  def test_canonical_format(self):
    """Test canonical property format."""
    meta = self._create_build_metadata()
    self.assertEqual(meta.canonical, "0.9.7-abc123def456-release")

  def test_ui_description_format(self):
    """Test ui_description property format."""
    meta = self._create_build_metadata(channel="release-tizi")
    self.assertEqual(meta.ui_description, "0.9.7 / abc123 / release-tizi")


class TestBuildMetadataFromDict(unittest.TestCase):
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
      }
    }
    meta = build_metadata_from_dict(data)
    self.assertEqual(meta.channel, "release-tizi")
    self.assertEqual(meta.openpilot.version, "0.9.7")
    self.assertEqual(meta.openpilot.git_commit, "abc123")
    self.assertFalse(meta.openpilot.is_dirty)

  def test_empty_dict_uses_defaults(self):
    """Test parsing an empty dict uses unknown defaults."""
    meta = build_metadata_from_dict({})
    self.assertEqual(meta.channel, "unknown")
    self.assertEqual(meta.openpilot.version, "unknown")
    self.assertEqual(meta.openpilot.git_commit, "unknown")

  def test_partial_dict(self):
    """Test parsing a partial dict."""
    data = {
      "channel": "nightly",
      "openpilot": {
        "version": "0.9.8",
      }
    }
    meta = build_metadata_from_dict(data)
    self.assertEqual(meta.channel, "nightly")
    self.assertEqual(meta.openpilot.version, "0.9.8")
    self.assertEqual(meta.openpilot.git_commit, "unknown")


class TestGetVersion(unittest.TestCase):
  """Test get_version function."""

  def test_get_version_returns_string(self):
    """Test get_version returns a version string."""
    from openpilot.common.basedir import BASEDIR
    version = get_version(BASEDIR)
    self.assertIsInstance(version, str)
    self.assertGreater(len(version), 0)

  def test_get_version_format(self):
    """Test version has expected format (contains dots)."""
    from openpilot.common.basedir import BASEDIR
    version = get_version(BASEDIR)
    self.assertIn('.', version)


class TestGetReleaseNotes(unittest.TestCase):
  """Test get_release_notes function."""

  def test_get_release_notes_returns_string(self):
    """Test get_release_notes returns content."""
    from openpilot.common.basedir import BASEDIR
    notes = get_release_notes(BASEDIR)
    self.assertIsInstance(notes, str)
    self.assertGreater(len(notes), 0)


class TestIsPrebuilt(unittest.TestCase):
  """Test is_prebuilt function."""

  def test_is_prebuilt_returns_bool(self):
    """Test is_prebuilt returns boolean."""
    from openpilot.common.basedir import BASEDIR
    result = is_prebuilt(BASEDIR)
    self.assertIsInstance(result, bool)

  def test_is_prebuilt_false_in_dev(self):
    """Test is_prebuilt is False in development environment."""
    from openpilot.common.basedir import BASEDIR
    # In a dev environment, there should be no 'prebuilt' file
    result = is_prebuilt(BASEDIR)
    self.assertFalse(result)


class TestGetBuildMetadata(unittest.TestCase):
  """Test get_build_metadata function."""

  def test_get_build_metadata_from_git(self):
    """Test get_build_metadata works from git repo."""
    from openpilot.common.basedir import BASEDIR
    meta = get_build_metadata(BASEDIR)
    self.assertIsInstance(meta, BuildMetadata)
    self.assertIsInstance(meta.openpilot, OpenpilotMetadata)

  def test_get_build_metadata_has_version(self):
    """Test metadata includes version."""
    from openpilot.common.basedir import BASEDIR
    meta = get_build_metadata(BASEDIR)
    self.assertIsNotNone(meta.openpilot.version)
    self.assertIn('.', meta.openpilot.version)

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
        }
      }
      build_json_path = Path(tmpdir) / "build.json"
      build_json_path.write_text(json.dumps(build_data))

      meta = get_build_metadata(tmpdir)
      self.assertEqual(meta.channel, "test-channel")
      self.assertEqual(meta.openpilot.version, "1.0.0")

  def test_get_build_metadata_raises_without_git_or_json(self):
    """Test get_build_metadata raises exception without git or build.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
      with self.assertRaises(Exception) as ctx:
        get_build_metadata(tmpdir)
      self.assertIn("invalid build metadata", str(ctx.exception))


if __name__ == '__main__':
  unittest.main()
