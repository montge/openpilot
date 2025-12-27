"""Tests for common/git.py - Git utility functions."""
import unittest
from unittest.mock import patch, MagicMock
import subprocess

from openpilot.common.git import (
  get_commit, get_commit_date, get_short_branch,
  get_branch, get_origin, get_normalized_origin,
)


class TestGetCommit(unittest.TestCase):
  """Test get_commit function."""

  @patch('openpilot.common.git.run_cmd_default')
  def test_get_commit_returns_hash(self, mock_run):
    """Test get_commit returns commit hash."""
    get_commit.cache_clear()  # Clear cache for test
    mock_run.return_value = "abc123def456"

    result = get_commit()

    mock_run.assert_called_once()
    self.assertEqual(result, "abc123def456")

  @patch('openpilot.common.git.run_cmd_default')
  def test_get_commit_with_cwd(self, mock_run):
    """Test get_commit with cwd parameter."""
    get_commit.cache_clear()
    mock_run.return_value = "xyz789"

    result = get_commit(cwd="/some/path")

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    self.assertEqual(call_args[1].get('cwd'), "/some/path")

  @patch('openpilot.common.git.run_cmd_default')
  def test_get_commit_with_branch(self, mock_run):
    """Test get_commit with branch parameter."""
    get_commit.cache_clear()
    mock_run.return_value = "branch123"

    result = get_commit(branch="main")

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    self.assertIn("main", call_args[0][0])


class TestGetCommitDate(unittest.TestCase):
  """Test get_commit_date function."""

  @patch('openpilot.common.git.run_cmd_default')
  def test_get_commit_date_returns_date(self, mock_run):
    """Test get_commit_date returns date string."""
    get_commit_date.cache_clear()
    mock_run.return_value = "'1234567890 2023-01-01 12:00:00 +0000'"

    result = get_commit_date()

    mock_run.assert_called_once()
    self.assertIn("1234567890", result)

  @patch('openpilot.common.git.run_cmd_default')
  def test_get_commit_date_with_commit(self, mock_run):
    """Test get_commit_date with commit parameter."""
    get_commit_date.cache_clear()
    mock_run.return_value = "date_output"

    result = get_commit_date(commit="abc123")

    call_args = mock_run.call_args
    self.assertIn("abc123", call_args[0][0])


class TestGetShortBranch(unittest.TestCase):
  """Test get_short_branch function."""

  @patch('openpilot.common.git.run_cmd_default')
  def test_get_short_branch_returns_name(self, mock_run):
    """Test get_short_branch returns branch name."""
    get_short_branch.cache_clear()
    mock_run.return_value = "develop"

    result = get_short_branch()

    self.assertEqual(result, "develop")

  @patch('openpilot.common.git.run_cmd_default')
  def test_get_short_branch_with_cwd(self, mock_run):
    """Test get_short_branch with cwd parameter."""
    get_short_branch.cache_clear()
    mock_run.return_value = "feature-branch"

    result = get_short_branch(cwd="/repo")

    call_args = mock_run.call_args
    self.assertEqual(call_args[1].get('cwd'), "/repo")


class TestGetBranch(unittest.TestCase):
  """Test get_branch function."""

  @patch('openpilot.common.git.run_cmd_default')
  def test_get_branch_returns_full_name(self, mock_run):
    """Test get_branch returns full branch name."""
    get_branch.cache_clear()
    mock_run.return_value = "origin/develop"

    result = get_branch()

    self.assertEqual(result, "origin/develop")


class TestGetOrigin(unittest.TestCase):
  """Test get_origin function."""

  @patch('openpilot.common.git.run_cmd')
  def test_get_origin_returns_url(self, mock_run):
    """Test get_origin returns origin URL."""
    get_origin.cache_clear()
    mock_run.side_effect = [
      "main",  # branch name
      "origin",  # tracking remote
      "https://github.com/user/repo.git",  # origin url
    ]

    result = get_origin()

    self.assertEqual(result, "https://github.com/user/repo.git")

  @patch('openpilot.common.git.run_cmd')
  @patch('openpilot.common.git.run_cmd_default')
  def test_get_origin_fallback_on_error(self, mock_default, mock_run):
    """Test get_origin fallback when not on a branch."""
    get_origin.cache_clear()
    mock_run.side_effect = subprocess.CalledProcessError(1, "git")
    mock_default.return_value = "git@github.com:user/repo.git"

    result = get_origin()

    self.assertEqual(result, "git@github.com:user/repo.git")


class TestGetNormalizedOrigin(unittest.TestCase):
  """Test get_normalized_origin function."""

  @patch('openpilot.common.git.get_origin')
  def test_normalize_https_url(self, mock_origin):
    """Test normalizing HTTPS URL."""
    get_normalized_origin.cache_clear()
    mock_origin.return_value = "https://github.com/user/repo.git"

    result = get_normalized_origin()

    self.assertEqual(result, "github.com/user/repo")

  @patch('openpilot.common.git.get_origin')
  def test_normalize_ssh_url(self, mock_origin):
    """Test normalizing SSH URL."""
    get_normalized_origin.cache_clear()
    mock_origin.return_value = "git@github.com:user/repo.git"

    result = get_normalized_origin()

    self.assertEqual(result, "github.com/user/repo")

  @patch('openpilot.common.git.get_origin')
  def test_normalize_removes_git_extension(self, mock_origin):
    """Test .git extension is removed."""
    get_normalized_origin.cache_clear()
    mock_origin.return_value = "https://github.com/user/repo.git"

    result = get_normalized_origin()

    self.assertNotIn(".git", result)


if __name__ == '__main__':
  unittest.main()
