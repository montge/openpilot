"""Tests for common/git.py - Git utility functions."""

import subprocess


from openpilot.common.git import (
  get_commit,
  get_commit_date,
  get_short_branch,
  get_branch,
  get_origin,
  get_normalized_origin,
)


class TestGetCommit:
  """Test get_commit function."""

  def test_get_commit_returns_hash(self, mocker):
    """Test get_commit returns commit hash."""
    get_commit.cache_clear()  # Clear cache for test
    mock_run = mocker.patch('openpilot.common.git.run_cmd_default')
    mock_run.return_value = "abc123def456"

    result = get_commit()

    mock_run.assert_called_once()
    assert result == "abc123def456"

  def test_get_commit_with_cwd(self, mocker):
    """Test get_commit with cwd parameter."""
    get_commit.cache_clear()
    mock_run = mocker.patch('openpilot.common.git.run_cmd_default')
    mock_run.return_value = "xyz789"

    get_commit(cwd="/some/path")

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args[1].get('cwd') == "/some/path"

  def test_get_commit_with_branch(self, mocker):
    """Test get_commit with branch parameter."""
    get_commit.cache_clear()
    mock_run = mocker.patch('openpilot.common.git.run_cmd_default')
    mock_run.return_value = "branch123"

    get_commit(branch="main")

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert "main" in call_args[0][0]


class TestGetCommitDate:
  """Test get_commit_date function."""

  def test_get_commit_date_returns_date(self, mocker):
    """Test get_commit_date returns date string."""
    get_commit_date.cache_clear()
    mock_run = mocker.patch('openpilot.common.git.run_cmd_default')
    mock_run.return_value = "'1234567890 2023-01-01 12:00:00 +0000'"

    result = get_commit_date()

    mock_run.assert_called_once()
    assert "1234567890" in result

  def test_get_commit_date_with_commit(self, mocker):
    """Test get_commit_date with commit parameter."""
    get_commit_date.cache_clear()
    mock_run = mocker.patch('openpilot.common.git.run_cmd_default')
    mock_run.return_value = "date_output"

    get_commit_date(commit="abc123")

    call_args = mock_run.call_args
    assert "abc123" in call_args[0][0]


class TestGetShortBranch:
  """Test get_short_branch function."""

  def test_get_short_branch_returns_name(self, mocker):
    """Test get_short_branch returns branch name."""
    get_short_branch.cache_clear()
    mock_run = mocker.patch('openpilot.common.git.run_cmd_default')
    mock_run.return_value = "develop"

    result = get_short_branch()

    assert result == "develop"

  def test_get_short_branch_with_cwd(self, mocker):
    """Test get_short_branch with cwd parameter."""
    get_short_branch.cache_clear()
    mock_run = mocker.patch('openpilot.common.git.run_cmd_default')
    mock_run.return_value = "feature-branch"

    get_short_branch(cwd="/repo")

    call_args = mock_run.call_args
    assert call_args[1].get('cwd') == "/repo"


class TestGetBranch:
  """Test get_branch function."""

  def test_get_branch_returns_full_name(self, mocker):
    """Test get_branch returns full branch name."""
    get_branch.cache_clear()
    mock_run = mocker.patch('openpilot.common.git.run_cmd_default')
    mock_run.return_value = "origin/develop"

    result = get_branch()

    assert result == "origin/develop"


class TestGetOrigin:
  """Test get_origin function."""

  def test_get_origin_returns_url(self, mocker):
    """Test get_origin returns origin URL."""
    get_origin.cache_clear()
    mock_run = mocker.patch('openpilot.common.git.run_cmd')
    mock_run.side_effect = [
      "main",  # branch name
      "origin",  # tracking remote
      "https://github.com/user/repo.git",  # origin url
    ]

    result = get_origin()

    assert result == "https://github.com/user/repo.git"

  def test_get_origin_fallback_on_error(self, mocker):
    """Test get_origin fallback when not on a branch."""
    get_origin.cache_clear()
    mock_run = mocker.patch('openpilot.common.git.run_cmd')
    mock_default = mocker.patch('openpilot.common.git.run_cmd_default')
    mock_run.side_effect = subprocess.CalledProcessError(1, "git")
    mock_default.return_value = "git@github.com:user/repo.git"

    result = get_origin()

    assert result == "git@github.com:user/repo.git"


class TestGetNormalizedOrigin:
  """Test get_normalized_origin function."""

  def test_normalize_https_url(self, mocker):
    """Test normalizing HTTPS URL."""
    get_normalized_origin.cache_clear()
    mock_origin = mocker.patch('openpilot.common.git.get_origin')
    mock_origin.return_value = "https://github.com/user/repo.git"

    result = get_normalized_origin()

    assert result == "github.com/user/repo"

  def test_normalize_ssh_url(self, mocker):
    """Test normalizing SSH URL."""
    get_normalized_origin.cache_clear()
    mock_origin = mocker.patch('openpilot.common.git.get_origin')
    mock_origin.return_value = "git@github.com:user/repo.git"

    result = get_normalized_origin()

    assert result == "github.com/user/repo"

  def test_normalize_removes_git_extension(self, mocker):
    """Test .git extension is removed."""
    get_normalized_origin.cache_clear()
    mock_origin = mocker.patch('openpilot.common.git.get_origin')
    mock_origin.return_value = "https://github.com/user/repo.git"

    result = get_normalized_origin()

    assert ".git" not in result
