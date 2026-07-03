"""Tests for tools/lib/github_utils.py - GitHub API utilities."""

import tempfile
import os

import pytest

from openpilot.tools.lib.github_utils import GithubUtils


class TestGithubUtilsInit:
  """Test GithubUtils initialization."""

  def test_init_default_values(self):
    """Test default initialization values."""
    gh = GithubUtils("api_token", "data_token")

    assert gh.OWNER == "commaai"
    assert gh.API_REPO == "openpilot"
    assert gh.DATA_REPO == "ci-artifacts"
    assert gh.API_TOKEN == "api_token"
    assert gh.DATA_TOKEN == "data_token"

  def test_init_custom_values(self):
    """Test custom initialization values."""
    gh = GithubUtils("api", "data", owner="custom", api_repo="myrepo", data_repo="mydata")

    assert gh.OWNER == "custom"
    assert gh.API_REPO == "myrepo"
    assert gh.DATA_REPO == "mydata"


class TestGithubUtilsRoutes:
  """Test route properties."""

  def test_api_route(self):
    """Test API_ROUTE property."""
    gh = GithubUtils("api", "data", owner="test", api_repo="repo")

    assert gh.API_ROUTE == "https://api.github.com/repos/test/repo"

  def test_data_route(self):
    """Test DATA_ROUTE property."""
    gh = GithubUtils("api", "data", owner="test", data_repo="datarepo")

    assert gh.DATA_ROUTE == "https://api.github.com/repos/test/datarepo"


class TestGithubUtilsApiCall:
  """Test api_call method."""

  def test_api_call_success(self, mocker):
    """Test successful API call."""
    mock_response = mocker.MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {"result": "success"}
    mocker.patch('openpilot.tools.lib.github_utils.requests.request', return_value=mock_response)

    gh = GithubUtils("api_token", "data_token")
    result = gh.api_call("test/path")

    assert result.json() == {"result": "success"}

  def test_api_call_with_data_call(self, mocker):
    """Test API call with data_call=True uses data token."""
    mock_response = mocker.MagicMock()
    mock_response.ok = True
    mock_request = mocker.patch('openpilot.tools.lib.github_utils.requests.request', return_value=mock_response)

    gh = GithubUtils("api_token", "data_token")
    gh.api_call("test/path", data_call=True)

    # Check that the data route is used
    call_args = mock_request.call_args
    assert "ci-artifacts" in call_args[1]['headers']['Authorization'] or "data_token" in call_args[1]['headers']['Authorization']

  def test_api_call_failure_raises(self, mocker):
    """Test API call failure raises exception."""
    mock_response = mocker.MagicMock()
    mock_response.ok = False
    mock_response.status_code = 404
    mocker.patch('openpilot.tools.lib.github_utils.requests.request', return_value=mock_response)

    gh = GithubUtils("api_token", "data_token")

    with pytest.raises(Exception, match="failed with 404"):
      gh.api_call("test/path")

  def test_api_call_failure_no_raise(self, mocker):
    """Test API call failure with raise_on_failure=False."""
    mock_response = mocker.MagicMock()
    mock_response.ok = False
    mock_response.status_code = 404
    mocker.patch('openpilot.tools.lib.github_utils.requests.request', return_value=mock_response)

    gh = GithubUtils("api_token", "data_token")
    result = gh.api_call("test/path", raise_on_failure=False)

    assert result.ok is False

  def test_api_call_without_token(self, mocker):
    """Test API call without token uses empty headers."""
    mock_response = mocker.MagicMock()
    mock_response.ok = True
    mock_request = mocker.patch('openpilot.tools.lib.github_utils.requests.request', return_value=mock_response)

    gh = GithubUtils(None, None)
    gh.api_call("test/path")

    call_args = mock_request.call_args
    assert call_args[1]['headers'] == {}


class TestGithubUtilsBucket:
  """Test bucket operations."""

  def test_get_bucket_sha_success(self, mocker):
    """Test get_bucket_sha returns SHA on success."""
    mock_response = mocker.MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {"object": {"sha": "abc123"}}
    mocker.patch('openpilot.tools.lib.github_utils.requests.request', return_value=mock_response)

    gh = GithubUtils("api", "data")
    result = gh.get_bucket_sha("my-bucket")

    assert result == "abc123"

  def test_get_bucket_sha_not_found(self, mocker):
    """Test get_bucket_sha returns None when not found."""
    mock_response = mocker.MagicMock()
    mock_response.ok = False
    mocker.patch('openpilot.tools.lib.github_utils.requests.request', return_value=mock_response)

    gh = GithubUtils("api", "data")
    result = gh.get_bucket_sha("nonexistent")

    assert result is None

  def test_create_bucket_already_exists(self, mocker):
    """Test create_bucket returns early if bucket exists."""
    mock_response = mocker.MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {"object": {"sha": "existing"}}
    mock_request = mocker.patch('openpilot.tools.lib.github_utils.requests.request', return_value=mock_response)

    gh = GithubUtils("api", "data")
    gh.create_bucket("existing-bucket")

    # Should only call once to check if bucket exists
    assert mock_request.call_count == 1

  def test_create_bucket_new(self, mocker):
    """Test create_bucket creates new bucket."""
    # First call: bucket doesn't exist, second call: get master sha, third call: create
    responses = [
      mocker.MagicMock(ok=False),  # bucket doesn't exist
      mocker.MagicMock(ok=True, json=lambda: {"object": {"sha": "master123"}}),  # get master
      mocker.MagicMock(ok=True),  # create bucket
    ]
    mock_request = mocker.patch('openpilot.tools.lib.github_utils.requests.request', side_effect=responses)

    gh = GithubUtils("api", "data")
    gh.create_bucket("new-bucket")

    assert mock_request.call_count == 3

  def test_get_bucket_link(self):
    """Test get_bucket_link returns correct URL."""
    gh = GithubUtils("api", "data", owner="myorg", data_repo="mydata")
    result = gh.get_bucket_link("my-bucket")

    assert result == "https://raw.githubusercontent.com/myorg/mydata/refs/heads/my-bucket"


class TestGithubUtilsFile:
  """Test file operations."""

  def test_get_file_url(self, mocker):
    """Test get_file_url returns download URL."""
    mock_response = mocker.MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {"download_url": "https://example.com/file.txt"}
    mocker.patch('openpilot.tools.lib.github_utils.requests.request', return_value=mock_response)

    gh = GithubUtils("api", "data")
    result = gh.get_file_url("bucket", "file.txt")

    assert result == "https://example.com/file.txt"

  def test_get_file_sha_exists(self, mocker):
    """Test get_file_sha returns SHA when file exists."""
    mock_response = mocker.MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {"sha": "file123"}
    mocker.patch('openpilot.tools.lib.github_utils.requests.request', return_value=mock_response)

    gh = GithubUtils("api", "data")
    result = gh.get_file_sha("bucket", "file.txt")

    assert result == "file123"

  def test_get_file_sha_not_exists(self, mocker):
    """Test get_file_sha returns None when file doesn't exist."""
    mock_response = mocker.MagicMock()
    mock_response.ok = False
    mocker.patch('openpilot.tools.lib.github_utils.requests.request', return_value=mock_response)

    gh = GithubUtils("api", "data")
    result = gh.get_file_sha("bucket", "nonexistent.txt")

    assert result is None

  def test_upload_file_new(self, mocker):
    """Test upload_file uploads a new file."""
    # First call: file doesn't exist, second call: upload
    responses = [
      mocker.MagicMock(ok=False),  # file doesn't exist
      mocker.MagicMock(ok=True),  # upload
    ]
    mock_request = mocker.patch('openpilot.tools.lib.github_utils.requests.request', side_effect=responses)

    gh = GithubUtils("api", "data")

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
      f.write("test content")
      f.flush()
      try:
        gh.upload_file("bucket", f.name, "test.txt")
      finally:
        os.unlink(f.name)

    assert mock_request.call_count == 2

  def test_upload_file_existing(self, mocker):
    """Test upload_file updates existing file with SHA."""
    # First call: file exists, second call: upload
    responses = [
      mocker.MagicMock(ok=True, json=lambda: {"sha": "existing123"}),  # file exists
      mocker.MagicMock(ok=True),  # upload
    ]
    mock_request = mocker.patch('openpilot.tools.lib.github_utils.requests.request', side_effect=responses)

    gh = GithubUtils("api", "data")

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
      f.write("test content")
      f.flush()
      try:
        gh.upload_file("bucket", f.name, "test.txt")
      finally:
        os.unlink(f.name)

    # Check that SHA was included in the upload
    upload_call = mock_request.call_args_list[1]
    assert "existing123" in upload_call[1]['data']

  def test_upload_files(self, mocker):
    """Test upload_files uploads multiple files."""
    mocker.patch.object(GithubUtils, 'create_bucket')
    mocker.patch.object(GithubUtils, 'upload_file')

    gh = GithubUtils("api", "data")
    files = [("file1.txt", "/path/to/file1"), ("file2.txt", "/path/to/file2")]

    gh.upload_files("bucket", files)

    gh.create_bucket.assert_called_once_with("bucket")
    assert gh.upload_file.call_count == 2


class TestGithubUtilsPR:
  """Test PR operations."""

  def test_get_pr_number(self, mocker):
    """Test get_pr_number returns PR number."""
    mock_response = mocker.MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = [{"number": 123}]
    mocker.patch('openpilot.tools.lib.github_utils.requests.request', return_value=mock_response)

    gh = GithubUtils("api", "data")
    result = gh.get_pr_number("feature-branch")

    assert result == 123

  def test_comment_on_pr_new(self, mocker):
    """Test comment_on_pr creates new comment."""
    mocker.patch.object(GithubUtils, 'get_pr_number', return_value=456)
    mock_response = mocker.MagicMock()
    mock_response.ok = True
    mock_request = mocker.patch('openpilot.tools.lib.github_utils.requests.request', return_value=mock_response)

    gh = GithubUtils("api", "data")
    gh.comment_on_pr("Test comment", "feature-branch")

    # Should post new comment
    assert mock_request.call_count == 1

  def test_comment_on_pr_overwrite_no_existing(self, mocker):
    """Test comment_on_pr with overwrite creates new if no existing."""
    mocker.patch.object(GithubUtils, 'get_pr_number', return_value=456)
    mock_response = mocker.MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = []  # No existing comments
    mock_request = mocker.patch('openpilot.tools.lib.github_utils.requests.request', return_value=mock_response)

    gh = GithubUtils("api", "data")
    gh.comment_on_pr("Test comment", "feature-branch", commenter="bot", overwrite=True)

    # Should get comments, then post new
    assert mock_request.call_count == 2

  def test_comment_on_pr_overwrite_existing(self, mocker):
    """Test comment_on_pr with overwrite updates existing comment."""
    mocker.patch.object(GithubUtils, 'get_pr_number', return_value=456)
    mock_response = mocker.MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = [{"id": 789, "user": {"login": "bot"}}]
    mock_request = mocker.patch('openpilot.tools.lib.github_utils.requests.request', return_value=mock_response)

    gh = GithubUtils("api", "data")
    gh.comment_on_pr("Updated comment", "feature-branch", commenter="bot", overwrite=True)

    # Should get comments, then patch existing
    assert mock_request.call_count == 2

  def test_comment_images_on_pr(self, mocker):
    """Test comment_images_on_pr uploads and comments."""
    mocker.patch.object(GithubUtils, 'upload_files')
    mocker.patch.object(GithubUtils, 'comment_on_pr')

    gh = GithubUtils("api", "data")
    images = [("img1.png", "/path/to/img1"), ("img2.png", "/path/to/img2")]

    gh.comment_images_on_pr("Test Title", "bot", "feature-branch", "bucket", images)

    gh.upload_files.assert_called_once()
    gh.comment_on_pr.assert_called_once()
