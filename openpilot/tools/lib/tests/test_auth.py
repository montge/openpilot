"""Tests for openpilot/tools/lib/auth.py - OAuth authentication utilities."""

import io
from urllib.parse import urlparse

import pytest

from openpilot.tools.lib.auth import (
  ClientRedirectHandler,
  ClientRedirectServer,
  auth_redirect_link,
)


class TestAuthRedirectLink:
  """Test auth_redirect_link function."""

  # fork: upstream made the local callback port an explicit argument
  PORT = 9090

  def test_google_redirect(self):
    """Test Google OAuth redirect URL."""
    url = auth_redirect_link('google', self.PORT)

    assert urlparse(url).hostname == 'accounts.google.com'
    assert 'client_id=' in url
    assert 'scope=' in url
    assert 'redirect_uri=' in url

  def test_github_redirect(self):
    """Test GitHub OAuth redirect URL."""
    url = auth_redirect_link('github', self.PORT)

    assert 'github.com/login/oauth/authorize' in url
    assert 'client_id=' in url
    assert 'scope=' in url

  def test_apple_redirect(self):
    """Test Apple OAuth redirect URL."""
    url = auth_redirect_link('apple', self.PORT)

    assert 'appleid.apple.com/auth/authorize' in url
    assert 'client_id=' in url
    assert 'response_mode=' in url

  def test_unsupported_method_raises(self):
    """Test unsupported method raises KeyError."""
    with pytest.raises(KeyError):
      auth_redirect_link('unsupported', self.PORT)


class TestClientRedirectServer:
  """Test ClientRedirectServer class."""

  def test_query_params_default(self):
    """Test query_params starts empty on a fresh server (per-instance since upstream #38893)."""
    with ClientRedirectServer(('localhost', 0), ClientRedirectHandler) as server:
      assert server.query_params == {}
      assert server.server_port > 0


class TestClientRedirectHandler:
  """Test ClientRedirectHandler class."""

  def test_do_get_non_auth_path(self, mocker):
    """Test do_GET with non-auth path returns 204."""
    # Create mock request and server
    mock_request = mocker.MagicMock()
    mock_request.makefile.return_value = io.BytesIO(b"GET /favicon.ico HTTP/1.1\r\n\r\n")

    mock_server = mocker.MagicMock()
    mock_server.query_params = {}

    # Create handler with mocked internals
    handler = ClientRedirectHandler.__new__(ClientRedirectHandler)
    handler.path = '/favicon.ico'
    handler.requestline = 'GET /favicon.ico HTTP/1.1'
    handler.request_version = 'HTTP/1.1'
    handler.server = mock_server
    handler.client_address = ('127.0.0.1', 12345)
    handler.wfile = io.BytesIO()
    handler.send_response = mocker.MagicMock()
    handler.send_header = mocker.MagicMock()
    handler.end_headers = mocker.MagicMock()

    handler.do_GET()

    handler.send_response.assert_called_once_with(204)

  def test_do_get_auth_path(self, mocker):
    """Test do_GET with auth path stores query params."""
    mock_server = mocker.MagicMock()
    mock_server.query_params = {}

    handler = ClientRedirectHandler.__new__(ClientRedirectHandler)
    handler.path = '/auth?code=abc123&provider=google'
    handler.requestline = 'GET /auth?code=abc123&provider=google HTTP/1.1'
    handler.request_version = 'HTTP/1.1'
    handler.server = mock_server
    handler.client_address = ('127.0.0.1', 12345)
    handler.wfile = io.BytesIO()
    handler.send_response = mocker.MagicMock()
    handler.send_header = mocker.MagicMock()
    handler.end_headers = mocker.MagicMock()

    handler.do_GET()

    handler.send_response.assert_called_once_with(200)
    assert mock_server.query_params == {'code': ['abc123'], 'provider': ['google']}

  def test_log_message_suppressed(self, mocker):
    """Test log_message does nothing (suppresses output)."""
    handler = ClientRedirectHandler.__new__(ClientRedirectHandler)

    # Should not raise and should not output anything
    handler.log_message("test message")


class TestLogin:
  """Test login function with mocking.

  fork: upstream #38893 reworked login() around a context-managed ThreadingHTTPServer,
  opens the browser via xdg-open/open in a subprocess, verifies the token with v1/me,
  and returns a status dict instead of printing.
  """

  def _mock_flow(self, mocker, query_params):
    from openpilot.tools.lib import auth

    browser = mocker.MagicMock()
    browser.poll.return_value = None
    mocker.patch.object(auth.subprocess, 'Popen', return_value=browser)

    mock_server = mocker.MagicMock()
    mock_server.__enter__.return_value = mock_server
    mock_server.server_port = 9090
    mock_server.query_params = query_params
    mocker.patch.object(auth, 'ClientRedirectServer', return_value=mock_server)

    mock_api = mocker.MagicMock()
    mocker.patch('openpilot.tools.lib.auth.CommaApi', return_value=mock_api)
    return auth, mock_api

  def test_login_success(self, mocker):
    """Test successful login flow saves the token and reports success."""
    auth, mock_api = self._mock_flow(mocker, {'code': ['authcode123'], 'provider': ['g']})
    mock_api.post.return_value = {'access_token': 'token123'}
    mock_api.get.return_value = {}
    mock_set_token = mocker.patch('openpilot.tools.lib.auth.set_token')

    result = auth.login('google')

    assert result == {"success": True}
    mock_api.post.assert_called_once()
    assert mock_api.post.call_args.kwargs['data'] == {'code': ['authcode123'], 'provider': ['g']}
    mock_set_token.assert_called_once_with('token123')

  def test_login_api_error(self, mocker):
    """Test login with API error returns an error status and saves nothing."""
    from openpilot.tools.lib.api import APIError

    auth, mock_api = self._mock_flow(mocker, {'code': ['authcode123'], 'provider': ['g']})
    mock_api.post.side_effect = APIError("API failed")
    mock_set_token = mocker.patch('openpilot.tools.lib.auth.set_token')

    result = auth.login('google')

    assert 'error' in result
    mock_set_token.assert_not_called()

  def test_login_declined(self, mocker):
    """Test an OAuth error callback is reported as declined."""
    auth, mock_api = self._mock_flow(mocker, {'error': ['access_denied']})

    result = auth.login('google')

    assert 'error' in result
    mock_api.post.assert_not_called()
