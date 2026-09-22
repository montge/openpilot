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
    """Each server starts with no captured query params (an instance attribute since #38893)."""
    with ClientRedirectServer(('localhost', 0), ClientRedirectHandler) as server:
      assert server.query_params == {}


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
  """Test login function with mocking (the browser flow returns a status dict since #38893)."""

  @pytest.fixture
  def server(self, mocker):
    from openpilot.tools.lib import auth

    server = mocker.MagicMock()
    server.__enter__.return_value = server
    server.server_port = 3000
    server.query_params = {'code': ['authcode123'], 'provider': ['g']}
    mocker.patch.object(auth, 'ClientRedirectServer', return_value=server)
    mocker.patch.object(auth, 'subprocess')  # never open a real browser
    return server

  def test_login_success(self, mocker, server):
    """Test successful login flow."""
    from openpilot.tools.lib import auth

    mock_api = mocker.MagicMock()
    mock_api.post.return_value = {'access_token': 'token123'}
    mocker.patch('openpilot.tools.lib.auth.CommaApi', return_value=mock_api)
    mock_set_token = mocker.patch('openpilot.tools.lib.auth.set_token')

    assert auth.login('google') == {"success": True}
    mock_set_token.assert_called_once_with('token123')

  def test_login_api_error(self, mocker, server):
    """An API failure is reported as an error status and saves no token."""
    from openpilot.tools.lib import auth
    from openpilot.tools.lib.api import APIError

    mock_api = mocker.MagicMock()
    mock_api.post.side_effect = APIError("API failed")
    mocker.patch('openpilot.tools.lib.auth.CommaApi', return_value=mock_api)
    mock_set_token = mocker.patch('openpilot.tools.lib.auth.set_token')

    assert "error" in auth.login('google')
    mock_set_token.assert_not_called()

  def test_login_declined(self, mocker, server):
    """The provider redirecting back with an error is reported as declined."""
    from openpilot.tools.lib import auth

    server.query_params = {'error': ['access_denied']}
    mock_set_token = mocker.patch('openpilot.tools.lib.auth.set_token')

    assert "declined" in auth.login('google')["error"]
    mock_set_token.assert_not_called()

  def test_login_provider_mismatch(self, mocker, server):
    """A code for a different provider than requested is rejected."""
    from openpilot.tools.lib import auth

    mock_api = mocker.patch('openpilot.tools.lib.auth.CommaApi')
    assert "Invalid" in auth.login('github')["error"]  # server returned a google ('g') code
    mock_api.assert_not_called()
