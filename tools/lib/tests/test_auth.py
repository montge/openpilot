"""Tests for tools/lib/auth.py - OAuth authentication utilities."""

import io
import pytest

from openpilot.tools.lib.auth import (
  ClientRedirectHandler,
  ClientRedirectServer,
  auth_redirect_link,
)


class TestAuthRedirectLink:
  """Test auth_redirect_link function."""

  def test_google_redirect(self):
    """Test Google OAuth redirect URL."""
    url = auth_redirect_link('google')

    assert 'accounts.google.com' in url
    assert 'client_id=' in url
    assert 'scope=' in url
    assert 'redirect_uri=' in url

  def test_github_redirect(self):
    """Test GitHub OAuth redirect URL."""
    url = auth_redirect_link('github')

    assert 'github.com/login/oauth/authorize' in url
    assert 'client_id=' in url
    assert 'scope=' in url

  def test_apple_redirect(self):
    """Test Apple OAuth redirect URL."""
    url = auth_redirect_link('apple')

    assert 'appleid.apple.com/auth/authorize' in url
    assert 'client_id=' in url
    assert 'response_mode=' in url

  def test_unsupported_method_raises(self):
    """Test unsupported method raises KeyError."""
    with pytest.raises(KeyError):
      auth_redirect_link('unsupported')


class TestClientRedirectServer:
  """Test ClientRedirectServer class."""

  def test_query_params_default(self):
    """Test query_params defaults to empty dict."""
    # We can't easily create a full server, but we can check the class attribute
    assert ClientRedirectServer.query_params == {}


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
  """Test login function with mocking."""

  def test_login_success(self, mocker):
    """Test successful login flow."""
    from openpilot.tools.lib import auth

    # Mock webbrowser
    mocker.patch.object(auth, 'webbrowser')

    # Mock server that returns code on first request
    mock_server = mocker.MagicMock()
    mock_server.query_params = {'code': ['authcode123'], 'provider': ['google']}
    mocker.patch.object(auth, 'ClientRedirectServer', return_value=mock_server)

    # Mock API response
    mock_api = mocker.MagicMock()
    mock_api.post.return_value = {'access_token': 'token123'}
    mocker.patch('openpilot.tools.lib.auth.CommaApi', return_value=mock_api)

    # Mock set_token
    mock_set_token = mocker.patch('openpilot.tools.lib.auth.set_token')

    auth.login('google')

    mock_set_token.assert_called_once_with('token123')

  def test_login_api_error(self, mocker, capsys):
    """Test login with API error."""
    from openpilot.tools.lib import auth
    from openpilot.tools.lib.api import APIError

    mocker.patch.object(auth, 'webbrowser')

    mock_server = mocker.MagicMock()
    mock_server.query_params = {'code': ['authcode123'], 'provider': ['google']}
    mocker.patch.object(auth, 'ClientRedirectServer', return_value=mock_server)

    mock_api = mocker.MagicMock()
    mock_api.post.side_effect = APIError("API failed")
    mocker.patch('openpilot.tools.lib.auth.CommaApi', return_value=mock_api)

    auth.login('google')

    captured = capsys.readouterr()
    assert 'Authentication Error' in captured.err
