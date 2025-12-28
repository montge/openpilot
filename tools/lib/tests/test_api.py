"""Tests for tools/lib/api.py - CommaApi client."""
import unittest
from unittest.mock import MagicMock, patch

from openpilot.tools.lib.api import (
  API_HOST, CommaApi, APIError, UnauthorizedError,
)


class TestAPIError(unittest.TestCase):
  """Test APIError exception."""

  def test_is_exception(self):
    """Test APIError is an Exception."""
    self.assertTrue(issubclass(APIError, Exception))

  def test_can_raise(self):
    """Test can raise APIError."""
    with self.assertRaises(APIError):
      raise APIError("test error")

  def test_has_message(self):
    """Test APIError carries message."""
    try:
      raise APIError("test message")
    except APIError as e:
      self.assertEqual(str(e), "test message")

  def test_can_have_status_code(self):
    """Test APIError can have status_code attribute."""
    e = APIError("error")
    e.status_code = 404
    self.assertEqual(e.status_code, 404)


class TestUnauthorizedError(unittest.TestCase):
  """Test UnauthorizedError exception."""

  def test_is_exception(self):
    """Test UnauthorizedError is an Exception."""
    self.assertTrue(issubclass(UnauthorizedError, Exception))

  def test_can_raise(self):
    """Test can raise UnauthorizedError."""
    with self.assertRaises(UnauthorizedError):
      raise UnauthorizedError("unauthorized")

  def test_has_message(self):
    """Test UnauthorizedError carries message."""
    try:
      raise UnauthorizedError("not authorized")
    except UnauthorizedError as e:
      self.assertEqual(str(e), "not authorized")


class TestCommaApiInit(unittest.TestCase):
  """Test CommaApi initialization."""

  @patch('openpilot.tools.lib.api.requests.Session')
  def test_creates_session(self, mock_session_class):
    """Test __init__ creates a requests Session."""
    CommaApi()

    mock_session_class.assert_called_once()

  @patch('openpilot.tools.lib.api.requests.Session')
  def test_sets_user_agent(self, mock_session_class):
    """Test __init__ sets User-agent header."""
    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session_class.return_value = mock_session

    CommaApi()

    self.assertEqual(mock_session.headers['User-agent'], 'OpenpilotTools')

  @patch('openpilot.tools.lib.api.requests.Session')
  def test_no_auth_header_without_token(self, mock_session_class):
    """Test __init__ without token sets no Authorization header."""
    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session_class.return_value = mock_session

    CommaApi()

    self.assertNotIn('Authorization', mock_session.headers)

  @patch('openpilot.tools.lib.api.requests.Session')
  def test_sets_auth_header_with_token(self, mock_session_class):
    """Test __init__ with token sets Authorization header."""
    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session_class.return_value = mock_session

    CommaApi(token='my_token')

    self.assertEqual(mock_session.headers['Authorization'], 'JWT my_token')

  @patch('openpilot.tools.lib.api.requests.Session')
  def test_jwt_prefix(self, mock_session_class):
    """Test Authorization header has JWT prefix."""
    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session_class.return_value = mock_session

    CommaApi(token='abc123')

    self.assertTrue(mock_session.headers['Authorization'].startswith('JWT '))


class TestCommaApiRequest(unittest.TestCase):
  """Test CommaApi.request method."""

  def setUp(self):
    """Set up test fixtures."""
    self.session_patcher = patch('openpilot.tools.lib.api.requests.Session')
    self.mock_session_class = self.session_patcher.start()
    self.mock_session = MagicMock()
    self.mock_session.headers = {}
    self.mock_session_class.return_value = self.mock_session

    self.mock_response = MagicMock()
    self.mock_session.request.return_value.__enter__ = MagicMock(return_value=self.mock_response)
    self.mock_session.request.return_value.__exit__ = MagicMock(return_value=False)

  def tearDown(self):
    """Tear down test fixtures."""
    self.session_patcher.stop()

  def test_calls_session_request(self):
    """Test request calls session.request with correct args."""
    self.mock_response.json.return_value = {'data': 'value'}

    api = CommaApi()
    api.request('GET', 'v1/me')

    self.mock_session.request.assert_called_once_with('GET', API_HOST + '/v1/me')

  def test_passes_kwargs(self):
    """Test request passes kwargs to session.request."""
    self.mock_response.json.return_value = {'data': 'value'}

    api = CommaApi()
    api.request('POST', 'v1/endpoint', json={'key': 'val'})

    self.mock_session.request.assert_called_once_with(
      'POST', API_HOST + '/v1/endpoint', json={'key': 'val'}
    )

  def test_returns_json_response(self):
    """Test request returns parsed JSON."""
    self.mock_response.json.return_value = {'result': 'success'}

    api = CommaApi()
    result = api.request('GET', 'v1/test')

    self.assertEqual(result, {'result': 'success'})

  def test_returns_list_response(self):
    """Test request returns list JSON."""
    self.mock_response.json.return_value = [1, 2, 3]

    api = CommaApi()
    result = api.request('GET', 'v1/list')

    self.assertEqual(result, [1, 2, 3])

  def test_raises_unauthorized_on_401(self):
    """Test request raises UnauthorizedError on 401."""
    self.mock_response.json.return_value = {'error': True}
    self.mock_response.status_code = 401

    api = CommaApi()

    with self.assertRaises(UnauthorizedError):
      api.request('GET', 'v1/protected')

  def test_raises_unauthorized_on_403(self):
    """Test request raises UnauthorizedError on 403."""
    self.mock_response.json.return_value = {'error': True}
    self.mock_response.status_code = 403

    api = CommaApi()

    with self.assertRaises(UnauthorizedError):
      api.request('GET', 'v1/forbidden')

  def test_raises_api_error_on_other_errors(self):
    """Test request raises APIError on other error codes."""
    self.mock_response.json.return_value = {'error': True, 'description': 'Not found'}
    self.mock_response.status_code = 404

    api = CommaApi()

    with self.assertRaises(APIError):
      api.request('GET', 'v1/missing')

  def test_api_error_includes_status_code(self):
    """Test APIError includes status code in message."""
    self.mock_response.json.return_value = {'error': True, 'description': 'Bad request'}
    self.mock_response.status_code = 400

    api = CommaApi()

    try:
      api.request('GET', 'v1/bad')
    except APIError as e:
      self.assertIn('400', str(e))
      self.assertEqual(e.status_code, 400)

  def test_api_error_includes_description(self):
    """Test APIError includes description in message."""
    self.mock_response.json.return_value = {'error': True, 'description': 'Specific error'}
    self.mock_response.status_code = 500

    api = CommaApi()

    try:
      api.request('GET', 'v1/error')
    except APIError as e:
      self.assertIn('Specific error', str(e))

  def test_api_error_fallback_without_description(self):
    """Test APIError uses error value when no description."""
    self.mock_response.json.return_value = {'error': 'some_error'}
    self.mock_response.status_code = 422

    api = CommaApi()

    try:
      api.request('GET', 'v1/error')
    except APIError as e:
      self.assertIn('some_error', str(e))

  def test_no_error_on_success(self):
    """Test request doesn't raise on success response."""
    self.mock_response.json.return_value = {'success': True}

    api = CommaApi()
    result = api.request('GET', 'v1/ok')

    self.assertEqual(result, {'success': True})


class TestCommaApiGet(unittest.TestCase):
  """Test CommaApi.get method."""

  @patch('openpilot.tools.lib.api.requests.Session')
  def test_calls_request_with_get(self, mock_session_class):
    """Test get calls request with GET method."""
    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session_class.return_value = mock_session

    mock_response = MagicMock()
    mock_response.json.return_value = {'data': 'value'}
    mock_session.request.return_value.__enter__ = MagicMock(return_value=mock_response)
    mock_session.request.return_value.__exit__ = MagicMock(return_value=False)

    api = CommaApi()
    api.get('v1/endpoint')

    mock_session.request.assert_called_once_with('GET', API_HOST + '/v1/endpoint')

  @patch('openpilot.tools.lib.api.requests.Session')
  def test_passes_kwargs(self, mock_session_class):
    """Test get passes kwargs to request."""
    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session_class.return_value = mock_session

    mock_response = MagicMock()
    mock_response.json.return_value = {'data': 'value'}
    mock_session.request.return_value.__enter__ = MagicMock(return_value=mock_response)
    mock_session.request.return_value.__exit__ = MagicMock(return_value=False)

    api = CommaApi()
    api.get('v1/endpoint', params={'q': 'search'})

    mock_session.request.assert_called_once_with(
      'GET', API_HOST + '/v1/endpoint', params={'q': 'search'}
    )


class TestCommaApiPost(unittest.TestCase):
  """Test CommaApi.post method."""

  @patch('openpilot.tools.lib.api.requests.Session')
  def test_calls_request_with_post(self, mock_session_class):
    """Test post calls request with POST method."""
    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session_class.return_value = mock_session

    mock_response = MagicMock()
    mock_response.json.return_value = {'created': True}
    mock_session.request.return_value.__enter__ = MagicMock(return_value=mock_response)
    mock_session.request.return_value.__exit__ = MagicMock(return_value=False)

    api = CommaApi()
    api.post('v1/endpoint')

    mock_session.request.assert_called_once_with('POST', API_HOST + '/v1/endpoint')

  @patch('openpilot.tools.lib.api.requests.Session')
  def test_passes_json_data(self, mock_session_class):
    """Test post passes json data."""
    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session_class.return_value = mock_session

    mock_response = MagicMock()
    mock_response.json.return_value = {'created': True}
    mock_session.request.return_value.__enter__ = MagicMock(return_value=mock_response)
    mock_session.request.return_value.__exit__ = MagicMock(return_value=False)

    api = CommaApi()
    api.post('v1/endpoint', json={'name': 'test'})

    mock_session.request.assert_called_once_with(
      'POST', API_HOST + '/v1/endpoint', json={'name': 'test'}
    )


class TestAPIHost(unittest.TestCase):
  """Test API_HOST configuration."""

  def test_default_host(self):
    """Test API_HOST has default value."""
    # Note: API_HOST is set at module load time from env
    self.assertIn('commadotai.com', API_HOST)

  def test_host_is_https(self):
    """Test default API_HOST uses HTTPS."""
    self.assertTrue(API_HOST.startswith('https://'))


if __name__ == '__main__':
  unittest.main()
