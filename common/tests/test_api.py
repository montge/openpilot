"""Tests for common/api.py - API utilities."""

import os
import tempfile


from openpilot.common.api import (
  Api,
  api_get,
  get_key_pair,
  API_HOST,
  KEYS,
)


class TestApiConstants:
  """Test API module constants."""

  def test_api_host_default(self):
    """Test API_HOST has a default value."""
    assert "commadotai.com" in API_HOST

  def test_keys_contains_rsa(self):
    """Test KEYS contains RSA algorithm."""
    assert "id_rsa" in KEYS
    assert KEYS["id_rsa"] == "RS256"

  def test_keys_contains_ecdsa(self):
    """Test KEYS contains ECDSA algorithm."""
    assert "id_ecdsa" in KEYS
    assert KEYS["id_ecdsa"] == "ES256"


class TestApiClass:
  """Test Api class."""

  def test_api_init(self, mocker):
    """Test Api initialization."""
    mock_get_key_pair = mocker.patch('openpilot.common.api.get_key_pair')
    mock_get_key_pair.return_value = ("RS256", "private_key", "public_key")

    api = Api("test_dongle_id")

    assert api.dongle_id == "test_dongle_id"
    assert api.jwt_algorithm == "RS256"
    assert api.private_key == "private_key"

  def test_api_get_calls_request(self, mocker):
    """Test Api.get calls api_get with GET method."""
    mock_get_key_pair = mocker.patch('openpilot.common.api.get_key_pair')
    mock_api_get = mocker.patch('openpilot.common.api.api_get')
    mock_get_key_pair.return_value = ("RS256", "private_key", "public_key")
    mock_api_get.return_value = mocker.MagicMock()

    api = Api("test_dongle")
    api.get("endpoint", timeout=10)

    mock_api_get.assert_called_once()
    call_args = mock_api_get.call_args
    assert call_args[0][0] == "endpoint"
    assert call_args[1].get('method') == 'GET'
    assert call_args[1].get('timeout') == 10

  def test_api_post_calls_request(self, mocker):
    """Test Api.post calls api_get with POST method."""
    mock_get_key_pair = mocker.patch('openpilot.common.api.get_key_pair')
    mock_api_get = mocker.patch('openpilot.common.api.api_get')
    mock_get_key_pair.return_value = ("RS256", "private_key", "public_key")
    mock_api_get.return_value = mocker.MagicMock()

    api = Api("test_dongle")
    api.post("endpoint")

    mock_api_get.assert_called_once()
    call_args = mock_api_get.call_args
    assert call_args[1].get('method') == 'POST'


class TestApiGetToken:
  """Test Api.get_token method."""

  def test_get_token_creates_jwt(self, mocker):
    """Test get_token creates a JWT."""
    mock_get_key_pair = mocker.patch('openpilot.common.api.get_key_pair')
    mock_encode = mocker.patch('openpilot.common.api.jwt.encode')
    mock_get_key_pair.return_value = ("RS256", "test_private_key", "public_key")
    mock_encode.return_value = "test_token"

    api = Api("test_dongle")
    token = api.get_token()

    assert token == "test_token"
    mock_encode.assert_called_once()

  def test_get_token_payload_contains_identity(self, mocker):
    """Test get_token payload contains dongle identity."""
    mock_get_key_pair = mocker.patch('openpilot.common.api.get_key_pair')
    mock_encode = mocker.patch('openpilot.common.api.jwt.encode')
    mock_get_key_pair.return_value = ("RS256", "private_key", "public_key")
    mock_encode.return_value = "token"

    api = Api("my_dongle_id")
    api.get_token()

    call_args = mock_encode.call_args
    payload = call_args[0][0]
    assert payload['identity'] == 'my_dongle_id'

  def test_get_token_payload_contains_timestamps(self, mocker):
    """Test get_token payload contains timing claims."""
    mock_get_key_pair = mocker.patch('openpilot.common.api.get_key_pair')
    mock_encode = mocker.patch('openpilot.common.api.jwt.encode')
    mock_get_key_pair.return_value = ("RS256", "private_key", "public_key")
    mock_encode.return_value = "token"

    api = Api("dongle")
    api.get_token()

    call_args = mock_encode.call_args
    payload = call_args[0][0]
    assert 'nbf' in payload
    assert 'iat' in payload
    assert 'exp' in payload

  def test_get_token_with_payload_extra(self, mocker):
    """Test get_token includes extra payload."""
    mock_get_key_pair = mocker.patch('openpilot.common.api.get_key_pair')
    mock_encode = mocker.patch('openpilot.common.api.jwt.encode')
    mock_get_key_pair.return_value = ("RS256", "private_key", "public_key")
    mock_encode.return_value = "token"

    api = Api("dongle")
    api.get_token(payload_extra={'scope': 'read'})

    call_args = mock_encode.call_args
    payload = call_args[0][0]
    assert payload['scope'] == 'read'

  def test_get_token_bytes_decoded(self, mocker):
    """Test get_token decodes bytes to string."""
    mock_get_key_pair = mocker.patch('openpilot.common.api.get_key_pair')
    mock_encode = mocker.patch('openpilot.common.api.jwt.encode')
    mock_get_key_pair.return_value = ("RS256", "private_key", "public_key")
    mock_encode.return_value = b"token_bytes"

    api = Api("dongle")
    token = api.get_token()

    assert token == "token_bytes"
    assert isinstance(token, str)


class TestApiGet:
  """Test api_get function."""

  def test_api_get_constructs_url(self, mocker):
    """Test api_get constructs proper URL."""
    mock_version = mocker.patch('openpilot.common.api.get_version')
    mock_request = mocker.patch('openpilot.common.api.requests.request')
    mock_version.return_value = "0.9.7"
    mock_request.return_value = mocker.MagicMock()

    api_get("v1/test")

    mock_request.assert_called_once()
    call_args = mock_request.call_args
    assert API_HOST + "/v1/test" in call_args[1].get('url', call_args[0][1])

  def test_api_get_includes_user_agent(self, mocker):
    """Test api_get includes User-Agent header."""
    mock_version = mocker.patch('openpilot.common.api.get_version')
    mock_request = mocker.patch('openpilot.common.api.requests.request')
    mock_version.return_value = "0.9.7"
    mock_request.return_value = mocker.MagicMock()

    api_get("endpoint")

    call_args = mock_request.call_args
    headers = call_args[1].get('headers', {})
    assert 'User-Agent' in headers
    assert "openpilot-0.9.7" in headers['User-Agent']

  def test_api_get_with_access_token(self, mocker):
    """Test api_get includes Authorization header."""
    mock_version = mocker.patch('openpilot.common.api.get_version')
    mock_request = mocker.patch('openpilot.common.api.requests.request')
    mock_version.return_value = "0.9.7"
    mock_request.return_value = mocker.MagicMock()

    api_get("endpoint", access_token="my_token")

    call_args = mock_request.call_args
    headers = call_args[1].get('headers', {})
    assert headers['Authorization'] == "JWT my_token"

  def test_api_get_without_access_token(self, mocker):
    """Test api_get without Authorization header."""
    mock_version = mocker.patch('openpilot.common.api.get_version')
    mock_request = mocker.patch('openpilot.common.api.requests.request')
    mock_version.return_value = "0.9.7"
    mock_request.return_value = mocker.MagicMock()

    api_get("endpoint")

    call_args = mock_request.call_args
    headers = call_args[1].get('headers', {})
    assert 'Authorization' not in headers

  def test_api_get_with_session(self, mocker):
    """Test api_get uses provided session."""
    mock_version = mocker.patch('openpilot.common.api.get_version')
    mock_request = mocker.patch('openpilot.common.api.requests.request')
    mock_version.return_value = "0.9.7"
    mock_session = mocker.MagicMock()

    api_get("endpoint", session=mock_session)

    mock_session.request.assert_called_once()
    mock_request.assert_not_called()

  def test_api_get_passes_params(self, mocker):
    """Test api_get passes query parameters."""
    mock_version = mocker.patch('openpilot.common.api.get_version')
    mock_request = mocker.patch('openpilot.common.api.requests.request')
    mock_version.return_value = "0.9.7"
    mock_request.return_value = mocker.MagicMock()

    api_get("endpoint", foo="bar", count=5)

    call_args = mock_request.call_args
    params = call_args[1].get('params', {})
    assert params['foo'] == 'bar'
    assert params['count'] == 5


class TestGetKeyPair:
  """Test get_key_pair function."""

  def test_get_key_pair_returns_none_when_no_keys(self, mocker):
    """Test get_key_pair returns None tuple when no keys exist."""
    mock_persist = mocker.patch('openpilot.common.api.Paths.persist_root')
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_persist.return_value = tmpdir

      result = get_key_pair()

      assert result == (None, None, None)

  def test_get_key_pair_reads_rsa_key(self, mocker):
    """Test get_key_pair reads RSA key pair."""
    mock_persist = mocker.patch('openpilot.common.api.Paths.persist_root')
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_persist.return_value = tmpdir
      comma_dir = os.path.join(tmpdir, "comma")
      os.makedirs(comma_dir)

      with open(os.path.join(comma_dir, "id_rsa"), 'w') as f:
        f.write("private_rsa_key")
      with open(os.path.join(comma_dir, "id_rsa.pub"), 'w') as f:
        f.write("public_rsa_key")

      alg, private, public = get_key_pair()

      assert alg == "RS256"
      assert private == "private_rsa_key"
      assert public == "public_rsa_key"

  def test_get_key_pair_reads_ecdsa_key(self, mocker):
    """Test get_key_pair reads ECDSA key pair."""
    mock_persist = mocker.patch('openpilot.common.api.Paths.persist_root')
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_persist.return_value = tmpdir
      comma_dir = os.path.join(tmpdir, "comma")
      os.makedirs(comma_dir)

      with open(os.path.join(comma_dir, "id_ecdsa"), 'w') as f:
        f.write("private_ecdsa_key")
      with open(os.path.join(comma_dir, "id_ecdsa.pub"), 'w') as f:
        f.write("public_ecdsa_key")

      alg, private, public = get_key_pair()

      assert alg == "ES256"
      assert private == "private_ecdsa_key"
      assert public == "public_ecdsa_key"

  def test_get_key_pair_prefers_rsa(self, mocker):
    """Test get_key_pair prefers RSA over ECDSA when both exist."""
    mock_persist = mocker.patch('openpilot.common.api.Paths.persist_root')
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_persist.return_value = tmpdir
      comma_dir = os.path.join(tmpdir, "comma")
      os.makedirs(comma_dir)

      for key_type in ["id_rsa", "id_ecdsa"]:
        with open(os.path.join(comma_dir, key_type), 'w') as f:
          f.write(f"private_{key_type}")
        with open(os.path.join(comma_dir, f"{key_type}.pub"), 'w') as f:
          f.write(f"public_{key_type}")

      alg, private, public = get_key_pair()

      assert alg == "RS256"

  def test_get_key_pair_requires_both_files(self, mocker):
    """Test get_key_pair requires both private and public key."""
    mock_persist = mocker.patch('openpilot.common.api.Paths.persist_root')
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_persist.return_value = tmpdir
      comma_dir = os.path.join(tmpdir, "comma")
      os.makedirs(comma_dir)

      with open(os.path.join(comma_dir, "id_rsa"), 'w') as f:
        f.write("private_key")

      result = get_key_pair()

      assert result == (None, None, None)
