"""Tests for common/api.py - API utilities."""
import os
import tempfile
import unittest
from datetime import datetime, timedelta, UTC
from unittest.mock import patch, MagicMock

from openpilot.common.api import (
  Api, api_get, get_key_pair,
  API_HOST, KEYS,
)


class TestApiConstants(unittest.TestCase):
  """Test API module constants."""

  def test_api_host_default(self):
    """Test API_HOST has a default value."""
    # When API_HOST env var is not set, it should have a default
    self.assertIn("commadotai.com", API_HOST)

  def test_keys_contains_rsa(self):
    """Test KEYS contains RSA algorithm."""
    self.assertIn("id_rsa", KEYS)
    self.assertEqual(KEYS["id_rsa"], "RS256")

  def test_keys_contains_ecdsa(self):
    """Test KEYS contains ECDSA algorithm."""
    self.assertIn("id_ecdsa", KEYS)
    self.assertEqual(KEYS["id_ecdsa"], "ES256")


class TestApiClass(unittest.TestCase):
  """Test Api class."""

  @patch('openpilot.common.api.get_key_pair')
  def test_api_init(self, mock_get_key_pair):
    """Test Api initialization."""
    mock_get_key_pair.return_value = ("RS256", "private_key", "public_key")

    api = Api("test_dongle_id")

    self.assertEqual(api.dongle_id, "test_dongle_id")
    self.assertEqual(api.jwt_algorithm, "RS256")
    self.assertEqual(api.private_key, "private_key")

  @patch('openpilot.common.api.get_key_pair')
  @patch('openpilot.common.api.api_get')
  def test_api_get_calls_request(self, mock_api_get, mock_get_key_pair):
    """Test Api.get calls api_get with GET method."""
    mock_get_key_pair.return_value = ("RS256", "private_key", "public_key")
    mock_api_get.return_value = MagicMock()

    api = Api("test_dongle")
    api.get("endpoint", timeout=10)

    mock_api_get.assert_called_once()
    call_args = mock_api_get.call_args
    self.assertEqual(call_args[0][0], "endpoint")
    self.assertEqual(call_args[1].get('method'), 'GET')
    self.assertEqual(call_args[1].get('timeout'), 10)

  @patch('openpilot.common.api.get_key_pair')
  @patch('openpilot.common.api.api_get')
  def test_api_post_calls_request(self, mock_api_get, mock_get_key_pair):
    """Test Api.post calls api_get with POST method."""
    mock_get_key_pair.return_value = ("RS256", "private_key", "public_key")
    mock_api_get.return_value = MagicMock()

    api = Api("test_dongle")
    api.post("endpoint")

    mock_api_get.assert_called_once()
    call_args = mock_api_get.call_args
    self.assertEqual(call_args[1].get('method'), 'POST')


class TestApiGetToken(unittest.TestCase):
  """Test Api.get_token method."""

  @patch('openpilot.common.api.get_key_pair')
  @patch('openpilot.common.api.jwt.encode')
  def test_get_token_creates_jwt(self, mock_encode, mock_get_key_pair):
    """Test get_token creates a JWT."""
    mock_get_key_pair.return_value = ("RS256", "test_private_key", "public_key")
    mock_encode.return_value = "test_token"

    api = Api("test_dongle")
    token = api.get_token()

    self.assertEqual(token, "test_token")
    mock_encode.assert_called_once()

  @patch('openpilot.common.api.get_key_pair')
  @patch('openpilot.common.api.jwt.encode')
  def test_get_token_payload_contains_identity(self, mock_encode, mock_get_key_pair):
    """Test get_token payload contains dongle identity."""
    mock_get_key_pair.return_value = ("RS256", "private_key", "public_key")
    mock_encode.return_value = "token"

    api = Api("my_dongle_id")
    api.get_token()

    call_args = mock_encode.call_args
    payload = call_args[0][0]
    self.assertEqual(payload['identity'], 'my_dongle_id')

  @patch('openpilot.common.api.get_key_pair')
  @patch('openpilot.common.api.jwt.encode')
  def test_get_token_payload_contains_timestamps(self, mock_encode, mock_get_key_pair):
    """Test get_token payload contains timing claims."""
    mock_get_key_pair.return_value = ("RS256", "private_key", "public_key")
    mock_encode.return_value = "token"

    api = Api("dongle")
    api.get_token()

    call_args = mock_encode.call_args
    payload = call_args[0][0]
    self.assertIn('nbf', payload)
    self.assertIn('iat', payload)
    self.assertIn('exp', payload)

  @patch('openpilot.common.api.get_key_pair')
  @patch('openpilot.common.api.jwt.encode')
  def test_get_token_with_payload_extra(self, mock_encode, mock_get_key_pair):
    """Test get_token includes extra payload."""
    mock_get_key_pair.return_value = ("RS256", "private_key", "public_key")
    mock_encode.return_value = "token"

    api = Api("dongle")
    api.get_token(payload_extra={'scope': 'read'})

    call_args = mock_encode.call_args
    payload = call_args[0][0]
    self.assertEqual(payload['scope'], 'read')

  @patch('openpilot.common.api.get_key_pair')
  @patch('openpilot.common.api.jwt.encode')
  def test_get_token_bytes_decoded(self, mock_encode, mock_get_key_pair):
    """Test get_token decodes bytes to string."""
    mock_get_key_pair.return_value = ("RS256", "private_key", "public_key")
    mock_encode.return_value = b"token_bytes"

    api = Api("dongle")
    token = api.get_token()

    self.assertEqual(token, "token_bytes")
    self.assertIsInstance(token, str)


class TestApiGet(unittest.TestCase):
  """Test api_get function."""

  @patch('openpilot.common.api.requests.request')
  @patch('openpilot.common.api.get_version')
  def test_api_get_constructs_url(self, mock_version, mock_request):
    """Test api_get constructs proper URL."""
    mock_version.return_value = "0.9.7"
    mock_request.return_value = MagicMock()

    api_get("v1/test")

    mock_request.assert_called_once()
    call_args = mock_request.call_args
    self.assertIn(API_HOST + "/v1/test", call_args[1].get('url', call_args[0][1]))

  @patch('openpilot.common.api.requests.request')
  @patch('openpilot.common.api.get_version')
  def test_api_get_includes_user_agent(self, mock_version, mock_request):
    """Test api_get includes User-Agent header."""
    mock_version.return_value = "0.9.7"
    mock_request.return_value = MagicMock()

    api_get("endpoint")

    call_args = mock_request.call_args
    headers = call_args[1].get('headers', {})
    self.assertIn('User-Agent', headers)
    self.assertIn("openpilot-0.9.7", headers['User-Agent'])

  @patch('openpilot.common.api.requests.request')
  @patch('openpilot.common.api.get_version')
  def test_api_get_with_access_token(self, mock_version, mock_request):
    """Test api_get includes Authorization header."""
    mock_version.return_value = "0.9.7"
    mock_request.return_value = MagicMock()

    api_get("endpoint", access_token="my_token")

    call_args = mock_request.call_args
    headers = call_args[1].get('headers', {})
    self.assertEqual(headers['Authorization'], "JWT my_token")

  @patch('openpilot.common.api.requests.request')
  @patch('openpilot.common.api.get_version')
  def test_api_get_without_access_token(self, mock_version, mock_request):
    """Test api_get without Authorization header."""
    mock_version.return_value = "0.9.7"
    mock_request.return_value = MagicMock()

    api_get("endpoint")

    call_args = mock_request.call_args
    headers = call_args[1].get('headers', {})
    self.assertNotIn('Authorization', headers)

  @patch('openpilot.common.api.requests.request')
  @patch('openpilot.common.api.get_version')
  def test_api_get_with_session(self, mock_version, mock_request):
    """Test api_get uses provided session."""
    mock_version.return_value = "0.9.7"
    mock_session = MagicMock()

    api_get("endpoint", session=mock_session)

    mock_session.request.assert_called_once()
    mock_request.assert_not_called()

  @patch('openpilot.common.api.requests.request')
  @patch('openpilot.common.api.get_version')
  def test_api_get_passes_params(self, mock_version, mock_request):
    """Test api_get passes query parameters."""
    mock_version.return_value = "0.9.7"
    mock_request.return_value = MagicMock()

    api_get("endpoint", foo="bar", count=5)

    call_args = mock_request.call_args
    params = call_args[1].get('params', {})
    self.assertEqual(params['foo'], 'bar')
    self.assertEqual(params['count'], 5)


class TestGetKeyPair(unittest.TestCase):
  """Test get_key_pair function."""

  @patch('openpilot.common.api.Paths.persist_root')
  def test_get_key_pair_returns_none_when_no_keys(self, mock_persist):
    """Test get_key_pair returns None tuple when no keys exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_persist.return_value = tmpdir

      result = get_key_pair()

      self.assertEqual(result, (None, None, None))

  @patch('openpilot.common.api.Paths.persist_root')
  def test_get_key_pair_reads_rsa_key(self, mock_persist):
    """Test get_key_pair reads RSA key pair."""
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_persist.return_value = tmpdir
      comma_dir = os.path.join(tmpdir, "comma")
      os.makedirs(comma_dir)

      # Create key files
      with open(os.path.join(comma_dir, "id_rsa"), 'w') as f:
        f.write("private_rsa_key")
      with open(os.path.join(comma_dir, "id_rsa.pub"), 'w') as f:
        f.write("public_rsa_key")

      alg, private, public = get_key_pair()

      self.assertEqual(alg, "RS256")
      self.assertEqual(private, "private_rsa_key")
      self.assertEqual(public, "public_rsa_key")

  @patch('openpilot.common.api.Paths.persist_root')
  def test_get_key_pair_reads_ecdsa_key(self, mock_persist):
    """Test get_key_pair reads ECDSA key pair."""
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_persist.return_value = tmpdir
      comma_dir = os.path.join(tmpdir, "comma")
      os.makedirs(comma_dir)

      # Create key files
      with open(os.path.join(comma_dir, "id_ecdsa"), 'w') as f:
        f.write("private_ecdsa_key")
      with open(os.path.join(comma_dir, "id_ecdsa.pub"), 'w') as f:
        f.write("public_ecdsa_key")

      alg, private, public = get_key_pair()

      self.assertEqual(alg, "ES256")
      self.assertEqual(private, "private_ecdsa_key")
      self.assertEqual(public, "public_ecdsa_key")

  @patch('openpilot.common.api.Paths.persist_root')
  def test_get_key_pair_prefers_rsa(self, mock_persist):
    """Test get_key_pair prefers RSA over ECDSA when both exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_persist.return_value = tmpdir
      comma_dir = os.path.join(tmpdir, "comma")
      os.makedirs(comma_dir)

      # Create both key pairs
      for key_type in ["id_rsa", "id_ecdsa"]:
        with open(os.path.join(comma_dir, key_type), 'w') as f:
          f.write(f"private_{key_type}")
        with open(os.path.join(comma_dir, f"{key_type}.pub"), 'w') as f:
          f.write(f"public_{key_type}")

      alg, private, public = get_key_pair()

      # RSA should be preferred (it's first in KEYS dict)
      self.assertEqual(alg, "RS256")

  @patch('openpilot.common.api.Paths.persist_root')
  def test_get_key_pair_requires_both_files(self, mock_persist):
    """Test get_key_pair requires both private and public key."""
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_persist.return_value = tmpdir
      comma_dir = os.path.join(tmpdir, "comma")
      os.makedirs(comma_dir)

      # Create only private key
      with open(os.path.join(comma_dir, "id_rsa"), 'w') as f:
        f.write("private_key")

      result = get_key_pair()

      # Should return None tuple because public key is missing
      self.assertEqual(result, (None, None, None))


if __name__ == '__main__':
  unittest.main()
