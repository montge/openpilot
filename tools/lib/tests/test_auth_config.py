"""Tests for tools/lib/auth_config.py - authentication configuration."""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from openpilot.tools.lib.auth_config import (
  MissingAuthConfigError, get_token, set_token, clear_token,
)


class TestMissingAuthConfigError(unittest.TestCase):
  """Test MissingAuthConfigError exception."""

  def test_is_exception(self):
    """Test MissingAuthConfigError is an Exception."""
    self.assertTrue(issubclass(MissingAuthConfigError, Exception))

  def test_can_raise(self):
    """Test can raise MissingAuthConfigError."""
    with self.assertRaises(MissingAuthConfigError):
      raise MissingAuthConfigError("test message")

  def test_has_message(self):
    """Test MissingAuthConfigError carries message."""
    try:
      raise MissingAuthConfigError("test message")
    except MissingAuthConfigError as e:
      self.assertEqual(str(e), "test message")


class TestGetToken(unittest.TestCase):
  """Test get_token function."""

  @patch('openpilot.tools.lib.auth_config.Paths.config_root')
  def test_returns_token_from_file(self, mock_config_root):
    """Test get_token returns token from auth.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_config_root.return_value = tmpdir
      auth_file = os.path.join(tmpdir, 'auth.json')
      with open(auth_file, 'w') as f:
        json.dump({'access_token': 'my_secret_token'}, f)

      result = get_token()

      self.assertEqual(result, 'my_secret_token')

  @patch('openpilot.tools.lib.auth_config.Paths.config_root')
  def test_returns_none_when_file_missing(self, mock_config_root):
    """Test get_token returns None when auth.json doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_config_root.return_value = tmpdir

      result = get_token()

      self.assertIsNone(result)

  @patch('openpilot.tools.lib.auth_config.Paths.config_root')
  def test_returns_none_on_invalid_json(self, mock_config_root):
    """Test get_token returns None on invalid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_config_root.return_value = tmpdir
      auth_file = os.path.join(tmpdir, 'auth.json')
      with open(auth_file, 'w') as f:
        f.write('not valid json')

      result = get_token()

      self.assertIsNone(result)

  @patch('openpilot.tools.lib.auth_config.Paths.config_root')
  def test_returns_none_on_missing_key(self, mock_config_root):
    """Test get_token returns None when access_token key missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_config_root.return_value = tmpdir
      auth_file = os.path.join(tmpdir, 'auth.json')
      with open(auth_file, 'w') as f:
        json.dump({'other_key': 'value'}, f)

      result = get_token()

      self.assertIsNone(result)


class TestSetToken(unittest.TestCase):
  """Test set_token function."""

  @patch('openpilot.tools.lib.auth_config.Paths.config_root')
  def test_creates_auth_file(self, mock_config_root):
    """Test set_token creates auth.json file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_config_root.return_value = tmpdir

      set_token('new_token')

      auth_file = os.path.join(tmpdir, 'auth.json')
      self.assertTrue(os.path.exists(auth_file))

  @patch('openpilot.tools.lib.auth_config.Paths.config_root')
  def test_writes_token(self, mock_config_root):
    """Test set_token writes token to file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_config_root.return_value = tmpdir

      set_token('my_new_token')

      auth_file = os.path.join(tmpdir, 'auth.json')
      with open(auth_file) as f:
        data = json.load(f)

      self.assertEqual(data['access_token'], 'my_new_token')

  @patch('openpilot.tools.lib.auth_config.Paths.config_root')
  def test_creates_config_dir(self, mock_config_root):
    """Test set_token creates config directory if needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
      config_dir = os.path.join(tmpdir, 'subdir')
      mock_config_root.return_value = config_dir

      set_token('token')

      self.assertTrue(os.path.isdir(config_dir))

  @patch('openpilot.tools.lib.auth_config.Paths.config_root')
  def test_overwrites_existing_token(self, mock_config_root):
    """Test set_token overwrites existing token."""
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_config_root.return_value = tmpdir

      set_token('first_token')
      set_token('second_token')

      auth_file = os.path.join(tmpdir, 'auth.json')
      with open(auth_file) as f:
        data = json.load(f)

      self.assertEqual(data['access_token'], 'second_token')


class TestClearToken(unittest.TestCase):
  """Test clear_token function."""

  @patch('openpilot.tools.lib.auth_config.Paths.config_root')
  def test_removes_auth_file(self, mock_config_root):
    """Test clear_token removes auth.json file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_config_root.return_value = tmpdir
      auth_file = os.path.join(tmpdir, 'auth.json')
      with open(auth_file, 'w') as f:
        json.dump({'access_token': 'token'}, f)

      clear_token()

      self.assertFalse(os.path.exists(auth_file))

  @patch('openpilot.tools.lib.auth_config.Paths.config_root')
  def test_does_not_raise_on_missing_file(self, mock_config_root):
    """Test clear_token doesn't raise when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_config_root.return_value = tmpdir

      # Should not raise
      clear_token()


class TestTokenRoundtrip(unittest.TestCase):
  """Test token get/set/clear integration."""

  @patch('openpilot.tools.lib.auth_config.Paths.config_root')
  def test_set_then_get(self, mock_config_root):
    """Test set_token then get_token returns same value."""
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_config_root.return_value = tmpdir

      set_token('roundtrip_token')
      result = get_token()

      self.assertEqual(result, 'roundtrip_token')

  @patch('openpilot.tools.lib.auth_config.Paths.config_root')
  def test_set_clear_get(self, mock_config_root):
    """Test set_token, clear_token, then get_token returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
      mock_config_root.return_value = tmpdir

      set_token('token_to_clear')
      clear_token()
      result = get_token()

      self.assertIsNone(result)


if __name__ == '__main__':
  unittest.main()
