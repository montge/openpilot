"""Tests for tools/lib/sanitizer.py - route sanitization utilities."""
import unittest
from unittest.mock import MagicMock

from openpilot.tools.lib.sanitizer import (
  sanitize_vin, sanitize_msg, sanitize, PRESERVE_SERVICES,
)


class TestSanitizeVin(unittest.TestCase):
  """Test sanitize_vin function."""

  def test_sanitize_vin_replaces_last_6_chars(self):
    """Test VIN last 6 characters are replaced with X."""
    vin = "1HGBH41JXMN109186"
    result = sanitize_vin(vin)

    self.assertEqual(result, "1HGBH41JXMNXXXXXX")

  def test_sanitize_vin_preserves_prefix(self):
    """Test VIN prefix is preserved."""
    vin = "ABCDEFGHIJK123456"
    result = sanitize_vin(vin)

    self.assertEqual(result[:11], "ABCDEFGHIJK")

  def test_sanitize_vin_length_preserved(self):
    """Test sanitized VIN has same length."""
    vin = "1HGBH41JXMN109186"
    result = sanitize_vin(vin)

    self.assertEqual(len(result), len(vin))

  def test_sanitize_vin_17_chars(self):
    """Test standard 17-character VIN."""
    vin = "X" * 17
    result = sanitize_vin(vin)

    self.assertEqual(result, "X" * 11 + "X" * 6)

  def test_sanitize_vin_short(self):
    """Test shorter VIN (edge case)."""
    vin = "ABC123456"  # 9 chars
    result = sanitize_vin(vin)

    self.assertEqual(result, "ABCXXXXXX")
    self.assertEqual(len(result), 9)


class TestSanitizeMsg(unittest.TestCase):
  """Test sanitize_msg function."""

  def test_sanitize_car_params_message(self):
    """Test carParams message VIN is sanitized."""
    msg = MagicMock()
    msg.which.return_value = "carParams"

    builder = MagicMock()
    builder.carParams.carVin = "1HGBH41JXMN109186"
    msg.as_builder.return_value = builder

    reader = MagicMock()
    builder.as_reader.return_value = reader

    result = sanitize_msg(msg)

    # Should have called as_builder and modified carVin
    msg.as_builder.assert_called_once()
    self.assertEqual(builder.carParams.carVin, "1HGBH41JXMNXXXXXX")
    builder.as_reader.assert_called_once()
    self.assertEqual(result, reader)

  def test_non_car_params_message_unchanged(self):
    """Test non-carParams messages are unchanged."""
    msg = MagicMock()
    msg.which.return_value = "can"

    result = sanitize_msg(msg)

    # Should return same message without modification
    self.assertEqual(result, msg)
    msg.as_builder.assert_not_called()

  def test_other_message_types_unchanged(self):
    """Test various other message types are unchanged."""
    for msg_type in ["pandaStates", "controlsState", "liveCalibration"]:
      msg = MagicMock()
      msg.which.return_value = msg_type

      result = sanitize_msg(msg)

      self.assertEqual(result, msg)


class TestPreserveServices(unittest.TestCase):
  """Test PRESERVE_SERVICES constant."""

  def test_contains_can(self):
    """Test can service is preserved."""
    self.assertIn("can", PRESERVE_SERVICES)

  def test_contains_car_params(self):
    """Test carParams service is preserved."""
    self.assertIn("carParams", PRESERVE_SERVICES)

  def test_contains_panda_states(self):
    """Test pandaStates service is preserved."""
    self.assertIn("pandaStates", PRESERVE_SERVICES)

  def test_contains_deprecated_panda_state(self):
    """Test deprecated pandaState is preserved."""
    self.assertIn("pandaStateDEPRECATED", PRESERVE_SERVICES)

  def test_does_not_contain_controls_state(self):
    """Test controlsState is not preserved (contains private data)."""
    self.assertNotIn("controlsState", PRESERVE_SERVICES)


class TestSanitize(unittest.TestCase):
  """Test sanitize function."""

  def test_filters_to_preserve_services(self):
    """Test sanitize filters to only preserved services."""
    msg_can = MagicMock()
    msg_can.which.return_value = "can"

    msg_controls = MagicMock()
    msg_controls.which.return_value = "controlsState"

    msg_panda = MagicMock()
    msg_panda.which.return_value = "pandaStates"

    lr = [msg_can, msg_controls, msg_panda]

    result = list(sanitize(lr))

    # Should only include can and pandaStates
    self.assertEqual(len(result), 2)
    self.assertIn(msg_can, result)
    self.assertIn(msg_panda, result)
    self.assertNotIn(msg_controls, result)

  def test_sanitizes_car_params(self):
    """Test sanitize sanitizes carParams messages."""
    msg_car_params = MagicMock()
    msg_car_params.which.return_value = "carParams"

    builder = MagicMock()
    builder.carParams.carVin = "1HGBH41JXMN109186"
    msg_car_params.as_builder.return_value = builder

    reader = MagicMock()
    builder.as_reader.return_value = reader

    lr = [msg_car_params]

    result = list(sanitize(lr))

    # Should sanitize VIN
    self.assertEqual(builder.carParams.carVin, "1HGBH41JXMNXXXXXX")
    self.assertEqual(result[0], reader)

  def test_empty_log(self):
    """Test sanitize with empty log."""
    lr = []

    result = list(sanitize(lr))

    self.assertEqual(result, [])

  def test_preserves_order(self):
    """Test sanitize preserves message order."""
    messages = []
    for i, svc in enumerate(["can", "pandaStates", "can"]):
      msg = MagicMock()
      msg.which.return_value = svc
      msg.order = i
      messages.append(msg)

    result = list(sanitize(messages))

    self.assertEqual(len(result), 3)
    for i, msg in enumerate(result):
      self.assertEqual(msg.order, i)


if __name__ == '__main__':
  unittest.main()
