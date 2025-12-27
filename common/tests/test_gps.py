"""Tests for common/gps.py - GPS utility functions."""
import unittest
from unittest.mock import MagicMock

from openpilot.common.gps import get_gps_location_service


class TestGetGpsLocationService(unittest.TestCase):
  """Test get_gps_location_service function."""

  def test_returns_external_when_ublox_available(self):
    """Test returns gpsLocationExternal when Ublox is available."""
    params = MagicMock()
    params.get_bool.return_value = True

    result = get_gps_location_service(params)

    self.assertEqual(result, "gpsLocationExternal")
    params.get_bool.assert_called_once_with("UbloxAvailable")

  def test_returns_internal_when_ublox_not_available(self):
    """Test returns gpsLocation when Ublox is not available."""
    params = MagicMock()
    params.get_bool.return_value = False

    result = get_gps_location_service(params)

    self.assertEqual(result, "gpsLocation")
    params.get_bool.assert_called_once_with("UbloxAvailable")

  def test_checks_correct_param(self):
    """Test checks UbloxAvailable param."""
    params = MagicMock()
    params.get_bool.return_value = False

    get_gps_location_service(params)

    params.get_bool.assert_called_with("UbloxAvailable")


if __name__ == '__main__':
  unittest.main()
