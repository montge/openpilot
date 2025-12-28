"""Tests for selfdrive/locationd/models/constants.py - observation kinds."""
import unittest

from openpilot.selfdrive.locationd.models.constants import (
  ObservationKind, SAT_OBS, GENERATED_DIR,
)


class TestGeneratedDir(unittest.TestCase):
  """Test GENERATED_DIR constant."""

  def test_generated_dir_path(self):
    """Test GENERATED_DIR is an absolute path."""
    self.assertTrue(GENERATED_DIR.startswith('/'))

  def test_generated_dir_ends_with_generated(self):
    """Test GENERATED_DIR ends with 'generated'."""
    self.assertTrue(GENERATED_DIR.endswith('generated'))


class TestObservationKindValues(unittest.TestCase):
  """Test ObservationKind enum values."""

  def test_unknown_is_zero(self):
    """Test UNKNOWN is 0."""
    self.assertEqual(ObservationKind.UNKNOWN, 0)

  def test_no_observation_is_one(self):
    """Test NO_OBSERVATION is 1."""
    self.assertEqual(ObservationKind.NO_OBSERVATION, 1)

  def test_gps_ned(self):
    """Test GPS_NED value."""
    self.assertEqual(ObservationKind.GPS_NED, 2)

  def test_odometric_speed(self):
    """Test ODOMETRIC_SPEED value."""
    self.assertEqual(ObservationKind.ODOMETRIC_SPEED, 3)

  def test_speed(self):
    """Test SPEED value."""
    self.assertEqual(ObservationKind.SPEED, 8)

  def test_ecef_pos(self):
    """Test ECEF_POS value."""
    self.assertEqual(ObservationKind.ECEF_POS, 12)

  def test_pseudorange_gps(self):
    """Test PSEUDORANGE_GPS value."""
    self.assertEqual(ObservationKind.PSEUDORANGE_GPS, 6)

  def test_pseudorange_glonass(self):
    """Test PSEUDORANGE_GLONASS value."""
    self.assertEqual(ObservationKind.PSEUDORANGE_GLONASS, 20)

  def test_road_frame_values(self):
    """Test ROAD_FRAME observation values."""
    self.assertEqual(ObservationKind.ROAD_FRAME_XY_SPEED, 24)
    self.assertEqual(ObservationKind.ROAD_FRAME_YAW_RATE, 25)
    self.assertEqual(ObservationKind.ROAD_FRAME_X_SPEED, 30)

  def test_steer_angle(self):
    """Test STEER_ANGLE value."""
    self.assertEqual(ObservationKind.STEER_ANGLE, 26)

  def test_stiffness(self):
    """Test STIFFNESS value."""
    self.assertEqual(ObservationKind.STIFFNESS, 28)

  def test_steer_ratio(self):
    """Test STEER_RATIO value."""
    self.assertEqual(ObservationKind.STEER_RATIO, 29)

  def test_road_roll(self):
    """Test ROAD_ROLL value."""
    self.assertEqual(ObservationKind.ROAD_ROLL, 31)


class TestObservationKindNames(unittest.TestCase):
  """Test ObservationKind names list."""

  def test_names_is_list(self):
    """Test names is a list."""
    self.assertIsInstance(ObservationKind.names, list)

  def test_names_has_entries(self):
    """Test names list has entries."""
    self.assertGreater(len(ObservationKind.names), 0)

  def test_unknown_name(self):
    """Test name for UNKNOWN."""
    self.assertEqual(ObservationKind.names[0], 'Unknown')

  def test_no_observation_name(self):
    """Test name for NO_OBSERVATION."""
    self.assertEqual(ObservationKind.names[1], 'No observation')

  def test_gps_ned_name(self):
    """Test name for GPS_NED."""
    self.assertEqual(ObservationKind.names[2], 'GPS NED')

  def test_speed_name(self):
    """Test name for SPEED."""
    self.assertEqual(ObservationKind.names[8], 'Speed')


class TestObservationKindToString(unittest.TestCase):
  """Test ObservationKind.to_string method."""

  def test_to_string_unknown(self):
    """Test to_string for UNKNOWN."""
    result = ObservationKind.to_string(ObservationKind.UNKNOWN)
    self.assertEqual(result, 'Unknown')

  def test_to_string_speed(self):
    """Test to_string for SPEED."""
    result = ObservationKind.to_string(ObservationKind.SPEED)
    self.assertEqual(result, 'Speed')

  def test_to_string_gps_ned(self):
    """Test to_string for GPS_NED."""
    result = ObservationKind.to_string(ObservationKind.GPS_NED)
    self.assertEqual(result, 'GPS NED')


class TestSatObs(unittest.TestCase):
  """Test SAT_OBS constant."""

  def test_sat_obs_is_list(self):
    """Test SAT_OBS is a list."""
    self.assertIsInstance(SAT_OBS, list)

  def test_sat_obs_contains_gps_pseudorange(self):
    """Test SAT_OBS contains GPS pseudorange."""
    self.assertIn(ObservationKind.PSEUDORANGE_GPS, SAT_OBS)

  def test_sat_obs_contains_gps_pseudorange_rate(self):
    """Test SAT_OBS contains GPS pseudorange rate."""
    self.assertIn(ObservationKind.PSEUDORANGE_RATE_GPS, SAT_OBS)

  def test_sat_obs_contains_glonass_pseudorange(self):
    """Test SAT_OBS contains GLONASS pseudorange."""
    self.assertIn(ObservationKind.PSEUDORANGE_GLONASS, SAT_OBS)

  def test_sat_obs_contains_glonass_pseudorange_rate(self):
    """Test SAT_OBS contains GLONASS pseudorange rate."""
    self.assertIn(ObservationKind.PSEUDORANGE_RATE_GLONASS, SAT_OBS)

  def test_sat_obs_has_four_entries(self):
    """Test SAT_OBS has exactly 4 entries."""
    self.assertEqual(len(SAT_OBS), 4)


if __name__ == '__main__':
  unittest.main()
