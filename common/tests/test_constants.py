"""Tests for common/constants.py - conversion constants."""
import math

import numpy as np
import pytest

from openpilot.common.constants import CV, ACCELERATION_DUE_TO_GRAVITY


class TestSpeedConversions:
  """Test speed conversion constants."""

  def test_mph_to_kph(self):
    """Test MPH to KPH conversion."""
    # 1 mph = 1.609344 kph
    assert CV.MPH_TO_KPH == pytest.approx(1.609344, rel=1e-6)

  def test_kph_to_mph(self):
    """Test KPH to MPH conversion is inverse."""
    assert CV.KPH_TO_MPH == pytest.approx(1.0 / CV.MPH_TO_KPH, rel=1e-10)

  def test_mph_kph_roundtrip(self):
    """Test MPH -> KPH -> MPH roundtrip."""
    mph = 60.0
    kph = mph * CV.MPH_TO_KPH
    mph_back = kph * CV.KPH_TO_MPH
    assert mph_back == pytest.approx(mph, rel=1e-10)

  def test_ms_to_kph(self):
    """Test m/s to KPH conversion."""
    # 1 m/s = 3.6 kph
    assert CV.MS_TO_KPH == 3.6

  def test_kph_to_ms(self):
    """Test KPH to m/s conversion is inverse."""
    assert CV.KPH_TO_MS == pytest.approx(1.0 / CV.MS_TO_KPH, rel=1e-10)

  def test_ms_kph_roundtrip(self):
    """Test m/s -> KPH -> m/s roundtrip."""
    ms = 30.0
    kph = ms * CV.MS_TO_KPH
    ms_back = kph * CV.KPH_TO_MS
    assert ms_back == pytest.approx(ms, rel=1e-10)

  def test_ms_to_mph(self):
    """Test m/s to MPH conversion consistency."""
    expected = CV.MS_TO_KPH * CV.KPH_TO_MPH
    assert CV.MS_TO_MPH == pytest.approx(expected, rel=1e-10)

  def test_mph_to_ms(self):
    """Test MPH to m/s conversion consistency."""
    expected = CV.MPH_TO_KPH * CV.KPH_TO_MS
    assert CV.MPH_TO_MS == pytest.approx(expected, rel=1e-10)

  def test_ms_mph_roundtrip(self):
    """Test m/s -> MPH -> m/s roundtrip."""
    ms = 20.0
    mph = ms * CV.MS_TO_MPH
    ms_back = mph * CV.MPH_TO_MS
    assert ms_back == pytest.approx(ms, rel=1e-10)

  def test_ms_to_knots(self):
    """Test m/s to knots conversion."""
    # 1 m/s = 1.9438 knots (approx)
    assert CV.MS_TO_KNOTS == pytest.approx(1.9438, rel=1e-4)

  def test_knots_to_ms(self):
    """Test knots to m/s conversion is inverse."""
    assert CV.KNOTS_TO_MS == pytest.approx(1.0 / CV.MS_TO_KNOTS, rel=1e-10)

  def test_knots_roundtrip(self):
    """Test m/s -> knots -> m/s roundtrip."""
    ms = 15.0
    knots = ms * CV.MS_TO_KNOTS
    ms_back = knots * CV.KNOTS_TO_MS
    assert ms_back == pytest.approx(ms, rel=1e-10)


class TestAngleConversions:
  """Test angle conversion constants."""

  def test_deg_to_rad(self):
    """Test degrees to radians conversion."""
    # 180 degrees = pi radians
    assert 180.0 * CV.DEG_TO_RAD == pytest.approx(np.pi, rel=1e-10)

  def test_rad_to_deg(self):
    """Test radians to degrees conversion is inverse."""
    assert CV.RAD_TO_DEG == pytest.approx(1.0 / CV.DEG_TO_RAD, rel=1e-10)

  def test_deg_rad_roundtrip(self):
    """Test degrees -> radians -> degrees roundtrip."""
    deg = 45.0
    rad = deg * CV.DEG_TO_RAD
    deg_back = rad * CV.RAD_TO_DEG
    assert deg_back == pytest.approx(deg, rel=1e-10)

  def test_90_degrees(self):
    """Test 90 degrees equals pi/2 radians."""
    assert 90.0 * CV.DEG_TO_RAD == pytest.approx(np.pi / 2, rel=1e-10)

  def test_360_degrees(self):
    """Test 360 degrees equals 2*pi radians."""
    assert 360.0 * CV.DEG_TO_RAD == pytest.approx(2 * np.pi, rel=1e-10)

  def test_pi_radians(self):
    """Test pi radians equals 180 degrees."""
    assert np.pi * CV.RAD_TO_DEG == pytest.approx(180.0, rel=1e-10)

  def test_uses_numpy_pi(self):
    """Test that numpy pi is used for precision."""
    assert CV.DEG_TO_RAD == np.pi / 180.0


class TestMassConversions:
  """Test mass conversion constants."""

  def test_lb_to_kg(self):
    """Test pounds to kg conversion."""
    # 1 lb = 0.453592 kg
    assert CV.LB_TO_KG == pytest.approx(0.453592, rel=1e-6)

  def test_100_lb_to_kg(self):
    """Test 100 lb conversion."""
    kg = 100 * CV.LB_TO_KG
    assert kg == pytest.approx(45.3592, rel=1e-4)


class TestGravity:
  """Test gravity constant."""

  def test_gravity_value(self):
    """Test acceleration due to gravity value."""
    assert ACCELERATION_DUE_TO_GRAVITY == 9.81

  def test_gravity_positive(self):
    """Test gravity is positive."""
    assert ACCELERATION_DUE_TO_GRAVITY > 0


class TestConversionConsistency:
  """Test consistency across all conversions."""

  def test_all_inversions_correct(self):
    """Test all inverse conversions multiply to 1."""
    pairs = [
      (CV.MPH_TO_KPH, CV.KPH_TO_MPH),
      (CV.MS_TO_KPH, CV.KPH_TO_MS),
      (CV.MS_TO_KNOTS, CV.KNOTS_TO_MS),
      (CV.DEG_TO_RAD, CV.RAD_TO_DEG),
    ]
    for forward, backward in pairs:
      assert forward * backward == pytest.approx(1.0, rel=1e-10)

  def test_mph_ms_chain(self):
    """Test mph -> kph -> ms chain equals direct conversion."""
    via_kph = CV.MPH_TO_KPH * CV.KPH_TO_MS
    assert via_kph == pytest.approx(CV.MPH_TO_MS, rel=1e-10)

  def test_ms_mph_chain(self):
    """Test ms -> kph -> mph chain equals direct conversion."""
    via_kph = CV.MS_TO_KPH * CV.KPH_TO_MPH
    assert via_kph == pytest.approx(CV.MS_TO_MPH, rel=1e-10)

  def test_all_speed_conversions_positive(self):
    """Test all speed conversions are positive."""
    conversions = [
      CV.MPH_TO_KPH, CV.KPH_TO_MPH,
      CV.MS_TO_KPH, CV.KPH_TO_MS,
      CV.MS_TO_MPH, CV.MPH_TO_MS,
      CV.MS_TO_KNOTS, CV.KNOTS_TO_MS,
    ]
    for conv in conversions:
      assert conv > 0

  def test_all_angle_conversions_positive(self):
    """Test all angle conversions are positive."""
    assert CV.DEG_TO_RAD > 0
    assert CV.RAD_TO_DEG > 0

  def test_mass_conversion_positive(self):
    """Test mass conversion is positive."""
    assert CV.LB_TO_KG > 0
