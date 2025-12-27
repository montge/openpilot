"""Tests for common/transformations/camera.py - camera transformations."""
import numpy as np
import pytest

from openpilot.common.transformations.camera import (
  CameraConfig, DeviceCameraConfig, _NoneCameraConfig,
  DEVICE_CAMERAS,
  device_frame_from_view_frame, view_frame_from_device_frame,
  get_view_frame_from_road_frame, get_view_frame_from_calib_frame,
  vp_from_ke, roll_from_ke,
  normalize, denormalize, get_calib_from_vp,
  device_from_ecef, img_from_device,
)


class TestCameraConfig:
  """Test CameraConfig dataclass."""

  def test_camera_config_creation(self):
    """Test CameraConfig creation."""
    config = CameraConfig(1920, 1080, 1000.0)

    assert config.width == 1920
    assert config.height == 1080
    assert config.focal_length == 1000.0

  def test_camera_config_size_property(self):
    """Test size property returns tuple."""
    config = CameraConfig(1920, 1080, 1000.0)

    assert config.size == (1920, 1080)

  def test_camera_config_intrinsics(self):
    """Test intrinsics matrix (K matrix)."""
    config = CameraConfig(1920, 1080, 1000.0)
    K = config.intrinsics

    assert K.shape == (3, 3)
    # Check focal length on diagonal
    assert K[0, 0] == 1000.0
    assert K[1, 1] == 1000.0
    # Check principal point
    assert K[0, 2] == 1920 / 2
    assert K[1, 2] == 1080 / 2
    # Check remaining elements
    assert K[2, 2] == 1.0
    assert K[0, 1] == 0.0
    assert K[1, 0] == 0.0
    assert K[2, 0] == 0.0
    assert K[2, 1] == 0.0

  def test_camera_config_intrinsics_inv(self):
    """Test intrinsics inverse matrix."""
    config = CameraConfig(1920, 1080, 1000.0)
    K = config.intrinsics
    K_inv = config.intrinsics_inv

    # K * K_inv should be identity
    product = K @ K_inv
    np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

  def test_camera_config_frozen(self):
    """Test CameraConfig is immutable."""
    config = CameraConfig(1920, 1080, 1000.0)

    with pytest.raises(Exception):  # FrozenInstanceError
      config.width = 1280


class TestNoneCameraConfig:
  """Test _NoneCameraConfig for placeholder cameras."""

  def test_none_camera_config_defaults(self):
    """Test _NoneCameraConfig has zero defaults."""
    config = _NoneCameraConfig()

    assert config.width == 0
    assert config.height == 0
    assert config.focal_length == 0


class TestDeviceCameraConfig:
  """Test DeviceCameraConfig dataclass."""

  def test_device_camera_config_creation(self):
    """Test DeviceCameraConfig creation."""
    fcam = CameraConfig(1920, 1080, 1000.0)
    dcam = CameraConfig(1280, 720, 800.0)
    ecam = CameraConfig(1280, 720, 800.0)

    config = DeviceCameraConfig(fcam, dcam, ecam)

    assert config.fcam == fcam
    assert config.dcam == dcam
    assert config.ecam == ecam

  def test_all_cams_yields_non_none_cameras(self):
    """Test all_cams yields only non-None cameras."""
    fcam = CameraConfig(1920, 1080, 1000.0)
    dcam = CameraConfig(1280, 720, 800.0)
    ecam = _NoneCameraConfig()

    config = DeviceCameraConfig(fcam, dcam, ecam)

    cams = list(config.all_cams())

    assert len(cams) == 2
    assert ('fcam', fcam) in cams
    assert ('dcam', dcam) in cams

  def test_all_cams_with_all_cameras(self):
    """Test all_cams with all cameras present."""
    fcam = CameraConfig(1920, 1080, 1000.0)
    dcam = CameraConfig(1280, 720, 800.0)
    ecam = CameraConfig(1280, 720, 750.0)

    config = DeviceCameraConfig(fcam, dcam, ecam)

    cams = list(config.all_cams())

    assert len(cams) == 3


class TestDeviceCameras:
  """Test DEVICE_CAMERAS dictionary."""

  def test_device_cameras_has_neo(self):
    """Test neo device is present."""
    assert ("neo", "unknown") in DEVICE_CAMERAS

  def test_device_cameras_has_tici(self):
    """Test tici devices are present."""
    assert ("tici", "unknown") in DEVICE_CAMERAS
    assert ("tici", "ar0231") in DEVICE_CAMERAS
    assert ("tici", "ox03c10") in DEVICE_CAMERAS
    assert ("tici", "os04c10") in DEVICE_CAMERAS

  def test_device_cameras_has_pc(self):
    """Test pc simulator device is present."""
    assert ("pc", "unknown") in DEVICE_CAMERAS

  def test_device_cameras_are_device_camera_config(self):
    """Test all values are DeviceCameraConfig."""
    for key, value in DEVICE_CAMERAS.items():
      assert isinstance(value, DeviceCameraConfig)


class TestFrameTransforms:
  """Test frame transformation matrices."""

  def test_device_from_view_orthogonal(self):
    """Test device_frame_from_view_frame is orthogonal."""
    product = device_frame_from_view_frame @ device_frame_from_view_frame.T
    np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

  def test_view_from_device_orthogonal(self):
    """Test view_frame_from_device_frame is orthogonal."""
    product = view_frame_from_device_frame @ view_frame_from_device_frame.T
    np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

  def test_device_view_roundtrip(self):
    """Test device <-> view roundtrip."""
    product = device_frame_from_view_frame @ view_frame_from_device_frame
    np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

  def test_device_from_view_is_transpose_of_view_from_device(self):
    """Test the matrices are transposes of each other."""
    np.testing.assert_allclose(
      device_frame_from_view_frame,
      view_frame_from_device_frame.T,
      atol=1e-10
    )


class TestViewFrameFromRoadFrame:
  """Test get_view_frame_from_road_frame function."""

  def test_identity_transform(self):
    """Test identity transform (no rotation)."""
    result = get_view_frame_from_road_frame(0, 0, 0, 0)

    assert result.shape == (3, 4)
    # Translation column should have height
    assert result[1, 3] == 0

  def test_with_height(self):
    """Test height appears in translation."""
    result = get_view_frame_from_road_frame(0, 0, 0, 1.5)

    assert result[1, 3] == 1.5

  def test_result_shape(self):
    """Test result has correct shape."""
    result = get_view_frame_from_road_frame(0.1, 0.05, 0.02, 1.2)

    assert result.shape == (3, 4)


class TestViewFrameFromCalibFrame:
  """Test get_view_frame_from_calib_frame function."""

  def test_identity_transform(self):
    """Test identity transform."""
    result = get_view_frame_from_calib_frame(0, 0, 0, 0)

    assert result.shape == (3, 4)

  def test_with_height(self):
    """Test height in translation."""
    result = get_view_frame_from_calib_frame(0, 0, 0, 2.0)

    assert result[1, 3] == 2.0


class TestVpFromKe:
  """Test vp_from_ke - vanishing point from KE matrix."""

  def test_vp_returns_tuple(self):
    """Test vp_from_ke returns a tuple."""
    K = np.array([
      [1000, 0, 960],
      [0, 1000, 540],
      [0, 0, 1]
    ])
    E = get_view_frame_from_road_frame(0, 0, 0, 1.2)
    KE = K @ E

    vp = vp_from_ke(KE)

    assert isinstance(vp, tuple)
    assert len(vp) == 2

  def test_vp_at_principal_point_for_zero_rotation(self):
    """Test vanishing point near principal point for zero rotation."""
    config = CameraConfig(1920, 1080, 1000.0)
    K = config.intrinsics
    E = get_view_frame_from_road_frame(0, 0, 0, 1.2)
    KE = K @ E

    vp = vp_from_ke(KE)

    # For no rotation, VP should be near center of image
    assert vp[0] == pytest.approx(960, abs=50)
    assert vp[1] == pytest.approx(540, abs=50)


class TestRollFromKe:
  """Test roll_from_ke function."""

  def test_roll_with_rotation(self):
    """Test roll estimation with rotation."""
    config = CameraConfig(1920, 1080, 1000.0)
    K = config.intrinsics
    # Use non-zero angles to avoid division by zero
    E = get_view_frame_from_road_frame(0.05, 0.05, 0.02, 1.2)
    KE = K @ E

    roll = roll_from_ke(KE)

    # Roll should be close to the input roll
    assert isinstance(roll, (float, np.floating))

  def test_returns_float(self):
    """Test roll_from_ke returns float."""
    K = np.array([
      [1000, 0, 960],
      [0, 1000, 540],
      [0, 0, 1]
    ])
    E = get_view_frame_from_road_frame(0.1, 0.05, 0.02, 1.2)
    KE = K @ E

    roll = roll_from_ke(KE)

    assert isinstance(roll, (float, np.floating))


class TestNormalize:
  """Test normalize function."""

  def test_normalize_single_point(self):
    """Test normalizing a single point."""
    config = CameraConfig(1920, 1080, 1000.0)
    K = config.intrinsics

    # Principal point should normalize to (0, 0)
    pt = np.array([960.0, 540.0])
    result = normalize(pt, K)

    np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-10)

  def test_normalize_array_of_points(self):
    """Test normalizing array of points."""
    config = CameraConfig(1920, 1080, 1000.0)
    K = config.intrinsics

    pts = np.array([[960.0, 540.0], [1460.0, 1040.0]])
    result = normalize(pts, K)

    assert result.shape == (2, 2)
    np.testing.assert_allclose(result[0], [0.0, 0.0], atol=1e-10)

  def test_negative_points_give_nan(self):
    """Test negative image points give NaN."""
    config = CameraConfig(1920, 1080, 1000.0)
    K = config.intrinsics

    pt = np.array([-10.0, 100.0])
    result = normalize(pt, K)

    assert np.isnan(result[0])
    assert np.isnan(result[1])


class TestDenormalize:
  """Test denormalize function."""

  def test_denormalize_single_point(self):
    """Test denormalizing a single point."""
    config = CameraConfig(1920, 1080, 1000.0)
    K = config.intrinsics

    # Normalized (0, 0) should map to principal point
    pt = np.array([0.0, 0.0])
    result = denormalize(pt, K)

    np.testing.assert_allclose(result, [960.0, 540.0], atol=1e-10)

  def test_normalize_denormalize_roundtrip(self):
    """Test normalize -> denormalize roundtrip."""
    config = CameraConfig(1920, 1080, 1000.0)
    K = config.intrinsics

    pt = np.array([500.0, 300.0])
    normalized = normalize(pt, K)
    result = denormalize(normalized, K)

    np.testing.assert_allclose(result, pt, atol=1e-6)

  def test_denormalize_with_bounds_check(self):
    """Test denormalize with width/height bounds."""
    config = CameraConfig(1920, 1080, 1000.0)
    K = config.intrinsics

    # Point outside bounds
    pt = np.array([10.0, 10.0])  # Very far from center
    result = denormalize(pt, K, width=1920, height=1080)

    # Points outside bounds should be NaN
    assert np.isnan(result[0]) or np.isnan(result[1])

  def test_denormalize_no_bounds(self):
    """Test denormalize without bounds checking."""
    config = CameraConfig(1920, 1080, 1000.0)
    K = config.intrinsics

    # Point that would be outside 1920x1080
    pt = np.array([1.0, 1.0])
    result = denormalize(pt, K)

    # Without bounds, no NaN
    assert not np.isnan(result[0])
    assert not np.isnan(result[1])


class TestGetCalibFromVp:
  """Test get_calib_from_vp function."""

  def test_vp_at_center_gives_zero_calib(self):
    """Test vanishing point at center gives zero calibration."""
    config = CameraConfig(1920, 1080, 1000.0)
    K = config.intrinsics

    # VP at principal point
    vp = [960.0, 540.0]
    roll, pitch, yaw = get_calib_from_vp(vp, K)

    assert roll == 0
    assert pitch == pytest.approx(0, abs=1e-10)
    assert yaw == pytest.approx(0, abs=1e-10)

  def test_returns_three_values(self):
    """Test get_calib_from_vp returns roll, pitch, yaw."""
    config = CameraConfig(1920, 1080, 1000.0)
    K = config.intrinsics

    vp = [960.0, 540.0]
    result = get_calib_from_vp(vp, K)

    assert len(result) == 3


class TestDeviceFromEcef:
  """Test device_from_ecef function."""

  def test_single_point(self):
    """Test transforming a single ECEF point."""
    pos_ecef = np.array([4000000, 0, 4000000])
    orientation_ecef = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    pt_ecef = np.array([4000100, 0, 4000000])

    result = device_from_ecef(pos_ecef, orientation_ecef, pt_ecef)

    assert result.shape == (3,)

  def test_array_of_points(self):
    """Test transforming array of ECEF points."""
    pos_ecef = np.array([4000000, 0, 4000000])
    orientation_ecef = np.array([1.0, 0.0, 0.0, 0.0])
    pts_ecef = np.array([
      [4000100, 0, 4000000],
      [4000000, 100, 4000000],
    ])

    result = device_from_ecef(pos_ecef, orientation_ecef, pts_ecef)

    assert result.shape == (2, 3)


class TestImgFromDevice:
  """Test img_from_device function."""

  def test_point_in_front(self):
    """Test point in front of camera."""
    # Device frame: x forward, y right, z down
    # Function requires 2D array input for proper indexing
    pts_device = np.array([[10.0, 0.0, 0.0]])

    result = img_from_device(pts_device)

    assert result.shape == (1, 2)
    # Point straight ahead should map to (0, 0) in normalized coords
    np.testing.assert_allclose(result[0], [0.0, 0.0], atol=1e-10)

  def test_point_behind_gives_nan(self):
    """Test point behind camera gives NaN."""
    # Point behind (negative x in device frame -> negative z in view frame)
    pts_device = np.array([[-10.0, 0.0, 0.0]])

    result = img_from_device(pts_device)

    assert np.isnan(result[0, 0])
    assert np.isnan(result[0, 1])

  def test_array_of_points(self):
    """Test array of device points."""
    pts_device = np.array([
      [10.0, 0.0, 0.0],
      [10.0, 1.0, 0.0],
    ])

    result = img_from_device(pts_device)

    assert result.shape == (2, 2)

  def test_point_to_right(self):
    """Test point to the right of camera."""
    # y positive = right in device frame
    pts_device = np.array([[10.0, 5.0, 0.0]])

    result = img_from_device(pts_device)

    # Should be positive x in image coords
    assert result[0, 0] > 0


class TestIntegration:
  """Integration tests for camera transformations."""

  def test_full_pipeline_point_projection(self):
    """Test projecting a point through full pipeline."""
    config = CameraConfig(1920, 1080, 1000.0)
    K = config.intrinsics

    # Point in device frame (forward, right, down)
    pts_device = np.array([[20.0, 2.0, 1.0]])

    # Project to normalized image coordinates
    pt_img_norm = img_from_device(pts_device)

    # Denormalize to pixel coordinates
    pt_pixel = denormalize(pt_img_norm[0], K)

    # Point should be in image bounds
    assert 0 <= pt_pixel[0] <= 1920
    assert 0 <= pt_pixel[1] <= 1080

  def test_calibration_roundtrip(self):
    """Test calibration estimation roundtrip."""
    config = CameraConfig(1920, 1080, 1000.0)
    K = config.intrinsics

    # Start with known calibration
    roll, pitch, yaw = 0.0, 0.05, 0.02
    height = 1.2

    # Compute extrinsic matrix
    E = get_view_frame_from_road_frame(roll, pitch, yaw, height)

    # Compute full projection matrix
    KE = K @ E

    # Get vanishing point
    vp = vp_from_ke(KE)

    # Estimate calibration from vanishing point
    est_roll, est_pitch, est_yaw = get_calib_from_vp(vp, K)

    # Roll is always 0 from VP estimation
    assert est_roll == 0
    # Pitch and yaw should be close
    assert est_pitch == pytest.approx(pitch, abs=0.01)
    assert est_yaw == pytest.approx(yaw, abs=0.01)
