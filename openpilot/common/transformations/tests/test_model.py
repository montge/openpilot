"""Tests for common/transformations/model.py - model transformation constants and utilities."""

import numpy as np

from openpilot.common.transformations.model import (
  SEGNET_SIZE,
  MEDMODEL_INPUT_SIZE,
  MEDMODEL_YUV_SIZE,
  MEDMODEL_CY,
  medmodel_fl,
  medmodel_intrinsics,
  BIGMODEL_INPUT_SIZE,
  BIGMODEL_YUV_SIZE,
  bigmodel_fl,
  bigmodel_intrinsics,
  SBIGMODEL_INPUT_SIZE,
  SBIGMODEL_YUV_SIZE,
  sbigmodel_fl,
  sbigmodel_intrinsics,
  DM_INPUT_SIZE,
  dmonitoringmodel_fl,
  dmonitoringmodel_intrinsics,
  bigmodel_frame_from_calib_frame,
  sbigmodel_frame_from_calib_frame,
  medmodel_frame_from_calib_frame,
  medmodel_frame_from_bigmodel_frame,
  calib_from_medmodel,
  calib_from_sbigmodel,
  get_warp_matrix,
)


class TestModelConstants:
  """Test model size constants."""

  def test_segnet_size(self):
    """Test SEGNET_SIZE is tuple of two integers."""
    assert isinstance(SEGNET_SIZE, tuple)
    assert len(SEGNET_SIZE) == 2
    assert SEGNET_SIZE == (512, 384)

  def test_medmodel_input_size(self):
    """Test MEDMODEL_INPUT_SIZE dimensions."""
    assert MEDMODEL_INPUT_SIZE == (512, 256)

  def test_medmodel_yuv_size(self):
    """Test MEDMODEL_YUV_SIZE is 1.5x height for YUV420."""
    assert MEDMODEL_YUV_SIZE[0] == MEDMODEL_INPUT_SIZE[0]
    assert MEDMODEL_YUV_SIZE[1] == MEDMODEL_INPUT_SIZE[1] * 3 // 2

  def test_bigmodel_input_size(self):
    """Test BIGMODEL_INPUT_SIZE dimensions."""
    assert BIGMODEL_INPUT_SIZE == (1024, 512)

  def test_bigmodel_yuv_size(self):
    """Test BIGMODEL_YUV_SIZE is 1.5x height for YUV420."""
    assert BIGMODEL_YUV_SIZE[0] == BIGMODEL_INPUT_SIZE[0]
    assert BIGMODEL_YUV_SIZE[1] == BIGMODEL_INPUT_SIZE[1] * 3 // 2

  def test_sbigmodel_input_size(self):
    """Test SBIGMODEL_INPUT_SIZE dimensions."""
    assert SBIGMODEL_INPUT_SIZE == (512, 256)

  def test_sbigmodel_yuv_size(self):
    """Test SBIGMODEL_YUV_SIZE is 1.5x height for YUV420."""
    assert SBIGMODEL_YUV_SIZE[0] == SBIGMODEL_INPUT_SIZE[0]
    assert SBIGMODEL_YUV_SIZE[1] == SBIGMODEL_INPUT_SIZE[1] * 3 // 2

  def test_dm_input_size(self):
    """Test DM_INPUT_SIZE dimensions."""
    assert DM_INPUT_SIZE == (1440, 960)

  def test_medmodel_cy_positive(self):
    """Test MEDMODEL_CY is positive."""
    assert MEDMODEL_CY > 0


class TestIntrinsicMatrices:
  """Test camera intrinsic matrices."""

  def test_medmodel_intrinsics_shape(self):
    """Test medmodel_intrinsics is 3x3."""
    assert medmodel_intrinsics.shape == (3, 3)

  def test_medmodel_intrinsics_focal_length(self):
    """Test medmodel focal length in intrinsics."""
    assert medmodel_intrinsics[0, 0] == medmodel_fl
    assert medmodel_intrinsics[1, 1] == medmodel_fl

  def test_medmodel_intrinsics_principal_point(self):
    """Test medmodel principal point is at center x."""
    assert medmodel_intrinsics[0, 2] == 0.5 * MEDMODEL_INPUT_SIZE[0]
    assert medmodel_intrinsics[1, 2] == MEDMODEL_CY

  def test_bigmodel_intrinsics_shape(self):
    """Test bigmodel_intrinsics is 3x3."""
    assert bigmodel_intrinsics.shape == (3, 3)

  def test_bigmodel_intrinsics_focal_length(self):
    """Test bigmodel focal length in intrinsics."""
    assert bigmodel_intrinsics[0, 0] == bigmodel_fl
    assert bigmodel_intrinsics[1, 1] == bigmodel_fl

  def test_sbigmodel_intrinsics_shape(self):
    """Test sbigmodel_intrinsics is 3x3."""
    assert sbigmodel_intrinsics.shape == (3, 3)

  def test_sbigmodel_intrinsics_focal_length(self):
    """Test sbigmodel focal length in intrinsics."""
    assert sbigmodel_intrinsics[0, 0] == sbigmodel_fl
    assert sbigmodel_intrinsics[1, 1] == sbigmodel_fl

  def test_dmonitoringmodel_intrinsics_shape(self):
    """Test dmonitoringmodel_intrinsics is 3x3."""
    assert dmonitoringmodel_intrinsics.shape == (3, 3)

  def test_dmonitoringmodel_intrinsics_focal_length(self):
    """Test dmonitoringmodel focal length in intrinsics."""
    assert dmonitoringmodel_intrinsics[0, 0] == dmonitoringmodel_fl
    assert dmonitoringmodel_intrinsics[1, 1] == dmonitoringmodel_fl


class TestTransformMatrices:
  """Test transformation matrices."""

  def test_bigmodel_frame_from_calib_shape(self):
    """Test bigmodel_frame_from_calib_frame shape."""
    assert bigmodel_frame_from_calib_frame.shape == (3, 4)

  def test_sbigmodel_frame_from_calib_shape(self):
    """Test sbigmodel_frame_from_calib_frame shape."""
    assert sbigmodel_frame_from_calib_frame.shape == (3, 4)

  def test_medmodel_frame_from_calib_shape(self):
    """Test medmodel_frame_from_calib_frame shape."""
    assert medmodel_frame_from_calib_frame.shape == (3, 4)

  def test_medmodel_frame_from_bigmodel_shape(self):
    """Test medmodel_frame_from_bigmodel_frame shape."""
    assert medmodel_frame_from_bigmodel_frame.shape == (3, 3)

  def test_calib_from_medmodel_shape(self):
    """Test calib_from_medmodel shape."""
    assert calib_from_medmodel.shape == (3, 3)

  def test_calib_from_sbigmodel_shape(self):
    """Test calib_from_sbigmodel shape."""
    assert calib_from_sbigmodel.shape == (3, 3)


class TestGetWarpMatrix:
  """Test get_warp_matrix function."""

  def test_get_warp_matrix_returns_3x3(self):
    """Test get_warp_matrix returns 3x3 matrix."""
    euler = np.array([0.0, 0.0, 0.0])
    intrinsics = medmodel_intrinsics
    result = get_warp_matrix(euler, intrinsics)

    assert result.shape == (3, 3)

  def test_get_warp_matrix_with_bigmodel_frame(self):
    """Test get_warp_matrix with bigmodel_frame=True."""
    euler = np.array([0.0, 0.0, 0.0])
    intrinsics = bigmodel_intrinsics
    result = get_warp_matrix(euler, intrinsics, bigmodel_frame=True)

    assert result.shape == (3, 3)

  def test_get_warp_matrix_zero_euler(self):
    """Test get_warp_matrix with zero euler angles."""
    euler = np.array([0.0, 0.0, 0.0])
    intrinsics = medmodel_intrinsics
    result = get_warp_matrix(euler, intrinsics)

    # Should be a valid transformation matrix (not all zeros)
    assert not np.allclose(result, 0)

  def test_get_warp_matrix_nonzero_euler(self):
    """Test get_warp_matrix with non-zero euler angles."""
    euler = np.array([0.01, 0.02, 0.03])
    intrinsics = medmodel_intrinsics
    result = get_warp_matrix(euler, intrinsics)

    assert result.shape == (3, 3)
    assert not np.allclose(result, 0)

  def test_get_warp_matrix_different_intrinsics(self):
    """Test get_warp_matrix with different intrinsic matrices."""
    euler = np.array([0.0, 0.0, 0.0])

    result_med = get_warp_matrix(euler, medmodel_intrinsics)
    result_big = get_warp_matrix(euler, bigmodel_intrinsics)

    # Different intrinsics should give different results
    assert not np.allclose(result_med, result_big)
