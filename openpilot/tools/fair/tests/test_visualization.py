"""Tests for FAIR visualization utilities."""

import numpy as np
import pytest

from openpilot.tools.fair.visualization.features import (
  visualize_features,
  visualize_pca_features,
  visualize_attention,
  create_feature_grid,
  overlay_features_on_image,
  compute_feature_similarity,
  CV2_AVAILABLE,
)
from openpilot.tools.fair.visualization.depth import (
  depth_to_colormap,
  visualize_depth,
  create_depth_overlay,
  create_depth_comparison,
  compute_depth_error_map,
  visualize_depth_error,
)


class TestVisualizeFeatures:
  """Tests for visualize_features."""

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_2d_features(self):
    """Test with 2D feature map."""
    features = np.random.rand(32, 32)
    result = visualize_features(features)

    assert result.shape == (32, 32, 3)
    assert result.dtype == np.uint8

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_3d_features(self):
    """Test with multi-channel features."""
    features = np.random.rand(16, 16, 64)
    result = visualize_features(features)

    assert result.shape == (16, 16, 3)

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_with_resize(self):
    """Test with target size."""
    features = np.random.rand(16, 16)
    result = visualize_features(features, image_size=(64, 128))

    assert result.shape == (64, 128, 3)


class TestVisualizePCAFeatures:
  """Tests for visualize_pca_features."""

  def test_basic_pca(self):
    """Test basic PCA visualization."""
    # 16x16 = 256 patches with 64 dimensions
    features = np.random.rand(256, 64)
    result = visualize_pca_features(features)

    assert result.shape == (16, 16, 3)
    assert result.dtype == np.uint8

  def test_with_batch_dim(self):
    """Test with batch dimension."""
    features = np.random.rand(2, 64, 32)  # 8x8 patches
    result = visualize_pca_features(features)

    assert result.shape == (8, 8, 3)

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_with_resize(self):
    """Test with target size."""
    features = np.random.rand(64, 128)  # 8x8 patches
    result = visualize_pca_features(features, image_size=(128, 256))

    assert result.shape == (128, 256, 3)

  def test_invalid_patch_count(self):
    """Test error with non-square patch count."""
    features = np.random.rand(99, 32)  # Not a perfect square (99 != n*n)
    with pytest.raises(ValueError, match="Cannot reshape"):
      visualize_pca_features(features)


class TestVisualizeAttention:
  """Tests for visualize_attention."""

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_single_head(self):
    """Test with single attention head."""
    # 65 tokens (CLS + 64 patches for 8x8)
    attention = np.random.rand(65, 65)
    result = visualize_attention(attention)

    assert result.shape == (8, 8, 3)

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_multi_head(self):
    """Test with multi-head attention."""
    attention = np.random.rand(12, 65, 65)  # 12 heads
    result = visualize_attention(attention)

    assert result.shape == (8, 8, 3)

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_specific_head(self):
    """Test selecting specific head."""
    attention = np.random.rand(8, 65, 65)
    result = visualize_attention(attention, head_idx=3)

    assert result.shape == (8, 8, 3)

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_with_resize(self):
    """Test with target size."""
    attention = np.random.rand(17, 17)  # CLS + 16 patches (4x4)
    result = visualize_attention(attention, image_size=(128, 128))

    assert result.shape == (128, 128, 3)


class TestCreateFeatureGrid:
  """Tests for create_feature_grid."""

  def test_from_flat_features(self):
    """Test grid from flat feature array."""
    features = np.random.rand(64, 128)  # 8x8 patches, 128 dims
    grid = create_feature_grid(features, grid_size=4)

    # Should be 4x4 grid of 8x8 feature maps
    assert grid.shape == (32, 32)

  def test_from_spatial_features(self):
    """Test grid from spatial features."""
    features = np.random.rand(8, 8, 64)
    grid = create_feature_grid(features, grid_size=8)

    assert grid.shape == (64, 64)

  def test_normalize(self):
    """Test normalization."""
    features = np.random.rand(16, 32) * 100 - 50
    grid = create_feature_grid(features, normalize=True)

    assert grid.min() >= 0
    assert grid.max() <= 1


class TestOverlayFeaturesOnImage:
  """Tests for overlay_features_on_image."""

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_basic_overlay(self):
    """Test basic overlay."""
    image = np.random.randint(0, 255, (128, 256, 3), dtype=np.uint8)
    features = np.random.rand(32, 64)

    result = overlay_features_on_image(image, features, alpha=0.5)

    assert result.shape == image.shape

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_alpha_values(self):
    """Test different alpha values."""
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    features = np.random.rand(16, 16)

    # Alpha = 0 should be mostly image
    result_image = overlay_features_on_image(image, features, alpha=0.0)

    # Alpha = 1 should be mostly features
    result_features = overlay_features_on_image(image, features, alpha=1.0)

    # Results should be different
    assert not np.allclose(result_image, result_features)


class TestComputeFeatureSimilarity:
  """Tests for compute_feature_similarity."""

  def test_self_similarity(self):
    """Test similarity to self is 1."""
    features = np.random.rand(64, 32)  # 8x8 patches
    similarity = compute_feature_similarity(features, query_point=(4, 4))

    # Query point should have high self-similarity
    assert similarity[4, 4] > 0.99

  def test_output_shape(self):
    """Test output shape matches patch grid."""
    features = np.random.rand(256, 64)  # 16x16 patches
    similarity = compute_feature_similarity(features, query_point=(8, 8))

    assert similarity.shape == (16, 16)


class TestDepthToColormap:
  """Tests for depth_to_colormap."""

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_basic_colormap(self):
    """Test basic depth colormap."""
    depth = np.random.rand(64, 128) * 100
    result = depth_to_colormap(depth)

    assert result.shape == (64, 128, 3)

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_custom_range(self):
    """Test custom depth range."""
    depth = np.random.rand(32, 32) * 50
    result = depth_to_colormap(depth, min_depth=0, max_depth=100)

    assert result.shape == (32, 32, 3)


class TestVisualizeDepth:
  """Tests for visualize_depth."""

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_basic(self):
    """Test basic depth visualization."""
    depth = np.random.rand(128, 256) * 80 + 1
    result = visualize_depth(depth)

    assert result.shape == (128, 256, 3)

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_different_colormaps(self):
    """Test different colormap names."""
    depth = np.random.rand(64, 64) * 50

    for cmap in ["magma", "viridis", "turbo", "jet"]:
      result = visualize_depth(depth, colormap=cmap)
      assert result.shape == (64, 64, 3)


class TestCreateDepthOverlay:
  """Tests for create_depth_overlay."""

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_basic_overlay(self):
    """Test basic depth overlay."""
    image = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)
    depth = np.random.rand(256, 512) * 100

    result = create_depth_overlay(image, depth)

    assert result.shape == image.shape

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_depth_resize(self):
    """Test depth resizing to image size."""
    image = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)
    depth = np.random.rand(64, 128) * 100

    result = create_depth_overlay(image, depth)

    assert result.shape == image.shape


class TestCreateDepthComparison:
  """Tests for create_depth_comparison."""

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_without_gt(self):
    """Test comparison without ground truth."""
    image = np.random.randint(0, 255, (128, 256, 3), dtype=np.uint8)
    depth_pred = np.random.rand(128, 256) * 50

    result = create_depth_comparison(image, depth_pred)

    # Should be image | pred (2x width)
    assert result.shape == (128, 512, 3)

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_with_gt(self):
    """Test comparison with ground truth."""
    image = np.random.randint(0, 255, (128, 256, 3), dtype=np.uint8)
    depth_pred = np.random.rand(128, 256) * 50
    depth_gt = np.random.rand(128, 256) * 50

    result = create_depth_comparison(image, depth_pred, depth_gt)

    # Should be image | pred | gt (3x width)
    assert result.shape == (128, 768, 3)


class TestComputeDepthErrorMap:
  """Tests for compute_depth_error_map."""

  def test_abs_rel(self):
    """Test absolute relative error."""
    pred = np.array([[10.0, 20.0], [30.0, 40.0]])
    gt = np.array([[12.0, 18.0], [30.0, 50.0]])

    error = compute_depth_error_map(pred, gt, metric="abs_rel")

    assert error.shape == (2, 2)
    assert error[1, 0] < 0.01  # Should be close to 0 where pred == gt

  def test_rmse(self):
    """Test RMSE error."""
    pred = np.ones((8, 8)) * 10
    gt = np.ones((8, 8)) * 10

    error = compute_depth_error_map(pred, gt, metric="rmse")

    np.testing.assert_array_almost_equal(error, 0)

  def test_invalid_metric(self):
    """Test error for invalid metric."""
    pred = np.random.rand(8, 8)
    gt = np.random.rand(8, 8)

    with pytest.raises(ValueError, match="Unknown metric"):
      compute_depth_error_map(pred, gt, metric="invalid")


class TestVisualizeDepthError:
  """Tests for visualize_depth_error."""

  @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
  def test_basic(self):
    """Test basic error visualization."""
    pred = np.random.rand(64, 64) * 50 + 1
    gt = np.random.rand(64, 64) * 50 + 1

    result = visualize_depth_error(pred, gt)

    assert result.shape == (64, 64, 3)
