"""Unit tests for voxel grid occupancy tracker."""

from __future__ import annotations

import numpy as np

from openpilot.selfdrive.controls.lib.trackers.voxel_grid import VoxelGrid, VoxelGridConfig


class TestVoxelGridConfig:
  """Tests for VoxelGridConfig."""

  def test_default_config(self):
    """Test default configuration values."""
    config = VoxelGridConfig()

    assert config.x_min == -50.0
    assert config.x_max == 150.0
    assert config.y_min == -25.0
    assert config.y_max == 25.0
    assert config.z_min == -2.0
    assert config.z_max == 3.0
    assert config.resolution == 0.5

  def test_custom_config(self):
    """Test custom configuration."""
    config = VoxelGridConfig(
      x_min=0.0,
      x_max=100.0,
      resolution=1.0,
    )

    assert config.x_min == 0.0
    assert config.x_max == 100.0
    assert config.resolution == 1.0


class TestVoxelGridInit:
  """Tests for VoxelGrid initialization."""

  def test_default_init(self):
    """Test default initialization."""
    grid = VoxelGrid()

    assert grid.config is not None
    assert grid.log_odds.shape == grid.shape
    assert grid.update_count == 0

  def test_custom_config_init(self):
    """Test initialization with custom config."""
    config = VoxelGridConfig(
      x_min=0.0,
      x_max=10.0,
      y_min=-5.0,
      y_max=5.0,
      z_min=0.0,
      z_max=2.0,
      resolution=1.0,
    )
    grid = VoxelGrid(config)

    assert grid.nx == 10
    assert grid.ny == 10
    assert grid.nz == 2
    assert grid.shape == (10, 10, 2)

  def test_grid_initialized_to_zero(self):
    """Test grid starts with all zeros (unknown state)."""
    grid = VoxelGrid()
    assert np.all(grid.log_odds == 0.0)


class TestCoordinateConversion:
  """Tests for coordinate conversion methods."""

  def test_world_to_grid(self):
    """Test world to grid coordinate conversion."""
    config = VoxelGridConfig(
      x_min=0.0,
      x_max=10.0,
      y_min=0.0,
      y_max=10.0,
      z_min=0.0,
      z_max=5.0,
      resolution=1.0,
    )
    grid = VoxelGrid(config)

    ix, iy, iz = grid.world_to_grid(5.5, 3.2, 1.8)
    assert ix == 5
    assert iy == 3
    assert iz == 1

  def test_grid_to_world(self):
    """Test grid to world coordinate conversion (voxel center)."""
    config = VoxelGridConfig(
      x_min=0.0,
      x_max=10.0,
      y_min=0.0,
      y_max=10.0,
      z_min=0.0,
      z_max=5.0,
      resolution=1.0,
    )
    grid = VoxelGrid(config)

    x, y, z = grid.grid_to_world(5, 3, 1)
    assert x == 5.5  # Center of voxel
    assert y == 3.5
    assert z == 1.5

  def test_coordinate_roundtrip(self):
    """Test world->grid->world gives voxel center."""
    config = VoxelGridConfig(
      x_min=-10.0,
      x_max=10.0,
      y_min=-10.0,
      y_max=10.0,
      z_min=-2.0,
      z_max=2.0,
      resolution=0.5,
    )
    grid = VoxelGrid(config)

    # Any point in a voxel should map to that voxel's center
    wx, wy, wz = 3.3, -2.1, 0.9
    ix, iy, iz = grid.world_to_grid(wx, wy, wz)
    cx, cy, cz = grid.grid_to_world(ix, iy, iz)

    # Center should be within one voxel size of original
    assert abs(cx - wx) < config.resolution
    assert abs(cy - wy) < config.resolution
    assert abs(cz - wz) < config.resolution


class TestIndexValidation:
  """Tests for index validation."""

  def test_valid_indices(self):
    """Test valid indices return True."""
    config = VoxelGridConfig(
      x_min=0.0,
      x_max=10.0,
      y_min=0.0,
      y_max=10.0,
      z_min=0.0,
      z_max=5.0,
      resolution=1.0,
    )
    grid = VoxelGrid(config)

    assert grid.is_valid_index(0, 0, 0)
    assert grid.is_valid_index(5, 5, 2)
    assert grid.is_valid_index(9, 9, 4)

  def test_invalid_indices(self):
    """Test invalid indices return False."""
    config = VoxelGridConfig(
      x_min=0.0,
      x_max=10.0,
      y_min=0.0,
      y_max=10.0,
      z_min=0.0,
      z_max=5.0,
      resolution=1.0,
    )
    grid = VoxelGrid(config)

    assert not grid.is_valid_index(-1, 0, 0)
    assert not grid.is_valid_index(0, -1, 0)
    assert not grid.is_valid_index(0, 0, -1)
    assert not grid.is_valid_index(10, 0, 0)  # nx=10, so max valid is 9
    assert not grid.is_valid_index(0, 10, 0)
    assert not grid.is_valid_index(0, 0, 5)


class TestVoxelUpdate:
  """Tests for voxel update operations."""

  def test_update_single_voxel(self):
    """Test updating a single voxel."""
    grid = VoxelGrid()
    ix, iy, iz = 100, 50, 5

    initial = grid.log_odds[ix, iy, iz]
    grid.update_voxel(ix, iy, iz, 0.5)

    assert grid.log_odds[ix, iy, iz] == initial + 0.5

  def test_update_clamps_max(self):
    """Test log-odds clamping at maximum."""
    config = VoxelGridConfig(log_odds_max=2.0)
    grid = VoxelGrid(config)

    ix, iy, iz = 100, 50, 5
    grid.update_voxel(ix, iy, iz, 10.0)  # Large update

    assert grid.log_odds[ix, iy, iz] == config.log_odds_max

  def test_update_clamps_min(self):
    """Test log-odds clamping at minimum."""
    config = VoxelGridConfig(log_odds_min=-2.0)
    grid = VoxelGrid(config)

    ix, iy, iz = 100, 50, 5
    grid.update_voxel(ix, iy, iz, -10.0)  # Large negative update

    assert grid.log_odds[ix, iy, iz] == config.log_odds_min

  def test_update_invalid_index_noop(self):
    """Test updating invalid index does nothing."""
    grid = VoxelGrid()
    original = grid.log_odds.copy()

    grid.update_voxel(-1, 0, 0, 1.0)
    grid.update_voxel(10000, 0, 0, 1.0)

    assert np.array_equal(grid.log_odds, original)


class TestPointCloudUpdate:
  """Tests for point cloud updates."""

  def test_update_with_single_point(self):
    """Test updating with a single point."""
    config = VoxelGridConfig(
      x_min=0.0,
      x_max=100.0,
      y_min=-50.0,
      y_max=50.0,
      z_min=-2.0,
      z_max=3.0,
      resolution=1.0,
      decay_rate=0.0,  # Disable decay for test
    )
    grid = VoxelGrid(config)

    point = np.array([[50.0, 0.0, 0.0]])
    grid.update_with_points(point, ray_trace=False)

    # Check the voxel at the point is occupied
    assert grid.is_occupied(50.0, 0.0, 0.0)
    assert grid.update_count == 1

  def test_update_with_multiple_points(self):
    """Test updating with multiple points."""
    config = VoxelGridConfig(
      x_min=0.0,
      x_max=100.0,
      y_min=-50.0,
      y_max=50.0,
      z_min=-2.0,
      z_max=3.0,
      resolution=1.0,
      decay_rate=0.0,
    )
    grid = VoxelGrid(config)

    points = np.array(
      [
        [10.0, 0.0, 0.0],
        [20.0, 5.0, 0.0],
        [30.0, -5.0, 1.0],
      ]
    )
    grid.update_with_points(points, ray_trace=False)

    assert grid.is_occupied(10.0, 0.0, 0.0)
    assert grid.is_occupied(20.0, 5.0, 0.0)
    assert grid.is_occupied(30.0, -5.0, 1.0)

  def test_ray_tracing_marks_free_space(self):
    """Test that ray tracing marks intermediate voxels as free."""
    config = VoxelGridConfig(
      x_min=0.0,
      x_max=100.0,
      y_min=-50.0,
      y_max=50.0,
      z_min=-2.0,
      z_max=3.0,
      resolution=1.0,
      decay_rate=0.0,
      log_odds_occupied=2.0,
      log_odds_free=-0.5,
    )
    grid = VoxelGrid(config)

    # Point at x=50, origin at x=0
    origin = np.array([0.0, 0.0, 0.0])
    point = np.array([[50.0, 0.0, 0.0]])
    grid.update_with_points(point, origin=origin, ray_trace=True)

    # Endpoint should be occupied
    assert grid.is_occupied(50.0, 0.0, 0.0)

    # Intermediate points should have been marked free
    # After one update with log_odds_free=-0.5, they'll be below threshold
    prob_25 = grid.get_probability(25.0, 0.0, 0.0)
    assert prob_25 < 0.5  # Free space has probability < 0.5

  def test_temporal_decay(self):
    """Test temporal decay reduces log-odds over time."""
    config = VoxelGridConfig(
      x_min=0.0,
      x_max=100.0,
      y_min=-50.0,
      y_max=50.0,
      z_min=-2.0,
      z_max=3.0,
      resolution=1.0,
      decay_rate=0.1,  # 10% decay per update
    )
    grid = VoxelGrid(config)

    # Set a voxel to high occupancy
    grid.log_odds[50, 50, 2] = 2.0

    # Update with empty point cloud triggers decay
    grid.update_with_points(np.empty((0, 3)), ray_trace=False)

    # Should be decayed by 10%
    assert abs(grid.log_odds[50, 50, 2] - 1.8) < 0.01


class TestOccupancyQuery:
  """Tests for occupancy query methods."""

  def test_is_occupied_threshold(self):
    """Test occupancy threshold check."""
    config = VoxelGridConfig(log_odds_threshold=0.5)
    grid = VoxelGrid(config)

    ix, iy, iz = 100, 50, 5
    x, y, z = grid.grid_to_world(ix, iy, iz)

    grid.log_odds[ix, iy, iz] = 0.4
    assert not grid.is_occupied(x, y, z)

    grid.log_odds[ix, iy, iz] = 0.6
    assert grid.is_occupied(x, y, z)

  def test_is_occupied_out_of_bounds(self):
    """Test occupancy query for out-of-bounds returns False."""
    grid = VoxelGrid()

    # Way outside grid bounds
    assert not grid.is_occupied(1000.0, 1000.0, 1000.0)
    assert not grid.is_occupied(-1000.0, 0.0, 0.0)

  def test_get_probability(self):
    """Test probability calculation from log-odds."""
    grid = VoxelGrid()
    ix, iy, iz = 100, 50, 5
    x, y, z = grid.grid_to_world(ix, iy, iz)

    # log_odds = 0 -> probability = 0.5
    grid.log_odds[ix, iy, iz] = 0.0
    assert abs(grid.get_probability(x, y, z) - 0.5) < 0.001

    # Positive log-odds -> probability > 0.5
    grid.log_odds[ix, iy, iz] = 2.0
    prob = grid.get_probability(x, y, z)
    assert prob > 0.5
    expected = 1.0 / (1.0 + np.exp(-2.0))
    assert abs(prob - expected) < 0.001

    # Negative log-odds -> probability < 0.5
    grid.log_odds[ix, iy, iz] = -2.0
    prob = grid.get_probability(x, y, z)
    assert prob < 0.5

  def test_get_probability_out_of_bounds(self):
    """Test probability query for out-of-bounds returns 0.5 (unknown)."""
    grid = VoxelGrid()
    assert grid.get_probability(1000.0, 1000.0, 1000.0) == 0.5


class TestOccupiedVoxelRetrieval:
  """Tests for retrieving occupied voxels."""

  def test_get_occupied_voxels_empty(self):
    """Test getting occupied voxels when none are occupied."""
    grid = VoxelGrid()
    occupied = grid.get_occupied_voxels()

    assert occupied.shape == (0, 3)

  def test_get_occupied_voxels(self):
    """Test getting occupied voxels."""
    config = VoxelGridConfig(
      x_min=0.0,
      x_max=10.0,
      y_min=0.0,
      y_max=10.0,
      z_min=0.0,
      z_max=5.0,
      resolution=1.0,
      log_odds_threshold=0.0,
    )
    grid = VoxelGrid(config)

    # Mark some voxels as occupied
    grid.log_odds[5, 5, 2] = 1.0
    grid.log_odds[3, 7, 1] = 0.5

    occupied = grid.get_occupied_voxels()
    assert occupied.shape[0] == 2
    assert occupied.shape[1] == 3

  def test_get_occupied_in_region(self):
    """Test getting occupied voxels in a specific region."""
    config = VoxelGridConfig(
      x_min=0.0,
      x_max=100.0,
      y_min=-50.0,
      y_max=50.0,
      z_min=-2.0,
      z_max=3.0,
      resolution=1.0,
      log_odds_threshold=0.0,
    )
    grid = VoxelGrid(config)

    # Mark voxels in different regions
    grid.log_odds[10, 50, 2] = 1.0  # x=10
    grid.log_odds[50, 50, 2] = 1.0  # x=50
    grid.log_odds[90, 50, 2] = 1.0  # x=90

    # Query region x=[40, 60]
    occupied = grid.get_occupied_in_region(40.0, 60.0, -50.0, 50.0)

    # Should only find the one at x=50
    assert occupied.shape[0] == 1


class TestGridManagement:
  """Tests for grid management operations."""

  def test_clear(self):
    """Test clearing the grid."""
    grid = VoxelGrid()

    # Set some values
    grid.log_odds[100, 50, 5] = 2.0
    grid._update_count = 10

    grid.clear()

    assert np.all(grid.log_odds == 0.0)
    assert grid.update_count == 0

  def test_shape_property(self):
    """Test shape property."""
    config = VoxelGridConfig(
      x_min=0.0,
      x_max=20.0,
      y_min=-10.0,
      y_max=10.0,
      z_min=0.0,
      z_max=5.0,
      resolution=1.0,
    )
    grid = VoxelGrid(config)

    assert grid.shape == (20, 20, 5)

  def test_get_statistics(self):
    """Test grid statistics."""
    config = VoxelGridConfig(
      x_min=0.0,
      x_max=10.0,
      y_min=0.0,
      y_max=10.0,
      z_min=0.0,
      z_max=2.0,
      resolution=1.0,
      log_odds_threshold=0.5,
    )
    grid = VoxelGrid(config)

    # Set some voxels
    grid.log_odds[0, 0, 0] = 1.0  # Occupied
    grid.log_odds[1, 1, 1] = -1.0  # Free
    grid._update_count = 5

    stats = grid.get_statistics()

    assert stats["shape"] == (10, 10, 2)
    assert stats["resolution"] == 1.0
    assert stats["total_voxels"] == 200
    assert stats["occupied_voxels"] == 1
    assert stats["free_voxels"] == 1
    assert stats["unknown_voxels"] == 198
    assert stats["update_count"] == 5
    assert "memory_mb" in stats


class TestEdgeCases:
  """Tests for edge cases and boundary conditions."""

  def test_point_at_grid_boundary(self):
    """Test point exactly at grid boundary."""
    config = VoxelGridConfig(
      x_min=0.0,
      x_max=10.0,
      y_min=0.0,
      y_max=10.0,
      z_min=0.0,
      z_max=5.0,
      resolution=1.0,
      decay_rate=0.0,
    )
    grid = VoxelGrid(config)

    # Point at x_max boundary (should be outside grid)
    point = np.array([[10.0, 5.0, 2.0]])
    grid.update_with_points(point, ray_trace=False)

    # Should not crash, point is out of bounds
    assert grid.update_count == 1

  def test_empty_point_cloud(self):
    """Test update with empty point cloud."""
    grid = VoxelGrid()

    # Should not crash
    grid.update_with_points(np.empty((0, 3)), ray_trace=True)
    assert grid.update_count == 1

  def test_very_small_resolution(self):
    """Test grid with very small resolution (high detail)."""
    config = VoxelGridConfig(
      x_min=0.0,
      x_max=5.0,
      y_min=0.0,
      y_max=5.0,
      z_min=0.0,
      z_max=2.0,
      resolution=0.1,
    )
    grid = VoxelGrid(config)

    assert grid.nx == 50
    assert grid.ny == 50
    assert grid.nz == 20

  def test_point_exactly_at_voxel_center(self):
    """Test point placed exactly at voxel center."""
    config = VoxelGridConfig(
      x_min=0.0,
      x_max=10.0,
      y_min=0.0,
      y_max=10.0,
      z_min=0.0,
      z_max=5.0,
      resolution=1.0,
      decay_rate=0.0,
    )
    grid = VoxelGrid(config)

    # Voxel center is at 0.5, 0.5, 0.5
    point = np.array([[0.5, 0.5, 0.5]])
    grid.update_with_points(point, ray_trace=False)

    assert grid.is_occupied(0.5, 0.5, 0.5)
