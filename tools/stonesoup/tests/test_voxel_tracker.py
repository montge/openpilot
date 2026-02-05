"""Tests for voxel grid tracker."""
import numpy as np

from openpilot.tools.stonesoup.voxel_tracker import (
  GPUVoxelGrid,
  SparseVoxelGrid,
  VoxelGrid,
  VoxelGridConfig,
  VoxelTracker,
  benchmark_voxel_grids,
  format_benchmark_report,
)


class TestVoxelGridConfig:
  def test_default_config(self):
    config = VoxelGridConfig()
    assert config.resolution == 0.5
    assert config.x_range == (0.0, 100.0)
    assert config.l_occ > 0
    assert config.l_free < 0


class TestVoxelGrid:
  def test_initialization(self):
    config = VoxelGridConfig(
      x_range=(0.0, 10.0),
      y_range=(-5.0, 5.0),
      z_range=(0.0, 2.0),
      resolution=0.5
    )
    grid = VoxelGrid(config)

    assert grid.nx == 20
    assert grid.ny == 20
    assert grid.nz == 4

  def test_world_to_voxel(self):
    config = VoxelGridConfig(
      x_range=(0.0, 10.0),
      y_range=(-5.0, 5.0),
      z_range=(0.0, 2.0),
      resolution=1.0
    )
    grid = VoxelGrid(config)

    voxel = grid.world_to_voxel(np.array([5.0, 0.0, 1.0]))
    assert voxel[0] == 5
    assert voxel[1] == 5
    assert voxel[2] == 1

  def test_voxel_to_world(self):
    config = VoxelGridConfig(resolution=1.0)
    grid = VoxelGrid(config)

    world = grid.voxel_to_world(np.array([10, 20, 2]))
    # Should be at center of voxel
    assert world[0] == 10.5  # 10 + 0.5
    assert world[1] == 0.5   # -20 + 20 + 0.5
    assert world[2] == 0.5   # -2 + 2 + 0.5

  def test_update_occupied(self):
    config = VoxelGridConfig(resolution=1.0)
    grid = VoxelGrid(config)

    point = np.array([10.0, 0.0, 0.5])
    grid.update_occupied(point)

    voxel = grid.world_to_voxel(point)
    prob = grid.get_probability(voxel)

    # Should be > 0.5 after one occupied update
    assert prob > 0.5

  def test_update_free(self):
    config = VoxelGridConfig(resolution=1.0)
    grid = VoxelGrid(config)

    point = np.array([10.0, 0.0, 0.5])

    # First mark as occupied
    for _ in range(3):
      grid.update_occupied(point)

    # Then mark as free
    for _ in range(5):
      grid.update_free(point)

    voxel = grid.world_to_voxel(point)
    prob = grid.get_probability(voxel)

    # Should be < 0.5 after more free updates
    assert prob < 0.5

  def test_update_ray(self):
    config = VoxelGridConfig(
      x_range=(0.0, 20.0),
      y_range=(-5.0, 5.0),
      z_range=(-1.0, 2.0),
      resolution=1.0
    )
    grid = VoxelGrid(config)

    origin = np.array([0.5, 0.0, 0.5])
    endpoint = np.array([10.0, 0.0, 0.5])

    grid.update_ray(origin, endpoint)

    # Endpoint should be occupied
    end_voxel = grid.world_to_voxel(endpoint)
    assert grid.get_probability(end_voxel) > 0.5

    # Point in between should be free
    mid_point = np.array([5.0, 0.0, 0.5])
    mid_voxel = grid.world_to_voxel(mid_point)
    assert grid.get_probability(mid_voxel) < 0.5

  def test_get_occupied_voxels(self):
    config = VoxelGridConfig(resolution=1.0)
    grid = VoxelGrid(config)

    # Mark several points as occupied
    points = [
      np.array([10.0, 0.0, 0.5]),
      np.array([20.0, 1.0, 0.5]),
      np.array([30.0, -1.0, 0.5]),
    ]

    for point in points:
      for _ in range(3):  # Multiple updates to exceed threshold
        grid.update_occupied(point)

    occupied = grid.get_occupied_voxels()
    assert len(occupied) == 3

  def test_decay(self):
    config = VoxelGridConfig(resolution=1.0)
    grid = VoxelGrid(config)

    point = np.array([10.0, 0.0, 0.5])
    for _ in range(5):
      grid.update_occupied(point)

    voxel = grid.world_to_voxel(point)
    prob_before = grid.get_probability(voxel)

    grid.decay(0.5)
    prob_after = grid.get_probability(voxel)

    # Probability should decrease toward 0.5
    assert abs(prob_after - 0.5) < abs(prob_before - 0.5)

  def test_clear(self):
    config = VoxelGridConfig(resolution=1.0)
    grid = VoxelGrid(config)

    point = np.array([10.0, 0.0, 0.5])
    grid.update_occupied(point)
    grid.clear()

    voxel = grid.world_to_voxel(point)
    assert grid.get_log_odds(voxel) == 0.0


class TestSparseVoxelGrid:
  def test_initialization(self):
    grid = SparseVoxelGrid()
    assert grid.n_allocated == 0
    assert grid.sparsity == 1.0

  def test_update_occupied(self):
    grid = SparseVoxelGrid()

    point = np.array([10.0, 0.0, 0.5])
    grid.update_occupied(point)

    assert grid.n_allocated == 1

    key = grid.world_to_voxel(point)
    prob = grid.get_probability(key)
    assert prob > 0.5

  def test_sparse_memory(self):
    """Sparse grid should use less memory than dense for sparse data."""
    config = VoxelGridConfig(
      x_range=(0.0, 100.0),
      y_range=(-20.0, 20.0),
      z_range=(-2.0, 3.0),
      resolution=0.25
    )

    dense = VoxelGrid(config)
    sparse = SparseVoxelGrid(config)

    # Add a few points
    np.random.seed(42)
    for _ in range(100):
      point = np.array([
        np.random.uniform(0, 100),
        np.random.uniform(-20, 20),
        np.random.uniform(-2, 3)
      ])
      dense.update_occupied(point)
      sparse.update_occupied(point)

    # Sparse should use much less memory
    dense_size = dense._grid.nbytes
    sparse_size = len(sparse._voxels) * 64  # Approximate dict overhead

    assert sparse_size < dense_size / 10

  def test_get_occupied_points(self):
    grid = SparseVoxelGrid()

    point = np.array([10.0, 0.0, 0.5])
    for _ in range(3):
      grid.update_occupied(point)

    occupied = grid.get_occupied_points()
    assert len(occupied) == 1

    # Check world coordinate is close to input
    np.testing.assert_array_almost_equal(
      occupied[0],
      grid.voxel_to_world(grid.world_to_voxel(point)),
      decimal=1
    )


class TestGPUVoxelGrid:
  def test_initialization(self):
    grid = GPUVoxelGrid()
    # Should work even without GPU (falls back to CPU)
    assert grid.device in ("GPU", "CPU")

  def test_batch_update(self):
    grid = GPUVoxelGrid()

    points = np.array([
      [10.0, 0.0, 0.5],
      [20.0, 1.0, 0.5],
      [30.0, -1.0, 0.5],
    ])

    for _ in range(3):
      grid.update_points_occupied(points)

    occupied = grid.get_occupied_points()
    assert len(occupied) == 3


class TestVoxelTracker:
  def test_initialization(self):
    tracker = VoxelTracker()
    assert len(tracker.results) == 0

  def test_process_detections(self):
    tracker = VoxelTracker(use_sparse=True)

    points = np.random.randn(50, 3) * 2 + np.array([30.0, 0.0, 0.5])
    result = tracker.process_detections(0.0, points)

    assert result.timestamp == 0.0
    assert result.n_occupied > 0

  def test_multiple_timesteps(self):
    tracker = VoxelTracker()

    for t in range(5):
      points = np.random.randn(20, 3) + np.array([30.0, 0.0, 0.5])
      tracker.process_detections(t * 0.1, points)

    assert len(tracker.results) == 5


class TestBenchmark:
  def test_benchmark_runs(self):
    results = benchmark_voxel_grids(n_points=50, n_iterations=2)

    assert len(results) >= 2  # At least dense and sparse
    for r in results:
      assert r.update_time_ms > 0
      assert r.query_time_ms > 0

  def test_format_report(self):
    results = benchmark_voxel_grids(n_points=50, n_iterations=2)
    report = format_benchmark_report(results)

    assert "# Voxel Grid Benchmark Results" in report
    assert "Dense" in report
    assert "Sparse" in report
