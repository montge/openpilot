"""Tests for octree spatial index."""
import numpy as np

from openpilot.tools.stonesoup.octree import (
  BenchmarkResult,
  BoundingBox,
  Octree,
  OctreeConfig,
  benchmark_octree,
  brute_force_knn,
  brute_force_radius_query,
  format_benchmark_report,
)


class TestBoundingBox:
  def test_contains_point(self):
    box = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([10.0, 10.0, 10.0])
    )

    assert box.contains_point(np.array([5.0, 5.0, 5.0]))
    assert box.contains_point(np.array([0.0, 0.0, 0.0]))  # On boundary
    assert not box.contains_point(np.array([-1.0, 5.0, 5.0]))
    assert not box.contains_point(np.array([15.0, 5.0, 5.0]))

  def test_intersects(self):
    box1 = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([10.0, 10.0, 10.0])
    )
    box2 = BoundingBox(
      min_corner=np.array([5.0, 5.0, 5.0]),
      max_corner=np.array([15.0, 15.0, 15.0])
    )
    box3 = BoundingBox(
      min_corner=np.array([20.0, 20.0, 20.0]),
      max_corner=np.array([30.0, 30.0, 30.0])
    )

    assert box1.intersects(box2)
    assert box2.intersects(box1)
    assert not box1.intersects(box3)

  def test_intersects_sphere(self):
    box = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([10.0, 10.0, 10.0])
    )

    # Sphere inside box
    assert box.intersects_sphere(np.array([5.0, 5.0, 5.0]), 1.0)

    # Sphere touching box
    assert box.intersects_sphere(np.array([11.0, 5.0, 5.0]), 1.0)

    # Sphere outside box
    assert not box.intersects_sphere(np.array([20.0, 5.0, 5.0]), 1.0)

  def test_split(self):
    box = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([10.0, 10.0, 10.0])
    )

    octants = box.split()
    assert len(octants) == 8

    # Each octant should be 5x5x5
    for octant in octants:
      np.testing.assert_array_almost_equal(octant.size, np.array([5.0, 5.0, 5.0]))

  def test_center(self):
    box = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([10.0, 10.0, 10.0])
    )

    np.testing.assert_array_equal(box.center, np.array([5.0, 5.0, 5.0]))


class TestOctree:
  def test_initialization(self):
    tree = Octree()
    assert tree.n_points == 0

  def test_insert_single_point(self):
    tree = Octree()
    point_id = tree.insert(np.array([50.0, 0.0, 1.0]))

    assert tree.n_points == 1
    assert point_id == 0

  def test_insert_multiple_points(self):
    tree = Octree()

    points = np.array([
      [10.0, 0.0, 0.0],
      [20.0, 0.0, 0.0],
      [30.0, 0.0, 0.0],
    ])

    ids = tree.insert_points(points)

    assert tree.n_points == 3
    assert len(ids) == 3

  def test_range_query(self):
    tree = Octree()

    # Insert points in a line
    for i in range(10):
      tree.insert(np.array([i * 10.0, 0.0, 0.0]))

    # Query for middle points
    query_box = BoundingBox(
      min_corner=np.array([25.0, -5.0, -5.0]),
      max_corner=np.array([55.0, 5.0, 5.0])
    )

    results = tree.range_query(query_box)

    # Should find points at 30, 40, 50
    assert len(results) == 3

  def test_radius_query(self):
    tree = Octree()

    # Insert a cluster of points
    np.random.seed(42)
    center = np.array([50.0, 0.0, 1.0])
    for _ in range(100):
      point = center + np.random.randn(3) * 2
      tree.insert(point)

    # Query around center
    results = tree.radius_query(center, 3.0)

    # Should find most points (within 1.5 std devs)
    assert len(results) > 50

    # Results should be sorted by distance
    distances = [r[2] for r in results]
    assert distances == sorted(distances)

  def test_k_nearest(self):
    tree = Octree()

    # Insert points in a line
    for i in range(20):
      tree.insert(np.array([i * 5.0, 0.0, 0.0]))

    # Query near one end
    query = np.array([0.0, 0.0, 0.0])
    neighbors = tree.k_nearest(query, 5)

    assert len(neighbors) == 5

    # Closest should be at origin
    np.testing.assert_array_almost_equal(neighbors[0][0], np.array([0.0, 0.0, 0.0]))

    # Distances should be increasing
    distances = [n[2] for n in neighbors]
    assert distances == sorted(distances)

  def test_clear(self):
    tree = Octree()
    tree.insert(np.array([50.0, 0.0, 1.0]))
    tree.clear()

    assert tree.n_points == 0

  def test_node_splitting(self):
    config = OctreeConfig(max_points_per_node=4)
    tree = Octree(config=config)

    # Insert more points than max per node
    for i in range(20):
      tree.insert(np.array([i * 5.0, 0.0, 0.0]))

    # Should have split nodes
    assert not tree.root.is_leaf


class TestBruteForce:
  def test_brute_force_radius(self):
    np.random.seed(42)
    points = np.random.randn(100, 3) * 5 + np.array([50, 0, 0])
    center = np.array([50.0, 0.0, 0.0])

    results = brute_force_radius_query(points, center, 3.0)

    # Results should be sorted by distance
    if len(results) > 1:
      distances = [r[1] for r in results]
      assert distances == sorted(distances)

  def test_brute_force_knn(self):
    np.random.seed(42)
    points = np.random.randn(100, 3) * 5
    center = np.array([0.0, 0.0, 0.0])

    results = brute_force_knn(points, center, 5)

    assert len(results) == 5

    # Distances should be sorted
    distances = [r[1] for r in results]
    assert distances == sorted(distances)


class TestOctreeVsBruteForce:
  def test_radius_query_matches(self):
    """Octree should return same results as brute force."""
    np.random.seed(42)
    points = np.random.randn(200, 3) * 10 + np.array([50, 0, 0])
    center = np.array([50.0, 0.0, 0.0])
    radius = 5.0

    # Octree
    bounds = BoundingBox(
      min_corner=np.array([0.0, -50.0, -50.0]),
      max_corner=np.array([100.0, 50.0, 50.0])
    )
    tree = Octree(bounds)
    tree.insert_points(points)
    octree_results = tree.radius_query(center, radius)

    # Brute force
    brute_results = brute_force_radius_query(points, center, radius)

    # Should find same number of points
    assert len(octree_results) == len(brute_results)

  def test_knn_matches(self):
    """Octree KNN should match brute force."""
    np.random.seed(42)
    points = np.random.randn(200, 3) * 10 + np.array([50, 0, 0])
    center = np.array([55.0, 2.0, 1.0])
    k = 10

    # Octree
    bounds = BoundingBox(
      min_corner=np.array([0.0, -50.0, -50.0]),
      max_corner=np.array([100.0, 50.0, 50.0])
    )
    tree = Octree(bounds)
    tree.insert_points(points)
    octree_results = tree.k_nearest(center, k)

    # Brute force
    brute_results = brute_force_knn(points, center, k)

    # Distances should match
    octree_dists = [r[2] for r in octree_results]
    brute_dists = [r[1] for r in brute_results]

    np.testing.assert_array_almost_equal(octree_dists, brute_dists, decimal=5)


class TestBenchmark:
  def test_benchmark_runs(self):
    results = benchmark_octree(n_points=500, n_queries=10, k=5)

    assert len(results) >= 5  # Build + 2 ops * 2 methods

    for r in results:
      assert isinstance(r, BenchmarkResult)
      assert r.time_ms > 0

  def test_format_report(self):
    results = benchmark_octree(n_points=500, n_queries=10)
    report = format_benchmark_report(results)

    assert "# Octree Benchmark Results" in report
    assert "Octree" in report
    assert "Brute Force" in report
    assert "Speedup" in report
