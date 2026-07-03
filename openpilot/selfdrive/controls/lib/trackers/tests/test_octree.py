"""Unit tests for octree spatial index."""

from __future__ import annotations

import numpy as np

from openpilot.selfdrive.controls.lib.trackers.octree import (
  BoundingBox,
  Octree,
  OctreeConfig,
  create_octree_from_bounds,
)


class TestBoundingBox:
  """Tests for BoundingBox."""

  def test_creation(self):
    """Test BoundingBox creation."""
    box = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([10.0, 10.0, 10.0]),
    )

    assert np.array_equal(box.min_corner, [0, 0, 0])
    assert np.array_equal(box.max_corner, [10, 10, 10])

  def test_center(self):
    """Test center calculation."""
    box = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([10.0, 20.0, 30.0]),
    )

    np.testing.assert_array_equal(box.center, [5.0, 10.0, 15.0])

  def test_size(self):
    """Test size calculation."""
    box = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([10.0, 20.0, 30.0]),
    )

    np.testing.assert_array_equal(box.size, [10.0, 20.0, 30.0])

  def test_contains_inside(self):
    """Test contains for point inside box."""
    box = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([10.0, 10.0, 10.0]),
    )

    assert box.contains(np.array([5.0, 5.0, 5.0]))
    assert box.contains(np.array([0.0, 0.0, 0.0]))  # On edge
    assert box.contains(np.array([10.0, 10.0, 10.0]))  # On edge

  def test_contains_outside(self):
    """Test contains for point outside box."""
    box = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([10.0, 10.0, 10.0]),
    )

    assert not box.contains(np.array([-1.0, 5.0, 5.0]))
    assert not box.contains(np.array([11.0, 5.0, 5.0]))
    assert not box.contains(np.array([5.0, -1.0, 5.0]))

  def test_intersects_overlapping(self):
    """Test intersects for overlapping boxes."""
    box1 = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([10.0, 10.0, 10.0]),
    )
    box2 = BoundingBox(
      min_corner=np.array([5.0, 5.0, 5.0]),
      max_corner=np.array([15.0, 15.0, 15.0]),
    )

    assert box1.intersects(box2)
    assert box2.intersects(box1)

  def test_intersects_separate(self):
    """Test intersects for non-overlapping boxes."""
    box1 = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([10.0, 10.0, 10.0]),
    )
    box2 = BoundingBox(
      min_corner=np.array([20.0, 20.0, 20.0]),
      max_corner=np.array([30.0, 30.0, 30.0]),
    )

    assert not box1.intersects(box2)
    assert not box2.intersects(box1)

  def test_get_octant(self):
    """Test octant calculation."""
    box = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([10.0, 10.0, 10.0]),
    )

    # Point in each octant
    assert box.get_octant(np.array([2.0, 2.0, 2.0])) == 0  # ---
    assert box.get_octant(np.array([8.0, 2.0, 2.0])) == 1  # +--
    assert box.get_octant(np.array([2.0, 8.0, 2.0])) == 2  # -+-
    assert box.get_octant(np.array([8.0, 8.0, 2.0])) == 3  # ++-
    assert box.get_octant(np.array([2.0, 2.0, 8.0])) == 4  # --+
    assert box.get_octant(np.array([8.0, 2.0, 8.0])) == 5  # +-+
    assert box.get_octant(np.array([2.0, 8.0, 8.0])) == 6  # -++
    assert box.get_octant(np.array([8.0, 8.0, 8.0])) == 7  # +++

  def test_get_octant_bounds(self):
    """Test octant bounds calculation."""
    box = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([10.0, 10.0, 10.0]),
    )

    # Octant 0 (---)
    oct0 = box.get_octant_bounds(0)
    np.testing.assert_array_equal(oct0.min_corner, [0, 0, 0])
    np.testing.assert_array_equal(oct0.max_corner, [5, 5, 5])

    # Octant 7 (+++)
    oct7 = box.get_octant_bounds(7)
    np.testing.assert_array_equal(oct7.min_corner, [5, 5, 5])
    np.testing.assert_array_equal(oct7.max_corner, [10, 10, 10])


class TestOctreeConfig:
  """Tests for OctreeConfig."""

  def test_default_config(self):
    """Test default configuration."""
    config = OctreeConfig()

    assert config.max_depth == 10
    assert config.max_points_per_node == 10
    assert config.min_size == 0.1

  def test_custom_config(self):
    """Test custom configuration."""
    config = OctreeConfig(max_depth=5, max_points_per_node=20)

    assert config.max_depth == 5
    assert config.max_points_per_node == 20


class TestOctreeInit:
  """Tests for Octree initialization."""

  def test_default_init(self):
    """Test default initialization."""
    bounds = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([100.0, 100.0, 100.0]),
    )
    tree = Octree(bounds)

    assert tree.count == 0
    assert tree.root is not None
    assert tree.root.is_leaf

  def test_custom_config_init(self):
    """Test initialization with custom config."""
    bounds = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([100.0, 100.0, 100.0]),
    )
    config = OctreeConfig(max_depth=5)
    tree = Octree(bounds, config)

    assert tree.config.max_depth == 5


class TestOctreeInsert:
  """Tests for point insertion."""

  def test_insert_single_point(self):
    """Test inserting a single point."""
    bounds = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([100.0, 100.0, 100.0]),
    )
    tree = Octree(bounds)

    result = tree.insert(np.array([50.0, 50.0, 50.0]))

    assert result
    assert tree.count == 1

  def test_insert_outside_bounds(self):
    """Test inserting point outside bounds."""
    bounds = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([100.0, 100.0, 100.0]),
    )
    tree = Octree(bounds)

    result = tree.insert(np.array([200.0, 50.0, 50.0]))

    assert not result
    assert tree.count == 0

  def test_insert_with_data(self):
    """Test inserting point with associated data."""
    bounds = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([100.0, 100.0, 100.0]),
    )
    tree = Octree(bounds)

    tree.insert(np.array([50.0, 50.0, 50.0]), data={"id": 1})

    results = tree.query_range(bounds)
    assert len(results) == 1
    assert results[0][1]["id"] == 1

  def test_insert_many_causes_split(self):
    """Test inserting many points causes node split."""
    bounds = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([100.0, 100.0, 100.0]),
    )
    config = OctreeConfig(max_points_per_node=5)
    tree = Octree(bounds, config)

    # Insert more than max_points_per_node
    for i in range(10):
      point = np.array([50.0 + i, 50.0, 50.0])
      tree.insert(point)

    assert tree.count == 10
    # Root should no longer be a leaf
    assert not tree.root.is_leaf


class TestOctreeRangeQuery:
  """Tests for range queries."""

  def test_query_empty_tree(self):
    """Test range query on empty tree."""
    bounds = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([100.0, 100.0, 100.0]),
    )
    tree = Octree(bounds)

    query = BoundingBox(
      min_corner=np.array([40.0, 40.0, 40.0]),
      max_corner=np.array([60.0, 60.0, 60.0]),
    )
    results = tree.query_range(query)

    assert len(results) == 0

  def test_query_finds_points_in_range(self):
    """Test range query finds points."""
    bounds = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([100.0, 100.0, 100.0]),
    )
    tree = Octree(bounds)

    # Insert points
    tree.insert(np.array([25.0, 25.0, 25.0]))  # Outside query
    tree.insert(np.array([50.0, 50.0, 50.0]))  # Inside query
    tree.insert(np.array([55.0, 55.0, 55.0]))  # Inside query
    tree.insert(np.array([75.0, 75.0, 75.0]))  # Outside query

    query = BoundingBox(
      min_corner=np.array([40.0, 40.0, 40.0]),
      max_corner=np.array([60.0, 60.0, 60.0]),
    )
    results = tree.query_range(query)

    assert len(results) == 2

  def test_query_excludes_points_outside_range(self):
    """Test range query excludes points outside."""
    bounds = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([100.0, 100.0, 100.0]),
    )
    tree = Octree(bounds)

    tree.insert(np.array([10.0, 10.0, 10.0]))
    tree.insert(np.array([90.0, 90.0, 90.0]))

    query = BoundingBox(
      min_corner=np.array([40.0, 40.0, 40.0]),
      max_corner=np.array([60.0, 60.0, 60.0]),
    )
    results = tree.query_range(query)

    assert len(results) == 0


class TestOctreeRadiusQuery:
  """Tests for radius queries."""

  def test_radius_query(self):
    """Test radius query finds nearby points."""
    bounds = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([100.0, 100.0, 100.0]),
    )
    tree = Octree(bounds)

    tree.insert(np.array([50.0, 50.0, 50.0]))  # At center
    tree.insert(np.array([51.0, 50.0, 50.0]))  # 1m away
    tree.insert(np.array([60.0, 50.0, 50.0]))  # 10m away

    results = tree.query_radius(np.array([50.0, 50.0, 50.0]), radius=5.0)

    assert len(results) == 2
    # Results should include distance
    distances = [r[2] for r in results]
    assert min(distances) == 0.0  # Point at center
    assert max(distances) == 1.0  # Point 1m away


class TestOctreeKNN:
  """Tests for k-nearest neighbor queries."""

  def test_knn_single(self):
    """Test finding single nearest neighbor."""
    bounds = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([100.0, 100.0, 100.0]),
    )
    tree = Octree(bounds)

    tree.insert(np.array([10.0, 10.0, 10.0]))
    tree.insert(np.array([50.0, 50.0, 50.0]))
    tree.insert(np.array([90.0, 90.0, 90.0]))

    results = tree.query_knn(np.array([51.0, 50.0, 50.0]), k=1)

    assert len(results) == 1
    np.testing.assert_array_almost_equal(results[0][0], [50, 50, 50])

  def test_knn_multiple(self):
    """Test finding multiple nearest neighbors."""
    bounds = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([100.0, 100.0, 100.0]),
    )
    tree = Octree(bounds)

    for i in range(10):
      tree.insert(np.array([float(i * 10), 50.0, 50.0]))

    results = tree.query_knn(np.array([55.0, 50.0, 50.0]), k=3)

    assert len(results) == 3
    # Should be sorted by distance
    distances = [r[2] for r in results]
    assert distances == sorted(distances)

  def test_knn_k_larger_than_points(self):
    """Test KNN when k > number of points."""
    bounds = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([100.0, 100.0, 100.0]),
    )
    tree = Octree(bounds)

    tree.insert(np.array([50.0, 50.0, 50.0]))
    tree.insert(np.array([60.0, 50.0, 50.0]))

    results = tree.query_knn(np.array([55.0, 50.0, 50.0]), k=10)

    assert len(results) == 2  # Only 2 points in tree


class TestOctreeUtilities:
  """Tests for utility functions."""

  def test_clear(self):
    """Test clearing the tree."""
    bounds = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([100.0, 100.0, 100.0]),
    )
    tree = Octree(bounds)

    for i in range(10):
      tree.insert(np.array([float(i * 10), 50.0, 50.0]))

    assert tree.count == 10

    tree.clear()

    assert tree.count == 0
    assert tree.root.is_leaf

  def test_build_from_points(self):
    """Test building tree from point array."""
    bounds = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([100.0, 100.0, 100.0]),
    )
    tree = Octree(bounds)

    points = np.array(
      [
        [10.0, 10.0, 10.0],
        [50.0, 50.0, 50.0],
        [90.0, 90.0, 90.0],
        [200.0, 200.0, 200.0],  # Outside bounds
      ]
    )

    count = tree.build_from_points(points)

    assert count == 3  # Only 3 inside bounds
    assert tree.count == 3

  def test_get_statistics(self):
    """Test statistics collection."""
    bounds = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([100.0, 100.0, 100.0]),
    )
    config = OctreeConfig(max_points_per_node=5)
    tree = Octree(bounds, config)

    for i in range(20):
      tree.insert(np.array([float(i * 5), 50.0, 50.0]))

    stats = tree.get_statistics()

    assert stats["count"] == 20
    assert "max_depth" in stats
    assert "node_count" in stats
    assert "leaf_count" in stats
    assert "avg_points_per_leaf" in stats


class TestCreateOctreeFromBounds:
  """Tests for factory function."""

  def test_create_octree_from_bounds(self):
    """Test factory function."""
    tree = create_octree_from_bounds(
      x_min=-50.0,
      x_max=50.0,
      y_min=-25.0,
      y_max=25.0,
      z_min=-2.0,
      z_max=8.0,
    )

    assert tree is not None
    assert tree.count == 0

    # Verify bounds
    np.testing.assert_array_equal(
      tree.root.bounds.min_corner,
      [-50, -25, -2],
    )
    np.testing.assert_array_equal(
      tree.root.bounds.max_corner,
      [50, 25, 8],
    )

  def test_create_with_config(self):
    """Test factory function with config."""
    config = OctreeConfig(max_depth=5)
    tree = create_octree_from_bounds(
      x_min=0.0,
      x_max=100.0,
      y_min=0.0,
      y_max=100.0,
      z_min=0.0,
      z_max=100.0,
      config=config,
    )

    assert tree.config.max_depth == 5


class TestOctreePerformance:
  """Tests for performance characteristics."""

  def test_large_point_cloud(self):
    """Test with larger point cloud."""
    bounds = BoundingBox(
      min_corner=np.array([-100.0, -100.0, -10.0]),
      max_corner=np.array([100.0, 100.0, 10.0]),
    )
    tree = Octree(bounds)

    # Insert 1000 random points
    np.random.seed(42)
    points = np.random.uniform(
      low=[-90, -90, -5],
      high=[90, 90, 5],
      size=(1000, 3),
    )
    tree.build_from_points(points)

    assert tree.count == 1000

    # Range query should be efficient
    query = BoundingBox(
      min_corner=np.array([0.0, 0.0, -5.0]),
      max_corner=np.array([20.0, 20.0, 5.0]),
    )
    results = tree.query_range(query)
    # Should find some points (not empty, not all)
    assert 0 < len(results) < 1000

  def test_query_vs_brute_force(self):
    """Test that query results match brute force."""
    bounds = BoundingBox(
      min_corner=np.array([0.0, 0.0, 0.0]),
      max_corner=np.array([100.0, 100.0, 100.0]),
    )
    tree = Octree(bounds)

    # Insert points
    np.random.seed(123)
    points = np.random.uniform(low=1, high=99, size=(100, 3))
    tree.build_from_points(points)

    # Query
    query = BoundingBox(
      min_corner=np.array([40.0, 40.0, 40.0]),
      max_corner=np.array([60.0, 60.0, 60.0]),
    )
    octree_results = tree.query_range(query)

    # Brute force
    brute_force = []
    for p in points:
      if query.contains(p):
        brute_force.append(p)

    assert len(octree_results) == len(brute_force)
