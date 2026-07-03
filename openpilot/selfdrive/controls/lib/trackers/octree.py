"""Octree spatial index for 3D point queries.

Provides efficient spatial queries for 3D point clouds:
- Range queries (find all points in a box)
- K-nearest neighbor queries
- Radius queries (find points within distance)

Useful for collision detection and spatial association in tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class OctreeConfig:
  """Configuration for octree.

  Attributes:
    max_depth: Maximum tree depth (limits memory)
    max_points_per_node: Split threshold
    min_size: Minimum node size (meters)
  """

  max_depth: int = 10
  max_points_per_node: int = 10
  min_size: float = 0.1


@dataclass
class BoundingBox:
  """Axis-aligned bounding box.

  Attributes:
    min_corner: [x_min, y_min, z_min]
    max_corner: [x_max, y_max, z_max]
  """

  min_corner: np.ndarray
  max_corner: np.ndarray

  @property
  def center(self) -> np.ndarray:
    """Get box center."""
    return (self.min_corner + self.max_corner) / 2

  @property
  def size(self) -> np.ndarray:
    """Get box size in each dimension."""
    return self.max_corner - self.min_corner

  def contains(self, point: np.ndarray) -> bool:
    """Check if point is inside box."""
    return bool(np.all(point >= self.min_corner) and np.all(point <= self.max_corner))

  def intersects(self, other: BoundingBox) -> bool:
    """Check if boxes intersect."""
    return bool(np.all(self.min_corner <= other.max_corner) and np.all(self.max_corner >= other.min_corner))

  def get_octant(self, point: np.ndarray) -> int:
    """Get octant index for a point (0-7).

    Octant numbering:
      0: ---  (x<c, y<c, z<c)
      1: +--  (x>=c, y<c, z<c)
      2: -+-  (x<c, y>=c, z<c)
      3: ++-  (x>=c, y>=c, z<c)
      4: --+  (x<c, y<c, z>=c)
      5: +-+  (x>=c, y<c, z>=c)
      6: -++  (x<c, y>=c, z>=c)
      7: +++  (x>=c, y>=c, z>=c)
    """
    c = self.center
    octant = 0
    if point[0] >= c[0]:
      octant |= 1
    if point[1] >= c[1]:
      octant |= 2
    if point[2] >= c[2]:
      octant |= 4
    return octant

  def get_octant_bounds(self, octant: int) -> BoundingBox:
    """Get bounding box for an octant.

    Args:
      octant: Octant index (0-7)

    Returns:
      Bounding box for the octant
    """
    c = self.center
    new_min = self.min_corner.copy()
    new_max = self.max_corner.copy()

    if octant & 1:  # x >= center
      new_min[0] = c[0]
    else:
      new_max[0] = c[0]

    if octant & 2:  # y >= center
      new_min[1] = c[1]
    else:
      new_max[1] = c[1]

    if octant & 4:  # z >= center
      new_min[2] = c[2]
    else:
      new_max[2] = c[2]

    return BoundingBox(new_min, new_max)


@dataclass
class OctreeNode:
  """Node in the octree.

  Attributes:
    bounds: Bounding box of this node
    points: Points stored in this leaf node
    point_data: Optional data associated with each point
    children: Child nodes (8 for internal nodes)
    depth: Depth in tree (root = 0)
  """

  bounds: BoundingBox
  depth: int = 0
  points: list[np.ndarray] = field(default_factory=list)
  point_data: list[Any] = field(default_factory=list)
  children: list[OctreeNode | None] = field(default_factory=lambda: [None] * 8)

  @property
  def is_leaf(self) -> bool:
    """Check if this is a leaf node."""
    return all(c is None for c in self.children)


class Octree:
  """Octree spatial index for 3D points.

  Provides efficient spatial queries for point clouds.

  Usage:
    # Create octree for a 100x100x10 meter region
    bounds = BoundingBox(
      min_corner=np.array([-50, -50, -2]),
      max_corner=np.array([50, 50, 8])
    )
    tree = Octree(bounds)

    # Insert points
    for point in points:
      tree.insert(point)

    # Query points in region
    query_box = BoundingBox(
      min_corner=np.array([0, 0, 0]),
      max_corner=np.array([10, 10, 2])
    )
    found = tree.query_range(query_box)

    # Find k nearest neighbors
    neighbors = tree.query_knn(np.array([5, 5, 1]), k=5)
  """

  def __init__(
    self,
    bounds: BoundingBox,
    config: OctreeConfig | None = None,
  ):
    """Initialize octree.

    Args:
      bounds: Bounding box for the entire tree
      config: Optional configuration
    """
    self.config = config or OctreeConfig()
    self.root = OctreeNode(bounds=bounds)
    self._count = 0

  def insert(self, point: np.ndarray, data: Any = None) -> bool:
    """Insert a point into the tree.

    Args:
      point: [x, y, z] coordinates
      data: Optional data to associate with point

    Returns:
      True if inserted, False if outside bounds
    """
    if not self.root.bounds.contains(point):
      return False

    self._insert_recursive(self.root, point, data)
    self._count += 1
    return True

  def _insert_recursive(
    self,
    node: OctreeNode,
    point: np.ndarray,
    data: Any,
  ) -> None:
    """Recursively insert point into tree.

    Args:
      node: Current node
      point: Point to insert
      data: Associated data
    """
    config = self.config

    # If leaf and under capacity, add here
    if node.is_leaf:
      if len(node.points) < config.max_points_per_node or node.depth >= config.max_depth or np.min(node.bounds.size) < config.min_size:
        node.points.append(point)
        node.point_data.append(data)
        return

      # Need to split - redistribute existing points
      self._split_node(node)

    # Insert into appropriate child
    octant = node.bounds.get_octant(point)
    if node.children[octant] is None:
      child_bounds = node.bounds.get_octant_bounds(octant)
      node.children[octant] = OctreeNode(bounds=child_bounds, depth=node.depth + 1)

    self._insert_recursive(node.children[octant], point, data)

  def _split_node(self, node: OctreeNode) -> None:
    """Split a leaf node into 8 children.

    Args:
      node: Node to split
    """
    # Create children and redistribute points
    old_points = node.points
    old_data = node.point_data
    node.points = []
    node.point_data = []

    for point, data in zip(old_points, old_data, strict=False):
      octant = node.bounds.get_octant(point)
      if node.children[octant] is None:
        child_bounds = node.bounds.get_octant_bounds(octant)
        node.children[octant] = OctreeNode(bounds=child_bounds, depth=node.depth + 1)

      # Add directly to child (no recursion to avoid deep splits)
      node.children[octant].points.append(point)
      node.children[octant].point_data.append(data)

  def query_range(self, query_box: BoundingBox) -> list[tuple[np.ndarray, Any]]:
    """Find all points within a bounding box.

    Args:
      query_box: Box to search

    Returns:
      List of (point, data) tuples
    """
    results: list[tuple[np.ndarray, Any]] = []
    self._query_range_recursive(self.root, query_box, results)
    return results

  def _query_range_recursive(
    self,
    node: OctreeNode,
    query_box: BoundingBox,
    results: list[tuple[np.ndarray, Any]],
  ) -> None:
    """Recursively query points in range.

    Args:
      node: Current node
      query_box: Search box
      results: List to append results
    """
    if not node.bounds.intersects(query_box):
      return

    # Check points in this node
    for point, data in zip(node.points, node.point_data, strict=False):
      if query_box.contains(point):
        results.append((point, data))

    # Recurse into children
    for child in node.children:
      if child is not None:
        self._query_range_recursive(child, query_box, results)

  def query_radius(
    self,
    center: np.ndarray,
    radius: float,
  ) -> list[tuple[np.ndarray, Any, float]]:
    """Find all points within radius of a point.

    Args:
      center: Query point
      radius: Search radius

    Returns:
      List of (point, data, distance) tuples
    """
    # Use bounding box to narrow search
    query_box = BoundingBox(
      min_corner=center - radius,
      max_corner=center + radius,
    )

    candidates = self.query_range(query_box)

    # Filter by actual distance
    results: list[tuple[np.ndarray, Any, float]] = []
    radius_sq = radius**2

    for point, data in candidates:
      dist_sq = float(np.sum((point - center) ** 2))
      if dist_sq <= radius_sq:
        results.append((point, data, np.sqrt(dist_sq)))

    return results

  def query_knn(
    self,
    point: np.ndarray,
    k: int = 1,
  ) -> list[tuple[np.ndarray, Any, float]]:
    """Find k nearest neighbors to a point.

    Args:
      point: Query point
      k: Number of neighbors to find

    Returns:
      List of (point, data, distance) tuples, sorted by distance
    """
    # Collect all points with distances
    all_results: list[tuple[np.ndarray, Any, float]] = []
    self._collect_all_points(self.root, point, all_results)

    # Sort by distance and take top k
    all_results.sort(key=lambda x: x[2])
    return all_results[:k]

  def _collect_all_points(
    self,
    node: OctreeNode,
    query_point: np.ndarray,
    results: list[tuple[np.ndarray, Any, float]],
  ) -> None:
    """Collect all points with distances.

    For small trees, brute force is acceptable.
    For large trees, a priority queue approach would be better.

    Args:
      node: Current node
      query_point: Point to measure distance from
      results: List to append results
    """
    for point, data in zip(node.points, node.point_data, strict=False):
      dist = float(np.sqrt(np.sum((point - query_point) ** 2)))
      results.append((point, data, dist))

    for child in node.children:
      if child is not None:
        self._collect_all_points(child, query_point, results)

  def clear(self) -> None:
    """Remove all points from tree."""
    self.root = OctreeNode(bounds=self.root.bounds)
    self._count = 0

  def build_from_points(
    self,
    points: np.ndarray,
    data_list: list[Any] | None = None,
  ) -> int:
    """Build tree from array of points.

    Args:
      points: [N, 3] array of points
      data_list: Optional list of data for each point

    Returns:
      Number of points inserted
    """
    if data_list is None:
      data_list = [None] * len(points)

    count = 0
    for point, data in zip(points, data_list, strict=False):
      if self.insert(point, data):
        count += 1

    return count

  @property
  def count(self) -> int:
    """Number of points in tree."""
    return self._count

  def get_statistics(self) -> dict[str, Any]:
    """Get tree statistics.

    Returns:
      Dictionary with tree statistics
    """
    stats = {
      "count": self._count,
      "max_depth": 0,
      "node_count": 0,
      "leaf_count": 0,
      "internal_count": 0,
      "points_per_leaf": [],
    }
    self._collect_stats(self.root, stats)

    if stats["points_per_leaf"]:
      stats["avg_points_per_leaf"] = np.mean(stats["points_per_leaf"])
    else:
      stats["avg_points_per_leaf"] = 0.0

    return stats

  def _collect_stats(self, node: OctreeNode, stats: dict[str, Any]) -> None:
    """Recursively collect statistics.

    Args:
      node: Current node
      stats: Statistics dictionary to update
    """
    stats["node_count"] += 1
    stats["max_depth"] = max(stats["max_depth"], node.depth)

    if node.is_leaf:
      stats["leaf_count"] += 1
      stats["points_per_leaf"].append(len(node.points))
    else:
      stats["internal_count"] += 1

    for child in node.children:
      if child is not None:
        self._collect_stats(child, stats)


def create_octree_from_bounds(
  x_min: float,
  x_max: float,
  y_min: float,
  y_max: float,
  z_min: float,
  z_max: float,
  config: OctreeConfig | None = None,
) -> Octree:
  """Create an octree with specified bounds.

  Args:
    x_min, x_max: X range
    y_min, y_max: Y range
    z_min, z_max: Z range
    config: Optional configuration

  Returns:
    New Octree instance
  """
  bounds = BoundingBox(
    min_corner=np.array([x_min, y_min, z_min]),
    max_corner=np.array([x_max, y_max, z_max]),
  )
  return Octree(bounds, config)
