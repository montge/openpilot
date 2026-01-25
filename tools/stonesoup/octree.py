"""Octree Spatial Index for efficient 3D point queries.

Implements an octree data structure for fast range queries and
k-nearest-neighbor searches in 3D space.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BoundingBox:
  """Axis-aligned bounding box in 3D."""
  min_corner: np.ndarray
  max_corner: np.ndarray

  @property
  def center(self) -> np.ndarray:
    return (self.min_corner + self.max_corner) / 2

  @property
  def size(self) -> np.ndarray:
    return self.max_corner - self.min_corner

  def contains_point(self, point: np.ndarray) -> bool:
    """Check if point is inside or on boundary of box."""
    return np.all(point >= self.min_corner) and np.all(point <= self.max_corner)

  def intersects(self, other: "BoundingBox") -> bool:
    """Check if this box intersects another box."""
    return (
      np.all(self.min_corner <= other.max_corner) and
      np.all(self.max_corner >= other.min_corner)
    )

  def intersects_sphere(self, center: np.ndarray, radius: float) -> bool:
    """Check if box intersects a sphere."""
    # Find closest point on box to sphere center
    closest = np.clip(center, self.min_corner, self.max_corner)
    distance = np.linalg.norm(center - closest)
    return distance <= radius

  def split(self) -> list["BoundingBox"]:
    """Split box into 8 octants."""
    center = self.center
    octants = []

    for i in range(8):
      # Binary representation of i gives octant location
      new_min = np.array([
        self.min_corner[0] if (i & 1) == 0 else center[0],
        self.min_corner[1] if (i & 2) == 0 else center[1],
        self.min_corner[2] if (i & 4) == 0 else center[2]
      ])
      new_max = np.array([
        center[0] if (i & 1) == 0 else self.max_corner[0],
        center[1] if (i & 2) == 0 else self.max_corner[1],
        center[2] if (i & 4) == 0 else self.max_corner[2]
      ])
      octants.append(BoundingBox(new_min, new_max))

    return octants


@dataclass
class OctreeNode:
  """Node in the octree."""
  bounds: BoundingBox
  points: list[tuple[np.ndarray, int]] = field(default_factory=list)  # (point, id)
  children: list["OctreeNode"] = field(default_factory=list)
  is_leaf: bool = True

  @property
  def n_points(self) -> int:
    """Total points in this node and children."""
    if self.is_leaf:
      return len(self.points)
    return sum(child.n_points for child in self.children)


@dataclass
class OctreeConfig:
  """Configuration for octree."""
  max_points_per_node: int = 8  # Split when exceeded
  max_depth: int = 10  # Maximum tree depth
  min_size: float = 0.1  # Minimum node size


class Octree:
  """Octree spatial index for 3D points.

  Provides efficient:
  - Point insertion: O(log n) average
  - Range queries: O(k + log n) where k is result size
  - K-nearest neighbor: O(k log n)
  """

  def __init__(
    self,
    bounds: BoundingBox | None = None,
    config: OctreeConfig | None = None
  ):
    self.config = config or OctreeConfig()

    if bounds is None:
      # Default bounds for vehicle tracking
      bounds = BoundingBox(
        min_corner=np.array([0.0, -50.0, -5.0]),
        max_corner=np.array([200.0, 50.0, 10.0])
      )

    self.root = OctreeNode(bounds=bounds)
    self.next_id = 0
    self._point_map: dict[int, np.ndarray] = {}  # id -> point

  @property
  def n_points(self) -> int:
    """Total number of points in tree."""
    return self.root.n_points

  def _get_octant_index(self, point: np.ndarray, center: np.ndarray) -> int:
    """Get octant index for point relative to center."""
    idx = 0
    if point[0] >= center[0]:
      idx |= 1
    if point[1] >= center[1]:
      idx |= 2
    if point[2] >= center[2]:
      idx |= 4
    return idx

  def _split_node(self, node: OctreeNode, depth: int) -> None:
    """Split a leaf node into 8 children."""
    if not node.is_leaf or depth >= self.config.max_depth:
      return

    # Check minimum size
    if np.min(node.bounds.size) < self.config.min_size * 2:
      return

    # Create children
    octant_bounds = node.bounds.split()
    node.children = [OctreeNode(bounds=b) for b in octant_bounds]
    node.is_leaf = False

    # Redistribute points
    for point, point_id in node.points:
      octant_idx = self._get_octant_index(point, node.bounds.center)
      node.children[octant_idx].points.append((point, point_id))

    node.points = []

    # Recursively split children if needed
    for child in node.children:
      if len(child.points) > self.config.max_points_per_node:
        self._split_node(child, depth + 1)

  def insert(self, point: np.ndarray, point_id: int | None = None) -> int:
    """Insert a point into the tree.

    Args:
      point: 3D point coordinates
      point_id: Optional ID for the point

    Returns:
      ID of inserted point
    """
    if point_id is None:
      point_id = self.next_id
      self.next_id += 1

    self._point_map[point_id] = point.copy()
    self._insert_recursive(self.root, point, point_id, 0)
    return point_id

  def _insert_recursive(
    self,
    node: OctreeNode,
    point: np.ndarray,
    point_id: int,
    depth: int
  ) -> None:
    """Recursively insert point into tree."""
    if not node.bounds.contains_point(point):
      return

    if node.is_leaf:
      node.points.append((point, point_id))

      if len(node.points) > self.config.max_points_per_node:
        self._split_node(node, depth)
    else:
      octant_idx = self._get_octant_index(point, node.bounds.center)
      self._insert_recursive(node.children[octant_idx], point, point_id, depth + 1)

  def insert_points(self, points: np.ndarray) -> list[int]:
    """Insert multiple points at once.

    Args:
      points: Nx3 array of points

    Returns:
      List of point IDs
    """
    ids = []
    for point in points:
      ids.append(self.insert(point))
    return ids

  def range_query(self, query_box: BoundingBox) -> list[tuple[np.ndarray, int]]:
    """Find all points within a bounding box.

    Args:
      query_box: Bounding box to search

    Returns:
      List of (point, id) tuples within box
    """
    results: list[tuple[np.ndarray, int]] = []
    self._range_query_recursive(self.root, query_box, results)
    return results

  def _range_query_recursive(
    self,
    node: OctreeNode,
    query_box: BoundingBox,
    results: list[tuple[np.ndarray, int]]
  ) -> None:
    """Recursively search for points in range."""
    if not node.bounds.intersects(query_box):
      return

    if node.is_leaf:
      for point, point_id in node.points:
        if query_box.contains_point(point):
          results.append((point, point_id))
    else:
      for child in node.children:
        self._range_query_recursive(child, query_box, results)

  def radius_query(
    self,
    center: np.ndarray,
    radius: float
  ) -> list[tuple[np.ndarray, int, float]]:
    """Find all points within radius of center.

    Args:
      center: Query point
      radius: Search radius

    Returns:
      List of (point, id, distance) tuples
    """
    results: list[tuple[np.ndarray, int, float]] = []
    self._radius_query_recursive(self.root, center, radius, results)
    return sorted(results, key=lambda x: x[2])

  def _radius_query_recursive(
    self,
    node: OctreeNode,
    center: np.ndarray,
    radius: float,
    results: list[tuple[np.ndarray, int, float]]
  ) -> None:
    """Recursively search for points within radius."""
    if not node.bounds.intersects_sphere(center, radius):
      return

    if node.is_leaf:
      for point, point_id in node.points:
        dist = np.linalg.norm(point - center)
        if dist <= radius:
          results.append((point, point_id, dist))
    else:
      for child in node.children:
        self._radius_query_recursive(child, center, radius, results)

  def k_nearest(
    self,
    center: np.ndarray,
    k: int
  ) -> list[tuple[np.ndarray, int, float]]:
    """Find k nearest neighbors to center point.

    Args:
      center: Query point
      k: Number of neighbors to find

    Returns:
      List of (point, id, distance) tuples, sorted by distance
    """
    # Use priority queue approach
    candidates: list[tuple[float, np.ndarray, int]] = []
    self._knn_recursive(self.root, center, k, candidates)

    # Sort and return top k
    candidates.sort(key=lambda x: x[0])
    return [(p, pid, d) for d, p, pid in candidates[:k]]

  def _knn_recursive(
    self,
    node: OctreeNode,
    center: np.ndarray,
    k: int,
    candidates: list[tuple[float, np.ndarray, int]]
  ) -> None:
    """Recursively search for k nearest neighbors."""
    # Current max distance in candidates
    max_dist = candidates[-1][0] if len(candidates) >= k else float('inf')

    # Prune if node is too far
    if not node.bounds.intersects_sphere(center, max_dist):
      return

    if node.is_leaf:
      for point, point_id in node.points:
        dist = np.linalg.norm(point - center)
        if len(candidates) < k or dist < candidates[-1][0]:
          candidates.append((dist, point, point_id))
          candidates.sort(key=lambda x: x[0])
          if len(candidates) > k:
            candidates.pop()
    else:
      # Sort children by distance to query point for better pruning
      child_dists = []
      for i, child in enumerate(node.children):
        closest = np.clip(center, child.bounds.min_corner, child.bounds.max_corner)
        dist = np.linalg.norm(center - closest)
        child_dists.append((dist, i))

      child_dists.sort(key=lambda x: x[0])

      for _, child_idx in child_dists:
        self._knn_recursive(node.children[child_idx], center, k, candidates)

  def clear(self) -> None:
    """Remove all points from tree."""
    self.root = OctreeNode(bounds=self.root.bounds)
    self.next_id = 0
    self._point_map.clear()


def brute_force_radius_query(
  points: np.ndarray,
  center: np.ndarray,
  radius: float
) -> list[tuple[int, float]]:
  """Brute force radius query for comparison."""
  results = []
  for i, point in enumerate(points):
    dist = np.linalg.norm(point - center)
    if dist <= radius:
      results.append((i, dist))
  return sorted(results, key=lambda x: x[1])


def brute_force_knn(
  points: np.ndarray,
  center: np.ndarray,
  k: int
) -> list[tuple[int, float]]:
  """Brute force KNN for comparison."""
  distances = [(i, np.linalg.norm(p - center)) for i, p in enumerate(points)]
  distances.sort(key=lambda x: x[1])
  return distances[:k]


@dataclass
class BenchmarkResult:
  """Benchmark result for octree operations."""
  method: str
  n_points: int
  operation: str
  time_ms: float
  n_results: int


def benchmark_octree(
  n_points: int = 10000,
  n_queries: int = 100,
  k: int = 10,
  radius: float = 5.0
) -> list[BenchmarkResult]:
  """Benchmark octree vs brute force."""
  import time

  results = []

  # Generate random points
  np.random.seed(42)
  points = np.column_stack([
    np.random.uniform(0, 100, n_points),
    np.random.uniform(-30, 30, n_points),
    np.random.uniform(-2, 5, n_points)
  ])

  # Build octree
  bounds = BoundingBox(
    min_corner=np.array([0.0, -30.0, -2.0]),
    max_corner=np.array([100.0, 30.0, 5.0])
  )
  tree = Octree(bounds)

  start = time.monotonic()
  tree.insert_points(points)
  build_time = (time.monotonic() - start) * 1000

  results.append(BenchmarkResult(
    method="Octree",
    n_points=n_points,
    operation="Build",
    time_ms=build_time,
    n_results=n_points
  ))

  # Generate query points
  query_centers = np.column_stack([
    np.random.uniform(0, 100, n_queries),
    np.random.uniform(-30, 30, n_queries),
    np.random.uniform(-2, 5, n_queries)
  ])

  # Benchmark radius queries
  start = time.monotonic()
  total_results = 0
  for center in query_centers:
    res = tree.radius_query(center, radius)
    total_results += len(res)
  octree_radius_time = (time.monotonic() - start) * 1000

  results.append(BenchmarkResult(
    method="Octree",
    n_points=n_points,
    operation=f"Radius Query (r={radius})",
    time_ms=octree_radius_time,
    n_results=total_results // n_queries
  ))

  start = time.monotonic()
  total_results = 0
  for center in query_centers:
    res = brute_force_radius_query(points, center, radius)
    total_results += len(res)
  brute_radius_time = (time.monotonic() - start) * 1000

  results.append(BenchmarkResult(
    method="Brute Force",
    n_points=n_points,
    operation=f"Radius Query (r={radius})",
    time_ms=brute_radius_time,
    n_results=total_results // n_queries
  ))

  # Benchmark KNN queries
  start = time.monotonic()
  for center in query_centers:
    tree.k_nearest(center, k)
  octree_knn_time = (time.monotonic() - start) * 1000

  results.append(BenchmarkResult(
    method="Octree",
    n_points=n_points,
    operation=f"KNN (k={k})",
    time_ms=octree_knn_time,
    n_results=k
  ))

  start = time.monotonic()
  for center in query_centers:
    brute_force_knn(points, center, k)
  brute_knn_time = (time.monotonic() - start) * 1000

  results.append(BenchmarkResult(
    method="Brute Force",
    n_points=n_points,
    operation=f"KNN (k={k})",
    time_ms=brute_knn_time,
    n_results=k
  ))

  return results


def format_benchmark_report(results: list[BenchmarkResult]) -> str:
  """Format benchmark results as markdown."""
  lines = [
    "# Octree Benchmark Results",
    "",
    "## Configuration",
    f"- Points: {results[0].n_points}",
    "",
    "## Results",
    "",
    "| Method | Operation | Time (ms) | Avg Results |",
    "|--------|-----------|-----------|-------------|",
  ]

  for r in results:
    lines.append(f"| {r.method} | {r.operation} | {r.time_ms:.2f} | {r.n_results} |")

  # Calculate speedups
  lines.extend(["", "## Speedups"])

  radius_octree = [r for r in results if r.method == "Octree" and "Radius" in r.operation]
  radius_brute = [r for r in results if r.method == "Brute Force" and "Radius" in r.operation]
  if radius_octree and radius_brute:
    speedup = radius_brute[0].time_ms / radius_octree[0].time_ms
    lines.append(f"- Radius Query: {speedup:.1f}x")

  knn_octree = [r for r in results if r.method == "Octree" and "KNN" in r.operation]
  knn_brute = [r for r in results if r.method == "Brute Force" and "KNN" in r.operation]
  if knn_octree and knn_brute:
    speedup = knn_brute[0].time_ms / knn_octree[0].time_ms
    lines.append(f"- KNN: {speedup:.1f}x")

  return "\n".join(lines)


if __name__ == "__main__":
  # Demo octree
  print("Building octree...")
  bounds = BoundingBox(
    min_corner=np.array([0.0, -20.0, -2.0]),
    max_corner=np.array([100.0, 20.0, 5.0])
  )
  tree = Octree(bounds)

  # Insert some points
  np.random.seed(42)
  points = np.random.randn(1000, 3) * 10 + np.array([50, 0, 1])
  tree.insert_points(points)
  print(f"Inserted {tree.n_points} points")

  # Query
  center = np.array([50.0, 0.0, 1.0])
  neighbors = tree.k_nearest(center, 5)
  print(f"\n5 nearest to {center}:")
  for p, pid, d in neighbors:
    print(f"  Point {pid}: {p}, dist={d:.3f}")

  # Benchmark
  print("\n" + format_benchmark_report(benchmark_octree(5000, 50)))
