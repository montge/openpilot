"""Voxel Grid Tracker for occupancy-based multi-target tracking.

Implements a 3D voxel grid with log-odds occupancy updates for
tracking objects in the vehicle's environment.
"""
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VoxelGridConfig:
  """Configuration for voxel grid."""
  x_range: tuple[float, float] = (0.0, 100.0)  # Forward range in meters
  y_range: tuple[float, float] = (-20.0, 20.0)  # Lateral range in meters
  z_range: tuple[float, float] = (-2.0, 3.0)  # Height range in meters
  resolution: float = 0.5  # Voxel size in meters

  # Log-odds update parameters
  l_occ: float = 0.85  # Log-odds for occupied
  l_free: float = -0.4  # Log-odds for free
  l_min: float = -2.0  # Minimum log-odds clamp
  l_max: float = 3.5  # Maximum log-odds clamp

  # Thresholds
  occupied_threshold: float = 0.7  # Probability threshold for occupied
  free_threshold: float = 0.3  # Probability threshold for free


class VoxelGrid:
  """3D voxel grid with log-odds occupancy representation.

  Uses log-odds for efficient Bayesian occupancy updates without
  multiplication. The log-odds L is related to probability P by:
    L = log(P / (1 - P))
    P = 1 / (1 + exp(-L))
  """

  def __init__(self, config: VoxelGridConfig | None = None):
    self.config = config or VoxelGridConfig()
    c = self.config

    # Compute grid dimensions
    self.nx = int((c.x_range[1] - c.x_range[0]) / c.resolution)
    self.ny = int((c.y_range[1] - c.y_range[0]) / c.resolution)
    self.nz = int((c.z_range[1] - c.z_range[0]) / c.resolution)

    # Initialize grid with log-odds = 0 (probability = 0.5)
    self._grid = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)

    # Origin offset for world-to-voxel conversion
    self._origin = np.array([c.x_range[0], c.y_range[0], c.z_range[0]])

  @property
  def shape(self) -> tuple[int, int, int]:
    return (self.nx, self.ny, self.nz)

  @property
  def n_voxels(self) -> int:
    return self.nx * self.ny * self.nz

  def world_to_voxel(self, point: np.ndarray) -> np.ndarray:
    """Convert world coordinates to voxel indices."""
    voxel = ((point - self._origin) / self.config.resolution).astype(int)
    return voxel

  def voxel_to_world(self, voxel: np.ndarray) -> np.ndarray:
    """Convert voxel indices to world coordinates (voxel center)."""
    world = (voxel + 0.5) * self.config.resolution + self._origin
    return world

  def is_valid_voxel(self, voxel: np.ndarray) -> bool:
    """Check if voxel indices are within grid bounds."""
    return (
      0 <= voxel[0] < self.nx and
      0 <= voxel[1] < self.ny and
      0 <= voxel[2] < self.nz
    )

  def get_log_odds(self, voxel: np.ndarray) -> float:
    """Get log-odds at voxel location."""
    if not self.is_valid_voxel(voxel):
      return 0.0  # Unknown
    return self._grid[voxel[0], voxel[1], voxel[2]]

  def get_probability(self, voxel: np.ndarray) -> float:
    """Get occupancy probability at voxel location."""
    l = self.get_log_odds(voxel)
    return 1.0 / (1.0 + np.exp(-l))

  def update_occupied(self, point: np.ndarray) -> None:
    """Update voxel at world point as occupied."""
    voxel = self.world_to_voxel(point)
    if self.is_valid_voxel(voxel):
      self._grid[voxel[0], voxel[1], voxel[2]] = np.clip(
        self._grid[voxel[0], voxel[1], voxel[2]] + self.config.l_occ,
        self.config.l_min,
        self.config.l_max
      )

  def update_free(self, point: np.ndarray) -> None:
    """Update voxel at world point as free."""
    voxel = self.world_to_voxel(point)
    if self.is_valid_voxel(voxel):
      self._grid[voxel[0], voxel[1], voxel[2]] = np.clip(
        self._grid[voxel[0], voxel[1], voxel[2]] + self.config.l_free,
        self.config.l_min,
        self.config.l_max
      )

  def update_ray(
    self,
    origin: np.ndarray,
    endpoint: np.ndarray,
    include_endpoint: bool = True
  ) -> None:
    """Update voxels along a ray from origin to endpoint.

    Voxels between origin and endpoint are marked as free.
    The endpoint voxel is marked as occupied (if include_endpoint=True).
    """
    # Bresenham-like 3D ray traversal
    direction = endpoint - origin
    distance = np.linalg.norm(direction)

    if distance < self.config.resolution:
      if include_endpoint:
        self.update_occupied(endpoint)
      return

    n_steps = int(distance / self.config.resolution)
    step = direction / n_steps

    # Mark voxels along ray as free (except endpoint)
    for i in range(n_steps - 1):
      point = origin + step * (i + 0.5)
      self.update_free(point)

    # Mark endpoint as occupied
    if include_endpoint:
      self.update_occupied(endpoint)

  def get_occupied_voxels(self) -> np.ndarray:
    """Get indices of all occupied voxels."""
    threshold_logodds = np.log(
      self.config.occupied_threshold / (1 - self.config.occupied_threshold)
    )
    occupied = np.argwhere(self._grid > threshold_logodds)
    return occupied

  def get_occupied_points(self) -> np.ndarray:
    """Get world coordinates of occupied voxel centers."""
    voxels = self.get_occupied_voxels()
    if len(voxels) == 0:
      return np.zeros((0, 3))
    return np.array([self.voxel_to_world(v) for v in voxels])

  def decay(self, factor: float = 0.95) -> None:
    """Apply decay to all voxels (move toward uncertainty)."""
    self._grid *= factor

  def clear(self) -> None:
    """Reset grid to unknown (log-odds = 0)."""
    self._grid.fill(0.0)


class SparseVoxelGrid:
  """Memory-efficient sparse voxel grid using a dictionary.

  Only stores voxels that have been updated, significantly reducing
  memory usage for large environments with sparse occupancy.
  """

  def __init__(self, config: VoxelGridConfig | None = None):
    self.config = config or VoxelGridConfig()
    c = self.config

    self.nx = int((c.x_range[1] - c.x_range[0]) / c.resolution)
    self.ny = int((c.y_range[1] - c.y_range[0]) / c.resolution)
    self.nz = int((c.z_range[1] - c.z_range[0]) / c.resolution)

    self._origin = np.array([c.x_range[0], c.y_range[0], c.z_range[0]])

    # Sparse storage: (x, y, z) -> log_odds
    self._voxels: dict[tuple[int, int, int], float] = {}

  @property
  def shape(self) -> tuple[int, int, int]:
    return (self.nx, self.ny, self.nz)

  @property
  def n_allocated(self) -> int:
    """Number of allocated voxels."""
    return len(self._voxels)

  @property
  def n_voxels(self) -> int:
    """Total possible voxels."""
    return self.nx * self.ny * self.nz

  @property
  def sparsity(self) -> float:
    """Fraction of grid that is sparse (unallocated)."""
    return 1.0 - (self.n_allocated / self.n_voxels)

  def world_to_voxel(self, point: np.ndarray) -> tuple[int, int, int]:
    """Convert world coordinates to voxel key."""
    voxel = ((point - self._origin) / self.config.resolution).astype(int)
    return (int(voxel[0]), int(voxel[1]), int(voxel[2]))

  def voxel_to_world(self, key: tuple[int, int, int]) -> np.ndarray:
    """Convert voxel key to world coordinates."""
    voxel = np.array(key, dtype=float)
    return (voxel + 0.5) * self.config.resolution + self._origin

  def is_valid_voxel(self, key: tuple[int, int, int]) -> bool:
    """Check if voxel key is within bounds."""
    return (
      0 <= key[0] < self.nx and
      0 <= key[1] < self.ny and
      0 <= key[2] < self.nz
    )

  def get_log_odds(self, key: tuple[int, int, int]) -> float:
    """Get log-odds at voxel (0.0 if not allocated)."""
    return self._voxels.get(key, 0.0)

  def get_probability(self, key: tuple[int, int, int]) -> float:
    """Get occupancy probability at voxel."""
    l = self.get_log_odds(key)
    return 1.0 / (1.0 + np.exp(-l))

  def update_occupied(self, point: np.ndarray) -> None:
    """Update voxel at world point as occupied."""
    key = self.world_to_voxel(point)
    if self.is_valid_voxel(key):
      current = self._voxels.get(key, 0.0)
      self._voxels[key] = np.clip(
        current + self.config.l_occ,
        self.config.l_min,
        self.config.l_max
      )

  def update_free(self, point: np.ndarray) -> None:
    """Update voxel at world point as free."""
    key = self.world_to_voxel(point)
    if self.is_valid_voxel(key):
      current = self._voxels.get(key, 0.0)
      new_val = np.clip(
        current + self.config.l_free,
        self.config.l_min,
        self.config.l_max
      )
      # Only store if significantly different from prior
      if abs(new_val) > 0.1:
        self._voxels[key] = new_val
      elif key in self._voxels:
        del self._voxels[key]

  def update_ray(
    self,
    origin: np.ndarray,
    endpoint: np.ndarray,
    include_endpoint: bool = True
  ) -> None:
    """Update voxels along a ray."""
    direction = endpoint - origin
    distance = np.linalg.norm(direction)

    if distance < self.config.resolution:
      if include_endpoint:
        self.update_occupied(endpoint)
      return

    n_steps = int(distance / self.config.resolution)
    step = direction / n_steps

    for i in range(n_steps - 1):
      point = origin + step * (i + 0.5)
      self.update_free(point)

    if include_endpoint:
      self.update_occupied(endpoint)

  def get_occupied_voxels(self) -> list[tuple[int, int, int]]:
    """Get keys of all occupied voxels."""
    threshold_logodds = np.log(
      self.config.occupied_threshold / (1 - self.config.occupied_threshold)
    )
    return [k for k, v in self._voxels.items() if v > threshold_logodds]

  def get_occupied_points(self) -> np.ndarray:
    """Get world coordinates of occupied voxels."""
    voxels = self.get_occupied_voxels()
    if not voxels:
      return np.zeros((0, 3))
    return np.array([self.voxel_to_world(v) for v in voxels])

  def decay(self, factor: float = 0.95) -> None:
    """Apply decay to all voxels."""
    to_remove = []
    for key in self._voxels:
      self._voxels[key] *= factor
      if abs(self._voxels[key]) < 0.1:
        to_remove.append(key)
    for key in to_remove:
      del self._voxels[key]

  def clear(self) -> None:
    """Clear all voxels."""
    self._voxels.clear()


def try_import_cupy() -> Any:
  """Try to import CuPy for GPU acceleration."""
  try:
    import cupy  # type: ignore[import]
    return cupy
  except ImportError:
    return None


class GPUVoxelGrid:
  """GPU-accelerated voxel grid using CuPy (if available).

  Falls back to NumPy if CuPy is not available.
  """

  def __init__(self, config: VoxelGridConfig | None = None):
    self.config = config or VoxelGridConfig()
    c = self.config

    self.nx = int((c.x_range[1] - c.x_range[0]) / c.resolution)
    self.ny = int((c.y_range[1] - c.y_range[0]) / c.resolution)
    self.nz = int((c.z_range[1] - c.z_range[0]) / c.resolution)

    self._origin = np.array([c.x_range[0], c.y_range[0], c.z_range[0]])

    # Try to use CuPy
    self.cp = try_import_cupy()
    self.use_gpu = self.cp is not None

    if self.use_gpu:
      self._grid = self.cp.zeros((self.nx, self.ny, self.nz), dtype=self.cp.float32)
    else:
      self._grid = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)

  @property
  def device(self) -> str:
    return "GPU" if self.use_gpu else "CPU"

  def world_to_voxel(self, point: np.ndarray) -> np.ndarray:
    """Convert world coordinates to voxel indices."""
    return ((point - self._origin) / self.config.resolution).astype(int)

  def is_valid_voxel(self, voxel: np.ndarray) -> bool:
    """Check if voxel is in bounds."""
    return (
      0 <= voxel[0] < self.nx and
      0 <= voxel[1] < self.ny and
      0 <= voxel[2] < self.nz
    )

  def update_points_occupied(self, points: np.ndarray) -> None:
    """Batch update multiple points as occupied (GPU optimized)."""
    if len(points) == 0:
      return

    voxels = ((points - self._origin) / self.config.resolution).astype(int)

    # Filter valid voxels
    valid_mask = (
      (voxels[:, 0] >= 0) & (voxels[:, 0] < self.nx) &
      (voxels[:, 1] >= 0) & (voxels[:, 1] < self.ny) &
      (voxels[:, 2] >= 0) & (voxels[:, 2] < self.nz)
    )
    voxels = voxels[valid_mask]

    if len(voxels) == 0:
      return

    if self.use_gpu:
      voxels_gpu = self.cp.asarray(voxels)
      self._grid[voxels_gpu[:, 0], voxels_gpu[:, 1], voxels_gpu[:, 2]] = self.cp.clip(
        self._grid[voxels_gpu[:, 0], voxels_gpu[:, 1], voxels_gpu[:, 2]] + self.config.l_occ,
        self.config.l_min,
        self.config.l_max
      )
    else:
      self._grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = np.clip(
        self._grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]] + self.config.l_occ,
        self.config.l_min,
        self.config.l_max
      )

  def get_occupied_points(self) -> np.ndarray:
    """Get world coordinates of occupied voxels."""
    threshold_logodds = np.log(
      self.config.occupied_threshold / (1 - self.config.occupied_threshold)
    )

    if self.use_gpu:
      occupied = self.cp.argwhere(self._grid > threshold_logodds)
      occupied = self.cp.asnumpy(occupied)
    else:
      occupied = np.argwhere(self._grid > threshold_logodds)

    if len(occupied) == 0:
      return np.zeros((0, 3))

    return (occupied + 0.5) * self.config.resolution + self._origin

  def decay(self, factor: float = 0.95) -> None:
    """Apply decay to grid."""
    self._grid *= factor

  def clear(self) -> None:
    """Clear grid."""
    if self.use_gpu:
      self._grid.fill(0.0)
    else:
      self._grid.fill(0.0)

  def to_numpy(self) -> np.ndarray:
    """Get grid as numpy array."""
    if self.use_gpu:
      return self.cp.asnumpy(self._grid)
    return self._grid


@dataclass
class VoxelTrackerResult:
  """Result from voxel tracker at a timestep."""
  timestamp: float
  n_occupied: int
  n_allocated: int
  occupied_points: np.ndarray


class VoxelTracker:
  """Multi-object tracker using voxel grid occupancy."""

  def __init__(
    self,
    config: VoxelGridConfig | None = None,
    use_sparse: bool = True,
    use_gpu: bool = False
  ):
    self.config = config or VoxelGridConfig()

    if use_gpu:
      self.grid: VoxelGrid | SparseVoxelGrid | GPUVoxelGrid = GPUVoxelGrid(config)
    elif use_sparse:
      self.grid = SparseVoxelGrid(config)
    else:
      self.grid = VoxelGrid(config)

    self.use_sparse = use_sparse
    self.use_gpu = use_gpu
    self.results: list[VoxelTrackerResult] = []

  def process_detections(
    self,
    timestamp: float,
    points: np.ndarray,
    sensor_origin: np.ndarray | None = None
  ) -> VoxelTrackerResult:
    """Process point cloud detections.

    Args:
      timestamp: Current time
      points: Nx3 array of detection points in world coordinates
      sensor_origin: Sensor origin for ray casting (optional)

    Returns:
      Tracking result with occupied voxels
    """
    if sensor_origin is not None and not self.use_gpu:
      # Ray casting for explicit free space
      for point in points:
        if isinstance(self.grid, (VoxelGrid, SparseVoxelGrid)):
          self.grid.update_ray(sensor_origin, point)
    else:
      # Direct occupancy update
      if isinstance(self.grid, GPUVoxelGrid):
        self.grid.update_points_occupied(points)
      else:
        for point in points:
          self.grid.update_occupied(point)

    occupied_points = self.grid.get_occupied_points()

    n_allocated = 0
    if isinstance(self.grid, SparseVoxelGrid):
      n_allocated = self.grid.n_allocated

    result = VoxelTrackerResult(
      timestamp=timestamp,
      n_occupied=len(occupied_points),
      n_allocated=n_allocated,
      occupied_points=occupied_points
    )

    self.results.append(result)
    return result

  def decay(self, factor: float = 0.95) -> None:
    """Apply decay to grid."""
    self.grid.decay(factor)

  def clear(self) -> None:
    """Clear grid and results."""
    self.grid.clear()
    self.results.clear()


@dataclass
class VoxelBenchmarkResult:
  """Benchmark results for voxel grid."""
  grid_type: str
  n_points: int
  update_time_ms: float
  query_time_ms: float
  memory_mb: float
  sparsity: float


def benchmark_voxel_grids(
  n_points: int = 1000,
  n_iterations: int = 10
) -> list[VoxelBenchmarkResult]:
  """Benchmark different voxel grid implementations."""
  import time

  results = []

  # Generate random points in grid range
  config = VoxelGridConfig(
    x_range=(0.0, 50.0),
    y_range=(-10.0, 10.0),
    z_range=(-1.0, 2.0),
    resolution=0.25
  )

  np.random.seed(42)
  points = np.column_stack([
    np.random.uniform(0, 50, n_points),
    np.random.uniform(-10, 10, n_points),
    np.random.uniform(-1, 2, n_points)
  ])

  # Benchmark dense grid
  grid = VoxelGrid(config)
  start = time.monotonic()
  for _ in range(n_iterations):
    for point in points:
      grid.update_occupied(point)
  update_time = (time.monotonic() - start) / n_iterations * 1000

  start = time.monotonic()
  for _ in range(n_iterations):
    _ = grid.get_occupied_points()
  query_time = (time.monotonic() - start) / n_iterations * 1000

  memory_mb = grid._grid.nbytes / 1024 / 1024

  results.append(VoxelBenchmarkResult(
    grid_type="Dense",
    n_points=n_points,
    update_time_ms=update_time,
    query_time_ms=query_time,
    memory_mb=memory_mb,
    sparsity=0.0
  ))

  # Benchmark sparse grid
  grid = SparseVoxelGrid(config)
  start = time.monotonic()
  for _ in range(n_iterations):
    for point in points:
      grid.update_occupied(point)
  update_time = (time.monotonic() - start) / n_iterations * 1000

  start = time.monotonic()
  for _ in range(n_iterations):
    _ = grid.get_occupied_points()
  query_time = (time.monotonic() - start) / n_iterations * 1000

  # Estimate memory (dict overhead)
  memory_mb = len(grid._voxels) * 64 / 1024 / 1024  # Rough estimate

  results.append(VoxelBenchmarkResult(
    grid_type="Sparse",
    n_points=n_points,
    update_time_ms=update_time,
    query_time_ms=query_time,
    memory_mb=memory_mb,
    sparsity=grid.sparsity
  ))

  # Benchmark GPU grid if available
  gpu_grid = GPUVoxelGrid(config)
  if gpu_grid.use_gpu:
    start = time.monotonic()
    for _ in range(n_iterations):
      gpu_grid.update_points_occupied(points)
    update_time = (time.monotonic() - start) / n_iterations * 1000

    start = time.monotonic()
    for _ in range(n_iterations):
      _ = gpu_grid.get_occupied_points()
    query_time = (time.monotonic() - start) / n_iterations * 1000

    memory_mb = gpu_grid._grid.nbytes / 1024 / 1024

    results.append(VoxelBenchmarkResult(
      grid_type="GPU",
      n_points=n_points,
      update_time_ms=update_time,
      query_time_ms=query_time,
      memory_mb=memory_mb,
      sparsity=0.0
    ))

  return results


def format_benchmark_report(results: list[VoxelBenchmarkResult]) -> str:
  """Format benchmark results as markdown."""
  lines = [
    "# Voxel Grid Benchmark Results",
    "",
    "## Configuration",
    f"- Points per update: {results[0].n_points}",
    "",
    "## Results",
    "",
    "| Grid Type | Update (ms) | Query (ms) | Memory (MB) | Sparsity |",
    "|-----------|-------------|------------|-------------|----------|",
  ]

  for r in results:
    lines.append(
      f"| {r.grid_type} | {r.update_time_ms:.2f} | {r.query_time_ms:.2f} | "
      + f"{r.memory_mb:.2f} | {r.sparsity:.2%} |"
    )

  return "\n".join(lines)


if __name__ == "__main__":
  # Demo voxel tracking
  config = VoxelGridConfig(resolution=0.5)
  tracker = VoxelTracker(config, use_sparse=True)

  # Simulate detections
  np.random.seed(42)
  for t in range(10):
    points = np.random.randn(50, 3) * 5 + np.array([30.0, 0.0, 0.5])
    result = tracker.process_detections(t * 0.1, points)
    print(f"t={t*0.1:.1f}s: {result.n_occupied} occupied voxels")

  # Run benchmark
  print("\n" + format_benchmark_report(benchmark_voxel_grids(500, 5)))
