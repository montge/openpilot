"""Voxel grid occupancy tracker.

Maintains a 3D occupancy grid for environment representation.
Uses log-odds update for probabilistic occupancy estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VoxelGridConfig:
  """Configuration for voxel grid."""

  # Grid dimensions in world coordinates (meters)
  x_min: float = -50.0
  x_max: float = 150.0  # Forward range
  y_min: float = -25.0
  y_max: float = 25.0  # Lateral range
  z_min: float = -2.0
  z_max: float = 3.0  # Height range

  # Resolution (meters per voxel)
  resolution: float = 0.5

  # Log-odds parameters
  log_odds_free: float = -0.4  # Update for free space
  log_odds_occupied: float = 0.85  # Update for occupied
  log_odds_max: float = 3.5  # Clamp max
  log_odds_min: float = -2.0  # Clamp min
  log_odds_threshold: float = 0.0  # Threshold for occupied

  # Decay rate per update (for temporal smoothing)
  decay_rate: float = 0.02


class VoxelGrid:
  """3D occupancy grid for environment tracking.

  Uses log-odds representation for efficient probabilistic updates.
  Supports sparse representation for memory efficiency.

  Usage:
    config = VoxelGridConfig(resolution=0.5)
    grid = VoxelGrid(config)

    # Update with point cloud
    points = np.array([[10, 0, 0], [20, 1, 0], ...])  # [N, 3]
    grid.update_with_points(points, origin=np.array([0, 0, 0]))

    # Query occupancy
    is_occupied = grid.is_occupied(10.0, 0.0, 0.0)
    prob = grid.get_probability(10.0, 0.0, 0.0)

    # Get all occupied voxels
    occupied = grid.get_occupied_voxels()  # [M, 3] array
  """

  def __init__(self, config: VoxelGridConfig | None = None):
    """Initialize voxel grid.

    Args:
      config: Grid configuration (uses defaults if None)
    """
    self.config = config or VoxelGridConfig()
    c = self.config

    # Compute grid dimensions
    self.nx = int((c.x_max - c.x_min) / c.resolution)
    self.ny = int((c.y_max - c.y_min) / c.resolution)
    self.nz = int((c.z_max - c.z_min) / c.resolution)

    # Log-odds grid (initialized to 0 = unknown/50% probability)
    self.log_odds = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)

    # Update counter for temporal decay
    self._update_count = 0

  def world_to_grid(self, x: float, y: float, z: float) -> tuple[int, int, int]:
    """Convert world coordinates to grid indices.

    Args:
      x: World X coordinate (forward)
      y: World Y coordinate (lateral)
      z: World Z coordinate (height)

    Returns:
      Tuple of (ix, iy, iz) grid indices
    """
    c = self.config
    ix = int((x - c.x_min) / c.resolution)
    iy = int((y - c.y_min) / c.resolution)
    iz = int((z - c.z_min) / c.resolution)
    return ix, iy, iz

  def grid_to_world(self, ix: int, iy: int, iz: int) -> tuple[float, float, float]:
    """Convert grid indices to world coordinates (voxel center).

    Args:
      ix: Grid X index
      iy: Grid Y index
      iz: Grid Z index

    Returns:
      Tuple of (x, y, z) world coordinates
    """
    c = self.config
    x = c.x_min + (ix + 0.5) * c.resolution
    y = c.y_min + (iy + 0.5) * c.resolution
    z = c.z_min + (iz + 0.5) * c.resolution
    return x, y, z

  def is_valid_index(self, ix: int, iy: int, iz: int) -> bool:
    """Check if grid indices are valid.

    Args:
      ix, iy, iz: Grid indices

    Returns:
      True if indices are within grid bounds
    """
    return 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz

  def update_voxel(self, ix: int, iy: int, iz: int, log_odds_update: float) -> None:
    """Update a single voxel with log-odds.

    Args:
      ix, iy, iz: Grid indices
      log_odds_update: Log-odds value to add
    """
    if not self.is_valid_index(ix, iy, iz):
      return

    c = self.config
    self.log_odds[ix, iy, iz] += log_odds_update
    self.log_odds[ix, iy, iz] = np.clip(
      self.log_odds[ix, iy, iz],
      c.log_odds_min,
      c.log_odds_max,
    )

  def update_with_points(
    self,
    points: np.ndarray,
    origin: np.ndarray | None = None,
    ray_trace: bool = True,
  ) -> None:
    """Update grid with observed point cloud.

    Args:
      points: [N, 3] array of observed points in world frame
      origin: [3] sensor origin for ray tracing (default: [0,0,0])
      ray_trace: Whether to mark free space along rays
    """
    if origin is None:
      origin = np.zeros(3)

    c = self.config

    # Apply temporal decay first
    self.log_odds *= 1.0 - c.decay_rate

    # Mark observed points as occupied
    for point in points:
      ix, iy, iz = self.world_to_grid(point[0], point[1], point[2])
      self.update_voxel(ix, iy, iz, c.log_odds_occupied)

      # Ray trace to mark free space
      if ray_trace:
        self._ray_trace_free(origin, point)

    self._update_count += 1

  def _ray_trace_free(self, origin: np.ndarray, endpoint: np.ndarray) -> None:
    """Mark voxels along ray as free (Bresenham-like 3D).

    Args:
      origin: Ray start point
      endpoint: Ray end point (observed obstacle)
    """
    c = self.config

    # Convert to grid coordinates
    ix0, iy0, iz0 = self.world_to_grid(origin[0], origin[1], origin[2])
    ix1, iy1, iz1 = self.world_to_grid(endpoint[0], endpoint[1], endpoint[2])

    # Number of steps (simple linear interpolation)
    dx = abs(ix1 - ix0)
    dy = abs(iy1 - iy0)
    dz = abs(iz1 - iz0)
    n_steps = max(dx, dy, dz, 1)

    # Iterate along ray (skip first and last)
    for i in range(1, n_steps):
      t = i / n_steps
      ix = int(ix0 + t * (ix1 - ix0))
      iy = int(iy0 + t * (iy1 - iy0))
      iz = int(iz0 + t * (iz1 - iz0))

      if self.is_valid_index(ix, iy, iz):
        self.update_voxel(ix, iy, iz, c.log_odds_free)

  def is_occupied(self, x: float, y: float, z: float) -> bool:
    """Check if a world coordinate is occupied.

    Args:
      x, y, z: World coordinates

    Returns:
      True if voxel is above occupancy threshold
    """
    ix, iy, iz = self.world_to_grid(x, y, z)
    if not self.is_valid_index(ix, iy, iz):
      return False

    return self.log_odds[ix, iy, iz] > self.config.log_odds_threshold

  def get_probability(self, x: float, y: float, z: float) -> float:
    """Get occupancy probability at a world coordinate.

    Args:
      x, y, z: World coordinates

    Returns:
      Probability of occupancy (0-1)
    """
    ix, iy, iz = self.world_to_grid(x, y, z)
    if not self.is_valid_index(ix, iy, iz):
      return 0.5  # Unknown

    # Convert log-odds to probability
    log_odds = self.log_odds[ix, iy, iz]
    return 1.0 / (1.0 + np.exp(-log_odds))

  def get_occupied_voxels(self) -> np.ndarray:
    """Get world coordinates of all occupied voxels.

    Returns:
      [M, 3] array of occupied voxel centers in world frame
    """
    # Find occupied indices
    occupied_mask = self.log_odds > self.config.log_odds_threshold
    indices = np.argwhere(occupied_mask)

    if len(indices) == 0:
      return np.empty((0, 3))

    # Convert to world coordinates
    points = []
    for ix, iy, iz in indices:
      x, y, z = self.grid_to_world(int(ix), int(iy), int(iz))
      points.append([x, y, z])

    return np.array(points)

  def get_occupied_in_region(
    self,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float | None = None,
    z_max: float | None = None,
  ) -> np.ndarray:
    """Get occupied voxels within a region.

    Args:
      x_min, x_max: X bounds
      y_min, y_max: Y bounds
      z_min, z_max: Z bounds (optional, uses full range if None)

    Returns:
      [M, 3] array of occupied voxel centers
    """
    c = self.config
    if z_min is None:
      z_min = c.z_min
    if z_max is None:
      z_max = c.z_max

    # Convert to grid bounds
    ix_min, iy_min, iz_min = self.world_to_grid(x_min, y_min, z_min)
    ix_max, iy_max, iz_max = self.world_to_grid(x_max, y_max, z_max)

    # Clamp to valid range
    ix_min = max(0, ix_min)
    ix_max = min(self.nx, ix_max + 1)
    iy_min = max(0, iy_min)
    iy_max = min(self.ny, iy_max + 1)
    iz_min = max(0, iz_min)
    iz_max = min(self.nz, iz_max + 1)

    # Extract region
    region = self.log_odds[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max]
    occupied_mask = region > self.config.log_odds_threshold
    local_indices = np.argwhere(occupied_mask)

    if len(local_indices) == 0:
      return np.empty((0, 3))

    # Convert to world coordinates
    points = []
    for lix, liy, liz in local_indices:
      ix = ix_min + lix
      iy = iy_min + liy
      iz = iz_min + liz
      x, y, z = self.grid_to_world(int(ix), int(iy), int(iz))
      points.append([x, y, z])

    return np.array(points)

  def clear(self) -> None:
    """Reset grid to unknown state."""
    self.log_odds.fill(0.0)
    self._update_count = 0

  @property
  def shape(self) -> tuple[int, int, int]:
    """Grid dimensions (nx, ny, nz)."""
    return (self.nx, self.ny, self.nz)

  @property
  def update_count(self) -> int:
    """Number of updates performed."""
    return self._update_count

  def get_statistics(self) -> dict[str, Any]:
    """Get grid statistics.

    Returns:
      Dictionary with grid statistics
    """
    c = self.config
    occupied_count = np.sum(self.log_odds > c.log_odds_threshold)
    free_count = np.sum(self.log_odds < -c.log_odds_threshold)
    unknown_count = self.log_odds.size - occupied_count - free_count

    return {
      "shape": self.shape,
      "resolution": c.resolution,
      "total_voxels": self.log_odds.size,
      "occupied_voxels": int(occupied_count),
      "free_voxels": int(free_count),
      "unknown_voxels": int(unknown_count),
      "update_count": self._update_count,
      "memory_mb": self.log_odds.nbytes / 1e6,
    }
