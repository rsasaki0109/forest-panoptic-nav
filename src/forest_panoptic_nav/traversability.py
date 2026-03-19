"""Traversability map generation from panoptic segmentation results.

Converts a semantically segmented point cloud into a 2D cost map suitable
for path planning.  Cost assignment:

    Class       | Cost  | Traversable
    ------------|-------|------------
    Ground (1)  | 0.1   | Yes
    Track  (2)  | 0.0   | Yes
    Lake   (3)  | 1.0   | No
    Spruce (4)  | 1.0   | No
    Birch  (5)  | 1.0   | No
    Pine   (6)  | 1.0   | No
    Unlabeled   | 0.5   | Maybe
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Default cost per semantic class
DEFAULT_COSTS: dict[int, float] = {
    0: 0.5,   # unlabeled — uncertain
    1: 0.1,   # ground
    2: 0.0,   # track — best surface
    3: 1.0,   # lake — impassable
    4: 1.0,   # spruce trunk — obstacle
    5: 1.0,   # birch trunk — obstacle
    6: 1.0,   # pine trunk — obstacle
    7: 1.0,   # obstacle — rocks, fallen logs, etc.
}

TRAVERSABLE_THRESHOLD = 0.6  # cost <= this is considered traversable


@dataclass
class CostMap:
    """2D occupancy / cost grid."""

    grid: np.ndarray        # (rows, cols) float32 cost values in [0, 1]
    cost: np.ndarray        # alias kept for serialization compat — same as grid
    origin: np.ndarray      # (2,) world XY of the grid corner (min_x, min_y)
    resolution: float       # meters per cell

    @property
    def traversable_ratio(self) -> float:
        valid = self.grid[self.grid >= 0]
        if len(valid) == 0:
            return 0.0
        return float((valid <= TRAVERSABLE_THRESHOLD).sum() / len(valid))

    def world_to_grid(self, xy: np.ndarray) -> np.ndarray:
        """Convert world XY coordinates to grid row/col indices."""
        return np.floor((xy - self.origin) / self.resolution).astype(np.int64)

    def grid_to_world(self, rc: np.ndarray) -> np.ndarray:
        """Convert grid row/col indices to world XY coordinates (cell center)."""
        return rc.astype(np.float64) * self.resolution + self.origin + self.resolution / 2

    def is_traversable(self, xy: np.ndarray) -> np.ndarray:
        """Check whether world XY positions are traversable."""
        idx = self.world_to_grid(xy)
        rows, cols = idx[:, 0], idx[:, 1]
        in_bounds = (rows >= 0) & (rows < self.grid.shape[0]) & (cols >= 0) & (cols < self.grid.shape[1])
        result = np.zeros(len(xy), dtype=bool)
        valid = in_bounds
        result[valid] = self.grid[rows[valid], cols[valid]] <= TRAVERSABLE_THRESHOLD
        return result


class TraversabilityMapper:
    """Generate traversability cost maps from segmented point clouds."""

    def __init__(
        self,
        resolution: float = 0.1,
        costs: dict[int, float] | None = None,
        unknown_cost: float = 0.5,
        kernel_size: int = 3,
    ) -> None:
        """
        Args:
            resolution: Grid cell size in meters.
            costs: Per-class cost overrides. Defaults to ``DEFAULT_COSTS``.
            unknown_cost: Cost for cells with no points.
            kernel_size: Smoothing kernel size (must be odd). 0 disables smoothing.
        """
        self.resolution = resolution
        self.costs = dict(DEFAULT_COSTS) if costs is None else dict(costs)
        self.unknown_cost = unknown_cost
        self.kernel_size = kernel_size

    def compute_cost_map(
        self,
        points: np.ndarray,
        semantic_labels: np.ndarray,
    ) -> CostMap:
        """Build a 2D cost map by projecting labelled points onto the XY plane.

        For each grid cell the cost is the *maximum* cost among all points that
        fall within it.  This is conservative: a single obstacle point makes the
        cell impassable.

        Args:
            points: (N, 3) float32 XYZ point cloud.
            semantic_labels: (N,) int32 semantic class per point.

        Returns:
            A ``CostMap`` instance.
        """
        xy = points[:, :2]
        min_xy = xy.min(axis=0)
        max_xy = xy.max(axis=0)

        # Grid dimensions
        grid_size = np.ceil((max_xy - min_xy) / self.resolution).astype(int) + 1
        rows, cols = int(grid_size[0]), int(grid_size[1])

        # Initialize with unknown cost
        grid = np.full((rows, cols), self.unknown_cost, dtype=np.float32)

        # Map points to grid cells
        cell_idx = np.floor((xy - min_xy) / self.resolution).astype(np.int64)
        cell_idx = np.clip(cell_idx, 0, [rows - 1, cols - 1])

        # Assign per-point cost from semantic label
        point_costs = np.array(
            [self.costs.get(int(lbl), self.unknown_cost) for lbl in semantic_labels],
            dtype=np.float32,
        )

        # For each cell, take the maximum cost (conservative)
        for i in range(len(points)):
            r, c = cell_idx[i]
            # First real observation resets the unknown cost
            if grid[r, c] == self.unknown_cost:
                grid[r, c] = point_costs[i]
            else:
                grid[r, c] = max(grid[r, c], point_costs[i])

        # Optional spatial smoothing to fill small gaps
        if self.kernel_size > 0:
            grid = self._smooth(grid)

        origin = min_xy.astype(np.float64)
        return CostMap(grid=grid, cost=grid, origin=origin, resolution=self.resolution)

    def _smooth(self, grid: np.ndarray) -> np.ndarray:
        """Apply conservative max-pool smoothing to propagate obstacles slightly."""
        import cv2

        k = self.kernel_size
        # Dilate obstacles: max filter spreads high-cost cells
        kernel = np.ones((k, k), dtype=np.uint8)
        dilated = cv2.dilate(grid, kernel, iterations=1)
        # Blend: keep the original where it's observed, use dilated for gaps
        unknown_mask = grid == self.unknown_cost
        result = grid.copy()
        result[unknown_mask] = dilated[unknown_mask]
        return result

    def merge_cost_maps(self, maps: list[CostMap]) -> CostMap:
        """Merge multiple cost maps into one, taking the minimum cost per cell.

        Useful for aggregating across frames to build a global map.
        """
        if not maps:
            raise ValueError("No cost maps to merge.")

        # Compute global bounds
        all_origins = np.array([m.origin for m in maps])
        all_ends = np.array([m.origin + np.array(m.grid.shape) * m.resolution for m in maps])
        global_min = all_origins.min(axis=0)
        global_max = all_ends.max(axis=0)

        grid_size = np.ceil((global_max - global_min) / self.resolution).astype(int)
        merged = np.full((grid_size[0], grid_size[1]), self.unknown_cost, dtype=np.float32)

        for m in maps:
            offset = np.round((m.origin - global_min) / self.resolution).astype(int)
            r0, c0 = offset
            r1 = r0 + m.grid.shape[0]
            c1 = c0 + m.grid.shape[1]
            sub = merged[r0:r1, c0:c1]
            merged[r0:r1, c0:c1] = np.minimum(sub, m.grid)

        return CostMap(
            grid=merged,
            cost=merged,
            origin=global_min,
            resolution=self.resolution,
        )
