"""A* path planning on traversability cost maps.

Given a 2D cost grid (from ``traversability.py``), plans the shortest
low-cost path between two world-coordinate positions.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .traversability import CostMap, TRAVERSABLE_THRESHOLD


@dataclass
class PathResult:
    """Result of a path planning query."""

    waypoints: list[tuple[float, float]]  # (x, y) world coordinates
    total_cost: float
    distance: float  # Euclidean path length in world units
    is_feasible: bool

    @property
    def num_waypoints(self) -> int:
        return len(self.waypoints)


# 8-connected neighbourhood: (drow, dcol, move_cost_multiplier)
_NEIGHBORS: list[tuple[int, int, float]] = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (1, 1, math.sqrt(2)),
]


def plan_path(
    cost_map: CostMap,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> PathResult:
    """Plan a path on *cost_map* from *start_xy* to *goal_xy* using A*.

    Both start and goal are given in **world coordinates**.  The planner
    operates on the cost grid and returns waypoints converted back to world
    coordinates.

    Cells with cost > ``TRAVERSABLE_THRESHOLD`` are treated as obstacles.

    Returns:
        A ``PathResult``.  If no feasible path exists, ``is_feasible`` is
        ``False`` and ``waypoints`` is empty.
    """
    grid = cost_map.grid
    rows, cols = grid.shape

    start_rc = _world_to_rc(cost_map, start_xy)
    goal_rc = _world_to_rc(cost_map, goal_xy)

    # Validate bounds
    for label, (r, c) in [("start", start_rc), ("goal", goal_rc)]:
        if not (0 <= r < rows and 0 <= c < cols):
            return PathResult(waypoints=[], total_cost=float("inf"), distance=0.0, is_feasible=False)
        if grid[r, c] > TRAVERSABLE_THRESHOLD:
            return PathResult(waypoints=[], total_cost=float("inf"), distance=0.0, is_feasible=False)

    # A* search
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {}
    g_score: dict[tuple[int, int], float] = {start_rc: 0.0}

    # Priority queue entries: (f_score, g_score, row, col)
    open_set: list[tuple[float, float, int, int]] = []
    h0 = _heuristic(start_rc, goal_rc)
    heapq.heappush(open_set, (h0, 0.0, start_rc[0], start_rc[1]))
    came_from[start_rc] = None

    while open_set:
        _f, g_curr, r, c = heapq.heappop(open_set)

        current = (r, c)
        if current == goal_rc:
            break

        # Skip stale entries
        if g_curr > g_score.get(current, float("inf")):
            continue

        for dr, dc, move_mult in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            neighbor = (nr, nc)
            cell_cost = grid[nr, nc]
            if cell_cost > TRAVERSABLE_THRESHOLD:
                continue

            tentative_g = g_curr + cell_cost * move_mult * cost_map.resolution
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + _heuristic(neighbor, goal_rc) * cost_map.resolution
                heapq.heappush(open_set, (f, tentative_g, nr, nc))

    if goal_rc not in came_from:
        return PathResult(waypoints=[], total_cost=float("inf"), distance=0.0, is_feasible=False)

    # Reconstruct path
    rc_path: list[tuple[int, int]] = []
    node: tuple[int, int] | None = goal_rc
    while node is not None:
        rc_path.append(node)
        node = came_from[node]
    rc_path.reverse()

    # Convert to world coordinates
    rc_arr = np.array(rc_path, dtype=np.float64)
    world_pts = cost_map.grid_to_world(rc_arr)
    waypoints = [(float(x), float(y)) for x, y in world_pts]

    # Compute Euclidean distance along path
    diffs = np.diff(world_pts, axis=0)
    distance = float(np.sum(np.linalg.norm(diffs, axis=1))) if len(diffs) > 0 else 0.0

    return PathResult(
        waypoints=waypoints,
        total_cost=g_score[goal_rc],
        distance=distance,
        is_feasible=True,
    )


def plot_path(
    cost_map: CostMap,
    path: PathResult,
    output_path: str,
    *,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """Visualize a cost map with a planned path overlay.

    Args:
        cost_map: The traversability cost map.
        path: A ``PathResult`` from ``plan_path``.
        output_path: File path for the saved figure.
        figsize: Matplotlib figure size.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    extent = [
        cost_map.origin[0],
        cost_map.origin[0] + cost_map.grid.shape[1] * cost_map.resolution,
        cost_map.origin[1],
        cost_map.origin[1] + cost_map.grid.shape[0] * cost_map.resolution,
    ]

    im = ax.imshow(
        cost_map.grid,
        cmap="RdYlGn_r",
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(im, ax=ax, label="Traversability Cost")

    if path.is_feasible and path.waypoints:
        xs = [w[0] for w in path.waypoints]
        ys = [w[1] for w in path.waypoints]
        ax.plot(xs, ys, "b-", linewidth=2, label="Planned path")
        ax.plot(xs[0], ys[0], "go", markersize=10, label="Start")
        ax.plot(xs[-1], ys[-1], "r*", markersize=14, label="Goal")
        ax.legend(loc="upper right")
        ax.set_title(
            f"Path: {path.distance:.1f}m, cost {path.total_cost:.2f}"
        )
    else:
        ax.set_title("No feasible path found")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _world_to_rc(cost_map: CostMap, xy: tuple[float, float]) -> tuple[int, int]:
    arr = np.array([list(xy)], dtype=np.float64)
    idx = cost_map.world_to_grid(arr)
    return (int(idx[0, 0]), int(idx[0, 1]))


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Octile distance heuristic for 8-connected grid."""
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)
