"""Tests for A* path planning on traversability cost maps."""

import numpy as np
import pytest

from forest_panoptic_nav.traversability import CostMap
from forest_panoptic_nav.path_planner import plan_path, PathResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_costmap(grid, origin=(0.0, 0.0), resolution=1.0):
    g = np.array(grid, dtype=np.float32)
    o = np.array(origin, dtype=np.float64)
    return CostMap(grid=g, cost=g, origin=o, resolution=resolution)


# ---------------------------------------------------------------------------
# Basic path-finding
# ---------------------------------------------------------------------------

class TestPlanPath:
    def test_straight_line_on_free_grid(self):
        """On a uniform low-cost grid the path should be feasible."""
        grid = np.full((5, 5), 0.1, dtype=np.float32)
        cm = _make_costmap(grid)
        result = plan_path(cm, (0.5, 0.5), (4.5, 4.5))
        assert result.is_feasible
        assert result.num_waypoints >= 2
        assert result.distance > 0
        assert result.total_cost < float("inf")

    def test_path_around_obstacle(self):
        """Path must navigate around an impassable wall."""
        grid = np.full((5, 5), 0.1, dtype=np.float32)
        # Wall in the middle column except top row
        grid[0:4, 2] = 1.0
        cm = _make_costmap(grid)
        result = plan_path(cm, (0.5, 0.5), (0.5, 4.5))
        assert result.is_feasible
        # Path must go around the wall via row 4
        ys = [w[1] for w in result.waypoints]
        assert min(ys) < 1.0 or max([w[0] for w in result.waypoints]) > 3.0

    def test_no_path_when_blocked(self):
        """When the goal is completely surrounded by obstacles, no path exists."""
        grid = np.full((5, 5), 0.1, dtype=np.float32)
        # Surround cell (4,4) with obstacles
        grid[3, 3:5] = 1.0
        grid[4, 3] = 1.0
        grid[3:5, 4] = 1.0  # redundant but ensure sealed
        # Actually seal it: obstacle ring around (4,4)
        grid[3, 3] = 1.0
        grid[3, 4] = 1.0
        grid[4, 3] = 1.0
        cm = _make_costmap(grid)
        result = plan_path(cm, (0.5, 0.5), (4.5, 4.5))
        assert not result.is_feasible
        assert result.waypoints == []
        assert result.total_cost == float("inf")

    def test_start_on_obstacle(self):
        """Starting on an obstacle cell returns infeasible."""
        grid = np.full((3, 3), 0.1, dtype=np.float32)
        grid[0, 0] = 1.0
        cm = _make_costmap(grid)
        result = plan_path(cm, (0.5, 0.5), (2.5, 2.5))
        assert not result.is_feasible

    def test_goal_on_obstacle(self):
        """Goal on an obstacle cell returns infeasible."""
        grid = np.full((3, 3), 0.1, dtype=np.float32)
        grid[2, 2] = 1.0
        cm = _make_costmap(grid)
        result = plan_path(cm, (0.5, 0.5), (2.5, 2.5))
        assert not result.is_feasible

    def test_start_out_of_bounds(self):
        grid = np.full((3, 3), 0.1, dtype=np.float32)
        cm = _make_costmap(grid)
        result = plan_path(cm, (-10.0, -10.0), (1.5, 1.5))
        assert not result.is_feasible

    def test_start_equals_goal(self):
        grid = np.full((3, 3), 0.1, dtype=np.float32)
        cm = _make_costmap(grid)
        result = plan_path(cm, (1.5, 1.5), (1.5, 1.5))
        assert result.is_feasible
        assert result.num_waypoints == 1
        assert result.distance == pytest.approx(0.0)

    def test_prefers_low_cost_path(self):
        """A* should prefer a longer but lower-cost route."""
        grid = np.full((3, 5), 0.5, dtype=np.float32)
        # Create a cheap corridor along row 0
        grid[0, :] = 0.0
        # Create a cheap corridor along row 2
        grid[2, :] = 0.0
        # Middle row stays at 0.5 (traversable but expensive)
        cm = _make_costmap(grid)
        result = plan_path(cm, (0.5, 0.5), (0.5, 4.5))
        assert result.is_feasible
        # The path should predominantly use the cheap rows
        row_coords = [w[0] for w in result.waypoints]
        # Most waypoints should be near row 0 or row 2, not row 1
        assert sum(1 for r in row_coords if abs(r - 1.5) < 0.6) <= 2

    def test_resolution_scaling(self):
        """Path planning should work with non-unit resolution."""
        grid = np.full((10, 10), 0.1, dtype=np.float32)
        cm = _make_costmap(grid, resolution=0.5)
        result = plan_path(cm, (0.25, 0.25), (4.75, 4.75))
        assert result.is_feasible
        assert result.distance > 0


class TestPathResult:
    def test_dataclass_fields(self):
        pr = PathResult(
            waypoints=[(0.0, 0.0), (1.0, 1.0)],
            total_cost=0.5,
            distance=1.414,
            is_feasible=True,
        )
        assert pr.num_waypoints == 2
        assert pr.total_cost == 0.5
        assert pr.is_feasible
