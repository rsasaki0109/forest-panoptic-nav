"""Tests for the traversability module."""

import numpy as np
import pytest

from forest_panoptic_nav.traversability import (
    DEFAULT_COSTS,
    TRAVERSABLE_THRESHOLD,
    CostMap,
    TraversabilityMapper,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mapper():
    """TraversabilityMapper with smoothing disabled for deterministic tests."""
    return TraversabilityMapper(resolution=1.0, kernel_size=0)


@pytest.fixture
def simple_cloud():
    """4 points at known grid positions with known labels.

    Layout (1m resolution, origin at 0,0):
        (0,0) ground   -> cost 0.1
        (1,0) track    -> cost 0.0
        (0,1) lake     -> cost 1.0
        (1,1) pine     -> cost 1.0
    """
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float32)
    labels = np.array([1, 2, 3, 6], dtype=np.int32)  # ground, track, lake, pine
    return points, labels


# ---------------------------------------------------------------------------
# CostMap unit tests
# ---------------------------------------------------------------------------

class TestCostMap:
    def _make_costmap(self, grid, origin=(0.0, 0.0), resolution=1.0):
        g = np.array(grid, dtype=np.float32)
        o = np.array(origin, dtype=np.float64)
        return CostMap(grid=g, cost=g, origin=o, resolution=resolution)

    def test_world_to_grid(self):
        cm = self._make_costmap([[0.0, 0.5], [1.0, 0.1]], resolution=1.0)
        idx = cm.world_to_grid(np.array([[0.5, 0.5]]))
        assert idx[0, 0] == 0
        assert idx[0, 1] == 0

    def test_grid_to_world(self):
        cm = self._make_costmap([[0.0]], resolution=1.0, origin=(10.0, 20.0))
        xy = cm.grid_to_world(np.array([[0, 0]]))
        np.testing.assert_allclose(xy[0], [10.5, 20.5])

    def test_is_traversable_in_bounds(self):
        cm = self._make_costmap([[0.0, 1.0], [0.1, 0.5]])
        xy = np.array([
            [0.5, 0.5],   # grid (0,0) cost=0.0 -> traversable
            [1.5, 0.5],   # grid (1,0) cost=0.1 -> traversable
            [0.5, 1.5],   # grid (0,1) cost=1.0 -> not traversable
            [1.5, 1.5],   # grid (1,1) cost=0.5 -> traversable
        ])
        result = cm.is_traversable(xy)
        np.testing.assert_array_equal(result, [True, True, False, True])

    def test_is_traversable_out_of_bounds(self):
        cm = self._make_costmap([[0.0]])
        xy = np.array([[-10.0, -10.0]])
        result = cm.is_traversable(xy)
        assert not result[0]

    def test_traversable_ratio(self):
        # 3 out of 4 cells are traversable (cost <= 0.6)
        cm = self._make_costmap([[0.0, 1.0], [0.1, 0.5]])
        assert cm.traversable_ratio == pytest.approx(0.75)

    def test_traversable_ratio_empty(self):
        g = np.array([], dtype=np.float32).reshape(0, 0)
        cm = CostMap(grid=g, cost=g, origin=np.array([0.0, 0.0]), resolution=1.0)
        assert cm.traversable_ratio == 0.0


# ---------------------------------------------------------------------------
# TraversabilityMapper tests
# ---------------------------------------------------------------------------

class TestTraversabilityMapper:
    def test_compute_cost_map_shape(self, mapper, simple_cloud):
        points, labels = simple_cloud
        cost_map = mapper.compute_cost_map(points, labels)
        assert cost_map.grid.shape[0] >= 2
        assert cost_map.grid.shape[1] >= 2
        assert cost_map.resolution == 1.0

    def test_compute_cost_map_values(self, mapper, simple_cloud):
        points, labels = simple_cloud
        cost_map = mapper.compute_cost_map(points, labels)
        # The grid cell at (0,0) should have ground cost
        assert cost_map.grid[0, 0] == pytest.approx(DEFAULT_COSTS[1])  # ground
        # Track cell
        assert cost_map.grid[1, 0] == pytest.approx(DEFAULT_COSTS[2])  # track

    def test_compute_cost_map_conservative_max(self, mapper):
        """When two points land in the same cell, the higher cost wins."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0],  # same cell at resolution=1.0
        ], dtype=np.float32)
        labels = np.array([2, 3], dtype=np.int32)  # track=0.0, lake=1.0
        cost_map = mapper.compute_cost_map(points, labels)
        assert cost_map.grid[0, 0] == pytest.approx(1.0)

    def test_unknown_label_gets_unknown_cost(self, mapper):
        points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        labels = np.array([99], dtype=np.int32)  # not in DEFAULT_COSTS
        cost_map = mapper.compute_cost_map(points, labels)
        assert cost_map.grid[0, 0] == pytest.approx(0.5)

    def test_custom_costs(self):
        custom = {1: 0.9}
        m = TraversabilityMapper(resolution=1.0, costs=custom, kernel_size=0)
        points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        cost_map = m.compute_cost_map(points, labels)
        assert cost_map.grid[0, 0] == pytest.approx(0.9)

    def test_origin_matches_min_xy(self, mapper, simple_cloud):
        points, labels = simple_cloud
        cost_map = mapper.compute_cost_map(points, labels)
        np.testing.assert_allclose(cost_map.origin, [0.0, 0.0])

    def test_merge_cost_maps_single(self, mapper, simple_cloud):
        points, labels = simple_cloud
        cm = mapper.compute_cost_map(points, labels)
        merged = mapper.merge_cost_maps([cm])
        assert merged.grid.shape == cm.grid.shape

    def test_merge_cost_maps_takes_minimum(self, mapper):
        """Merging two maps should take the minimum cost per cell."""
        g1 = np.array([[1.0, 0.5]], dtype=np.float32)
        g2 = np.array([[0.2, 0.8]], dtype=np.float32)
        origin = np.array([0.0, 0.0], dtype=np.float64)
        cm1 = CostMap(grid=g1, cost=g1, origin=origin, resolution=1.0)
        cm2 = CostMap(grid=g2, cost=g2, origin=origin, resolution=1.0)
        merged = mapper.merge_cost_maps([cm1, cm2])
        np.testing.assert_allclose(merged.grid[0, 0], 0.2)
        np.testing.assert_allclose(merged.grid[0, 1], 0.5)

    def test_merge_cost_maps_empty_raises(self, mapper):
        with pytest.raises(ValueError, match="No cost maps"):
            mapper.merge_cost_maps([])

    def test_merge_cost_maps_different_origins(self, mapper):
        g1 = np.array([[0.1]], dtype=np.float32)
        g2 = np.array([[0.2]], dtype=np.float32)
        cm1 = CostMap(grid=g1, cost=g1, origin=np.array([0.0, 0.0], dtype=np.float64), resolution=1.0)
        cm2 = CostMap(grid=g2, cost=g2, origin=np.array([2.0, 2.0], dtype=np.float64), resolution=1.0)
        merged = mapper.merge_cost_maps([cm1, cm2])
        # Merged grid should be large enough to contain both
        assert merged.grid.shape[0] >= 3
        assert merged.grid.shape[1] >= 3

    def test_smoothing_enabled(self):
        """Smoke test that smoothing with cv2 does not crash."""
        pytest.importorskip("cv2")
        m = TraversabilityMapper(resolution=1.0, kernel_size=3)
        points = np.random.rand(100, 3).astype(np.float32) * 10
        labels = np.random.choice([0, 1, 2, 3, 4, 5, 6], size=100).astype(np.int32)
        cost_map = m.compute_cost_map(points, labels)
        assert cost_map.grid.shape[0] > 0
