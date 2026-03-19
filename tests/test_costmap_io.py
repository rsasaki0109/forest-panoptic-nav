"""Tests for costmap I/O utilities."""

import numpy as np
import pytest

from forest_panoptic_nav.traversability import CostMap
from forest_panoptic_nav.costmap_io import save_costmap, load_costmap


def _make_costmap(grid, origin=(0.0, 0.0), resolution=1.0):
    g = np.array(grid, dtype=np.float32)
    o = np.array(origin, dtype=np.float64)
    return CostMap(grid=g, cost=g, origin=o, resolution=resolution)


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        cm = _make_costmap([[0.1, 0.5], [1.0, 0.0]], origin=(10.0, 20.0), resolution=0.25)
        path = tmp_path / "costmap.npz"
        save_costmap(cm, path)
        loaded = load_costmap(path)
        np.testing.assert_array_almost_equal(loaded.grid, cm.grid)
        np.testing.assert_array_almost_equal(loaded.origin, cm.origin)
        assert loaded.resolution == pytest.approx(cm.resolution)

    def test_creates_parent_dirs(self, tmp_path):
        cm = _make_costmap([[0.0]])
        path = tmp_path / "sub" / "dir" / "costmap.npz"
        save_costmap(cm, path)
        assert path.exists()

    def test_load_preserves_dtype(self, tmp_path):
        cm = _make_costmap([[0.1, 0.2]])
        path = tmp_path / "test.npz"
        save_costmap(cm, path)
        loaded = load_costmap(path)
        assert loaded.grid.dtype == np.float32
        assert loaded.origin.dtype == np.float64
