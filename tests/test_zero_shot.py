"""Tests for zero-shot forest segmentation."""

import numpy as np
import pytest

from forest_panoptic_nav.segmentation import PanopticResult, PanopticSegmenter
from forest_panoptic_nav.zero_shot import (
    CLASS_BIRCH,
    CLASS_GROUND,
    CLASS_PINE,
    CLASS_SPRUCE,
    CLASS_TRACK,
    ClusterInfo,
    ZeroShotForestSegmenter,
    _analyze_cluster,
    _classify_cluster,
    _height_above_ground,
    _ransac_ground_plane,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic forest point clouds
# ---------------------------------------------------------------------------


def _make_ground_plane(n: int = 500, z: float = 0.0, noise: float = 0.02) -> np.ndarray:
    """Flat ground plane at height z with slight noise."""
    rng = np.random.default_rng(42)
    xy = rng.uniform(-10, 10, size=(n, 2))
    zz = np.full(n, z) + rng.normal(0, noise, n)
    return np.column_stack([xy, zz]).astype(np.float32)


def _make_cylinder(
    center_xy: tuple[float, float],
    z_base: float,
    height: float,
    radius: float = 0.15,
    n: int = 100,
) -> np.ndarray:
    """Vertical cylinder simulating a tree trunk."""
    rng = np.random.default_rng(hash(center_xy) % (2**31))
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = rng.uniform(0, radius, n)
    x = center_xy[0] + radii * np.cos(angles)
    y = center_xy[1] + radii * np.sin(angles)
    z = rng.uniform(z_base, z_base + height, n)
    return np.column_stack([x, y, z]).astype(np.float32)


@pytest.fixture
def synthetic_forest():
    """Synthetic forest: flat ground + 3 tree trunk cylinders."""
    ground = _make_ground_plane(500, z=0.0, noise=0.02)
    tree1 = _make_cylinder((2, 3), z_base=0.5, height=5.0, n=80)   # tall -> pine
    tree2 = _make_cylinder((-3, 1), z_base=0.5, height=3.0, n=80)  # medium -> spruce
    tree3 = _make_cylinder((0, -4), z_base=0.5, height=1.5, n=60)  # short -> birch
    points = np.vstack([ground, tree1, tree2, tree3])
    return points, len(ground), len(tree1), len(tree2), len(tree3)


# ---------------------------------------------------------------------------
# RANSAC ground plane tests
# ---------------------------------------------------------------------------


class TestRansacGroundPlane:
    def test_flat_ground(self):
        ground = _make_ground_plane(300, z=0.0, noise=0.01)
        normal, d = _ransac_ground_plane(ground)
        # Normal should be approximately [0, 0, 1]
        assert abs(normal[2]) > 0.9
        # d should be close to 0
        assert abs(d) < 0.5

    def test_with_trees_above(self):
        ground = _make_ground_plane(500, z=0.0, noise=0.02)
        trees = _make_cylinder((0, 0), z_base=1.0, height=4.0, n=100)
        points = np.vstack([ground, trees])
        normal, d = _ransac_ground_plane(points)
        assert abs(normal[2]) > 0.8

    def test_few_points_fallback(self):
        """With < 3 points, should return a default plane."""
        points = np.array([[0, 0, 1], [1, 0, 1]], dtype=np.float32)
        normal, d = _ransac_ground_plane(points)
        assert normal.shape == (3,)


# ---------------------------------------------------------------------------
# Height above ground
# ---------------------------------------------------------------------------


class TestHeightAboveGround:
    def test_flat_plane(self):
        points = np.array([[0, 0, 0], [0, 0, 5], [0, 0, -1]], dtype=np.float32)
        normal = np.array([0, 0, 1.0])
        d = 0.0
        heights = _height_above_ground(points, normal, d)
        np.testing.assert_allclose(heights, [0, 5, -1], atol=1e-6)


# ---------------------------------------------------------------------------
# Cluster analysis and classification
# ---------------------------------------------------------------------------


class TestClusterAnalysis:
    def test_analyze_cylinder(self):
        trunk = _make_cylinder((0, 0), z_base=0.5, height=4.0, radius=0.15, n=100)
        info = _analyze_cluster(trunk)
        assert info.height_span > 3.0
        assert info.xy_radius < 0.5
        assert info.aspect_ratio > 1.5
        assert info.num_points == 100

    def test_classify_tall_trunk(self):
        info = ClusterInfo(
            center_xy=np.array([0, 0]),
            height_span=5.0,
            xy_radius=0.2,
            num_points=80,
            min_z=0.5,
            max_z=5.5,
        )
        assert _classify_cluster(info) == CLASS_PINE

    def test_classify_medium_trunk(self):
        info = ClusterInfo(
            center_xy=np.array([0, 0]),
            height_span=3.0,
            xy_radius=0.2,
            num_points=60,
            min_z=0.5,
            max_z=3.5,
        )
        assert _classify_cluster(info) == CLASS_SPRUCE

    def test_classify_short_trunk(self):
        info = ClusterInfo(
            center_xy=np.array([0, 0]),
            height_span=1.5,
            xy_radius=0.15,
            num_points=40,
            min_z=0.5,
            max_z=2.0,
        )
        assert _classify_cluster(info) == CLASS_BIRCH


# ---------------------------------------------------------------------------
# ZeroShotForestSegmenter integration tests
# ---------------------------------------------------------------------------


class TestZeroShotForestSegmenter:
    def test_returns_panoptic_result(self, synthetic_forest):
        points, *_ = synthetic_forest
        seg = ZeroShotForestSegmenter()
        result = seg.predict(points)
        assert isinstance(result, PanopticResult)
        assert len(result.semantic_labels) == len(points)
        assert len(result.instance_ids) == len(points)
        assert len(result.confidence) == len(points)

    def test_ground_detected(self, synthetic_forest):
        points, n_ground, *_ = synthetic_forest
        seg = ZeroShotForestSegmenter()
        result = seg.predict(points)
        # Most ground points should be classified as ground or track
        ground_labels = result.semantic_labels[:n_ground]
        ground_or_track = np.isin(ground_labels, [CLASS_GROUND, CLASS_TRACK])
        ratio = ground_or_track.sum() / n_ground
        assert ratio > 0.8, f"Only {ratio:.1%} of ground points classified as ground/track"

    def test_trees_have_instances(self, synthetic_forest):
        points, n_ground, *_ = synthetic_forest
        seg = ZeroShotForestSegmenter()
        result = seg.predict(points)
        # Tree points (after ground) should have some non-zero instance IDs
        tree_instances = result.instance_ids[n_ground:]
        assert (tree_instances > 0).any(), "No tree instances detected"

    def test_tree_species_assigned(self, synthetic_forest):
        points, *_ = synthetic_forest
        seg = ZeroShotForestSegmenter()
        result = seg.predict(points)
        tree_classes = {CLASS_SPRUCE, CLASS_BIRCH, CLASS_PINE}
        found = set(np.unique(result.semantic_labels)) & tree_classes
        assert len(found) >= 1, "No tree species classes detected"

    def test_confidence_in_range(self, synthetic_forest):
        points, *_ = synthetic_forest
        seg = ZeroShotForestSegmenter()
        result = seg.predict(points)
        assert result.confidence.min() >= 0.0
        assert result.confidence.max() <= 1.0

    def test_empty_cloud(self):
        points = np.empty((0, 3), dtype=np.float32)
        seg = ZeroShotForestSegmenter()
        result = seg.predict(points)
        assert len(result.semantic_labels) == 0

    def test_tiny_cloud(self):
        points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        seg = ZeroShotForestSegmenter()
        result = seg.predict(points)
        assert len(result.semantic_labels) == 2

    def test_multiple_tree_instances(self, synthetic_forest):
        """The 3 separate trunk cylinders should produce distinct instance IDs."""
        points, *_ = synthetic_forest
        seg = ZeroShotForestSegmenter()
        result = seg.predict(points)
        unique_instances = np.unique(result.instance_ids)
        n_instances = (unique_instances > 0).sum()
        assert n_instances >= 2, f"Expected >= 2 tree instances, got {n_instances}"


# ---------------------------------------------------------------------------
# Integration with PanopticSegmenter (method="zero_shot")
# ---------------------------------------------------------------------------


class TestPanopticSegmenterZeroShot:
    def test_default_method_is_zero_shot(self):
        seg = PanopticSegmenter()
        assert seg.method == "zero_shot"

    def test_zero_shot_predict(self, synthetic_forest):
        points, *_ = synthetic_forest
        seg = PanopticSegmenter(method="zero_shot")
        result = seg.predict(points)
        assert isinstance(result, PanopticResult)
        assert result.num_instances > 0

    def test_heuristic_still_works(self, synthetic_forest):
        points, *_ = synthetic_forest
        seg = PanopticSegmenter(method="heuristic")
        result = seg.predict(points)
        assert isinstance(result, PanopticResult)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            PanopticSegmenter(method="invalid")
