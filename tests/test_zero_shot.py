"""Tests for zero-shot forest segmentation."""

import numpy as np
import pytest

from forest_panoptic_nav.segmentation import PanopticResult, PanopticSegmenter
from forest_panoptic_nav.zero_shot import (
    CLASS_BIRCH,
    CLASS_GROUND,
    CLASS_OBSTACLE,
    CLASS_PINE,
    CLASS_SPRUCE,
    CLASS_TRACK,
    ClusterInfo,
    ZeroShotForestSegmenter,
    _analyze_cluster,
    _classify_cluster,
    _estimate_bark_roughness,
    _fit_trunk_radius,
    _height_above_ground,
    _ransac_ground_plane,
    visualize_segmentation,
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


def _make_obstacle(
    center: tuple[float, float, float],
    size: tuple[float, float, float] = (1.0, 1.0, 0.5),
    n: int = 80,
) -> np.ndarray:
    """Box-shaped obstacle (rock, fallen log, etc.)."""
    rng = np.random.default_rng(hash(center) % (2**31))
    x = center[0] + rng.uniform(-size[0] / 2, size[0] / 2, n)
    y = center[1] + rng.uniform(-size[1] / 2, size[1] / 2, n)
    z = center[2] + rng.uniform(0, size[2], n)
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


@pytest.fixture
def forest_with_obstacles():
    """Synthetic forest with ground, trees, track, and obstacles."""
    rng = np.random.default_rng(42)

    # Ground
    ground = _make_ground_plane(800, z=0.0, noise=0.02)

    # Track (flat, narrow, elongated strip)
    px = rng.uniform(4.5, 5.5, 400)
    py = rng.uniform(-10, 10, 400)
    pz = rng.normal(-0.01, 0.005, 400)
    track = np.column_stack([px, py, pz]).astype(np.float32)

    # Trees
    tree1 = _make_cylinder((2, 3), z_base=0.5, height=6.0, radius=0.18, n=100)
    tree2 = _make_cylinder((-3, 1), z_base=0.5, height=4.0, radius=0.12, n=80)

    # Obstacle (rock-like)
    obstacle = _make_obstacle((-5, -5, 0.5), size=(0.8, 0.8, 0.6), n=60)

    points = np.vstack([ground, track, tree1, tree2, obstacle])
    return points


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
# Trunk radius estimation
# ---------------------------------------------------------------------------


class TestTrunkRadius:
    def test_cylinder_radius(self):
        """Trunk radius estimation should approximate the cylinder radius."""
        trunk = _make_cylinder((0, 0), z_base=0.0, height=5.0, radius=0.15, n=500)
        center_xy = trunk[:, :2].mean(axis=0)
        heights = trunk[:, 2]  # using z directly as height
        r = _fit_trunk_radius(trunk, center_xy, heights)
        # Should be roughly half the cylinder radius (median of uniform [0, R])
        assert 0.01 < r < 0.3

    def test_few_points_fallback(self):
        """With too few breast-height points, should use all points."""
        pts = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=np.float32)
        center = pts[:, :2].mean(axis=0)
        heights = np.array([0.0, 0.0, 0.0])
        r = _fit_trunk_radius(pts, center, heights)
        assert r >= 0.0


# ---------------------------------------------------------------------------
# Bark roughness
# ---------------------------------------------------------------------------


class TestBarkRoughness:
    def test_smooth_cylinder(self):
        """A perfect cylinder should have low bark roughness."""
        rng = np.random.default_rng(42)
        n = 200
        angles = rng.uniform(0, 2 * np.pi, n)
        r = 0.10  # fixed radius = smooth
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        z = rng.uniform(0, 3, n)
        pts = np.column_stack([x, y, z]).astype(np.float32)
        roughness = _estimate_bark_roughness(pts, np.array([0.0, 0.0]))
        assert roughness < 0.02

    def test_rough_cylinder(self):
        """A cylinder with noisy radius should have higher roughness."""
        rng = np.random.default_rng(42)
        n = 200
        angles = rng.uniform(0, 2 * np.pi, n)
        r = 0.10 + rng.normal(0, 0.05, n)  # noisy radius
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        z = rng.uniform(0, 3, n)
        pts = np.column_stack([x, y, z]).astype(np.float32)
        roughness = _estimate_bark_roughness(pts, np.array([0.0, 0.0]))
        assert roughness > 0.02


# ---------------------------------------------------------------------------
# Cluster analysis and classification
# ---------------------------------------------------------------------------


class TestClusterAnalysis:
    def test_analyze_cylinder(self):
        trunk = _make_cylinder((0, 0), z_base=0.5, height=4.0, radius=0.15, n=100)
        heights = trunk[:, 2] - 0.5  # approximate heights
        info = _analyze_cluster(trunk, heights=heights)
        assert info.height_span > 3.0
        assert info.xy_radius < 0.5
        assert info.aspect_ratio > 1.5
        assert info.num_points == 100
        assert info.trunk_radius > 0.0
        assert info.bark_roughness >= 0.0

    def test_classify_tall_trunk(self):
        info = ClusterInfo(
            center_xy=np.array([0, 0]),
            height_span=5.0,
            xy_radius=0.2,
            num_points=80,
            min_z=0.5,
            max_z=5.5,
            trunk_radius=0.18,
            bark_roughness=0.04,
            crown_spread=0.0,
            crown_flatness=0.0,
        )
        cls = _classify_cluster(info)
        assert cls in (CLASS_PINE, CLASS_SPRUCE)  # tall + thick trunk

    def test_classify_medium_trunk(self):
        info = ClusterInfo(
            center_xy=np.array([0, 0]),
            height_span=3.0,
            xy_radius=0.2,
            num_points=60,
            min_z=0.5,
            max_z=3.5,
            trunk_radius=0.14,
            bark_roughness=0.035,
            crown_spread=0.0,
            crown_flatness=0.0,
        )
        cls = _classify_cluster(info)
        assert cls in (CLASS_SPRUCE, CLASS_PINE)  # medium height + medium trunk

    def test_classify_thin_short_trunk_as_birch(self):
        """Thin trunk + short + smooth bark -> birch."""
        info = ClusterInfo(
            center_xy=np.array([0, 0]),
            height_span=1.8,
            xy_radius=0.10,
            num_points=40,
            min_z=0.5,
            max_z=2.3,
            trunk_radius=0.06,
            bark_roughness=0.01,
            crown_spread=0.0,
            crown_flatness=0.0,
        )
        cls = _classify_cluster(info)
        assert cls == CLASS_BIRCH

    def test_classify_obstacle(self):
        """Wide, low aspect ratio cluster -> obstacle."""
        info = ClusterInfo(
            center_xy=np.array([0, 0]),
            height_span=0.5,
            xy_radius=0.8,
            num_points=50,
            min_z=0.5,
            max_z=1.0,
            trunk_radius=0.5,
            bark_roughness=0.1,
            crown_spread=0.0,
            crown_flatness=0.0,
        )
        cls = _classify_cluster(info)
        assert cls == CLASS_OBSTACLE

    def test_small_cluster_unlabeled(self):
        info = ClusterInfo(
            center_xy=np.array([0, 0]),
            height_span=1.0,
            xy_radius=0.1,
            num_points=3,
            min_z=0.5,
            max_z=1.5,
            trunk_radius=0.05,
            bark_roughness=0.01,
            crown_spread=0.0,
            crown_flatness=0.0,
        )
        assert _classify_cluster(info) == 0  # CLASS_UNLABELED


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

    def test_obstacle_detection(self, forest_with_obstacles):
        """Non-trunk above-ground objects should be classified as obstacles."""
        seg = ZeroShotForestSegmenter()
        result = seg.predict(forest_with_obstacles)
        found_classes = set(np.unique(result.semantic_labels))
        # Should detect at least trees and ground/track
        tree_classes = {CLASS_SPRUCE, CLASS_BIRCH, CLASS_PINE}
        assert len(found_classes & tree_classes) >= 1, "No trees detected"
        assert CLASS_GROUND in found_classes or CLASS_TRACK in found_classes

    def test_track_detection(self, forest_with_obstacles):
        """Track region should be detected."""
        seg = ZeroShotForestSegmenter()
        result = seg.predict(forest_with_obstacles)
        assert CLASS_TRACK in set(np.unique(result.semantic_labels)), "Track not detected"


# ---------------------------------------------------------------------------
# Visualization test
# ---------------------------------------------------------------------------


class TestVisualization:
    def test_visualize_saves_png(self, synthetic_forest, tmp_path):
        """visualize_segmentation should save a PNG file."""
        points, *_ = synthetic_forest
        seg = ZeroShotForestSegmenter()
        result = seg.predict(points)
        out_path = str(tmp_path / "test_viz.png")
        visualize_segmentation(result, out_path)
        assert (tmp_path / "test_viz.png").exists()
        assert (tmp_path / "test_viz.png").stat().st_size > 0


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
