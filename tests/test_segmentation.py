"""Tests for the segmentation module."""

import numpy as np
import pytest

from forest_panoptic_nav.segmentation import PanopticResult, PanopticSegmenter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def segmenter():
    """PanopticSegmenter using heuristic fallback (no model)."""
    return PanopticSegmenter(model_path=None, method="heuristic")


@pytest.fixture
def sample_points():
    """Point cloud with clear height separation for the heuristic segmenter."""
    rng = np.random.default_rng(42)
    n = 300
    xy = rng.uniform(-5, 5, size=(n, 2))
    # Low points -> ground, mid -> track, high -> trees
    z_low = rng.uniform(-2, -1, size=100)
    z_mid = rng.uniform(0, 1, size=100)
    z_high = rng.uniform(3, 5, size=100)
    z = np.concatenate([z_low, z_mid, z_high])
    return np.column_stack([xy, z]).astype(np.float32)


# ---------------------------------------------------------------------------
# PanopticResult tests
# ---------------------------------------------------------------------------

class TestPanopticResult:
    def _make_result(self, n=100):
        points = np.random.rand(n, 3).astype(np.float32)
        semantic = np.zeros(n, dtype=np.int32)
        semantic[:50] = 1  # ground
        semantic[50:] = 4  # spruce
        instance = np.zeros(n, dtype=np.int32)
        instance[50:75] = 1
        instance[75:] = 2
        confidence = np.full(n, 0.8, dtype=np.float32)
        return PanopticResult(
            points=points,
            semantic_labels=semantic,
            instance_ids=instance,
            confidence=confidence,
        )

    def test_num_instances(self):
        result = self._make_result()
        assert result.num_instances == 2

    def test_num_semantic_classes(self):
        result = self._make_result()
        assert result.num_semantic_classes == 2

    def test_get_instances_all(self):
        result = self._make_result()
        instances = result.get_instances()
        assert len(instances) == 2

    def test_get_instances_filtered(self):
        result = self._make_result()
        instances = result.get_instances(semantic_class=4)
        assert len(instances) == 2
        # Filter to class that has no instances
        instances = result.get_instances(semantic_class=1)
        assert len(instances) == 0

    def test_save_load_roundtrip(self, tmp_path):
        result = self._make_result()
        path = tmp_path / "result.npz"
        result.save(path)
        loaded = PanopticResult.load(path)
        np.testing.assert_array_equal(loaded.semantic_labels, result.semantic_labels)
        np.testing.assert_array_equal(loaded.instance_ids, result.instance_ids)
        np.testing.assert_array_equal(loaded.points, result.points)


# ---------------------------------------------------------------------------
# PanopticSegmenter tests
# ---------------------------------------------------------------------------

class TestPanopticSegmenter:
    def test_heuristic_predict_returns_result(self, segmenter, sample_points):
        result = segmenter.predict(sample_points)
        assert isinstance(result, PanopticResult)
        assert len(result.semantic_labels) == len(sample_points)
        assert len(result.instance_ids) == len(sample_points)
        assert len(result.confidence) == len(sample_points)

    def test_heuristic_assigns_ground(self, segmenter, sample_points):
        """Low points (below 20th percentile) should be classified as ground (1)."""
        result = segmenter.predict(sample_points)
        z = sample_points[:, 2]
        p20 = np.percentile(z, 20)
        ground_mask = z < p20
        assert (result.semantic_labels[ground_mask] == 1).all()

    def test_heuristic_assigns_track(self, segmenter, sample_points):
        """Mid-height points should be classified as track (2)."""
        result = segmenter.predict(sample_points)
        z = sample_points[:, 2]
        p20, p80 = np.percentile(z, [20, 80])
        mid_mask = (z >= p20) & (z < p80)
        assert (result.semantic_labels[mid_mask] == 2).all()

    def test_heuristic_assigns_trees(self, segmenter, sample_points):
        """High points (above 80th percentile) should be classified as pine (6)."""
        result = segmenter.predict(sample_points)
        z = sample_points[:, 2]
        p80 = np.percentile(z, 80)
        high_mask = z >= p80
        assert (result.semantic_labels[high_mask] == 6).all()

    def test_heuristic_creates_instances_for_trees(self, segmenter, sample_points):
        """High points should have non-zero instance IDs."""
        result = segmenter.predict(sample_points)
        z = sample_points[:, 2]
        p80 = np.percentile(z, 80)
        high_mask = z >= p80
        assert (result.instance_ids[high_mask] > 0).all()

    def test_model_load_raises(self, tmp_path):
        with pytest.raises(NotImplementedError):
            PanopticSegmenter(model_path=tmp_path / "fake_model.pt", method="ml")

    def test_confidence_in_range(self, segmenter, sample_points):
        result = segmenter.predict(sample_points)
        assert result.confidence.min() >= 0.0
        assert result.confidence.max() <= 1.0
