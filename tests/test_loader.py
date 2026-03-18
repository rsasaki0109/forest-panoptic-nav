"""Tests for the data loader module."""

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from forest_panoptic_nav.loader import (
    SEMANTIC_CLASSES,
    STUFF_CLASSES,
    THING_CLASSES,
    AnnotationData,
    CalibrationData,
    FinnWoodlandsLoader,
    Sample,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dataset_dir(tmp_path):
    """Create a minimal FinnWoodlands directory layout."""
    root = tmp_path / "dataset"
    (root / "calibration").mkdir(parents=True)
    (root / "point_clouds").mkdir(parents=True)
    (root / "images" / "left").mkdir(parents=True)
    (root / "images" / "right").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)

    # Create calibration file
    np.savez(
        root / "calibration" / "calib.npz",
        camera_matrix=np.eye(3, dtype=np.float64),
        dist_coeffs=np.zeros(5, dtype=np.float64),
        extrinsic_lidar_to_cam=np.eye(4, dtype=np.float64),
        baseline=0.12,
    )

    # Create one point cloud frame (frame 0)
    pc = np.random.rand(50, 3).astype(np.float32)
    np.save(root / "point_clouds" / "000000.npy", pc)

    # Create left image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "images" / "left" / "000000.png"), img)

    # Create annotation
    np.savez(
        root / "annotations" / "000000.npz",
        semantic_labels=np.ones(50, dtype=np.int32),
        instance_ids=np.zeros(50, dtype=np.int32),
    )

    return root


@pytest.fixture
def loader(dataset_dir):
    return FinnWoodlandsLoader(dataset_dir)


# ---------------------------------------------------------------------------
# Class definition tests
# ---------------------------------------------------------------------------

class TestSemanticClasses:
    def test_all_classes_defined(self):
        expected = {"unlabeled", "ground", "track", "lake", "spruce", "birch", "pine"}
        assert set(SEMANTIC_CLASSES.values()) == expected

    def test_stuff_thing_disjoint(self):
        assert STUFF_CLASSES & THING_CLASSES == set()

    def test_class_ids(self):
        assert SEMANTIC_CLASSES[0] == "unlabeled"
        assert SEMANTIC_CLASSES[1] == "ground"
        assert SEMANTIC_CLASSES[2] == "track"
        assert SEMANTIC_CLASSES[3] == "lake"
        assert SEMANTIC_CLASSES[4] == "spruce"
        assert SEMANTIC_CLASSES[5] == "birch"
        assert SEMANTIC_CLASSES[6] == "pine"


# ---------------------------------------------------------------------------
# CalibrationData tests
# ---------------------------------------------------------------------------

class TestCalibrationData:
    def test_from_file(self, dataset_dir):
        calib = CalibrationData.from_file(dataset_dir / "calibration" / "calib.npz")
        assert calib.camera_matrix.shape == (3, 3)
        assert calib.extrinsic_lidar_to_cam.shape == (4, 4)
        assert calib.baseline == pytest.approx(0.12)

    def test_default_baseline(self, tmp_path):
        np.savez(
            tmp_path / "calib.npz",
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            extrinsic_lidar_to_cam=np.eye(4),
        )
        calib = CalibrationData.from_file(tmp_path / "calib.npz")
        assert calib.baseline == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# AnnotationData tests
# ---------------------------------------------------------------------------

class TestAnnotationData:
    def test_from_file(self, dataset_dir):
        ann = AnnotationData.from_file(dataset_dir / "annotations" / "000000.npz")
        assert len(ann.semantic_labels) == 50
        assert len(ann.instance_ids) == 50


# ---------------------------------------------------------------------------
# FinnWoodlandsLoader tests
# ---------------------------------------------------------------------------

class TestFinnWoodlandsLoader:
    def test_init_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FinnWoodlandsLoader(tmp_path / "nonexistent")

    def test_list_frame_ids(self, loader):
        ids = loader.list_frame_ids()
        assert ids == [0]

    def test_list_annotated_frame_ids(self, loader):
        ids = loader.list_annotated_frame_ids()
        assert ids == [0]

    def test_list_frame_ids_empty(self, tmp_path):
        root = tmp_path / "empty_ds"
        root.mkdir()
        loader = FinnWoodlandsLoader(root)
        assert loader.list_frame_ids() == []

    def test_load_sample(self, loader):
        sample = loader.load_sample(0)
        assert isinstance(sample, Sample)
        assert sample.frame_id == 0
        assert sample.point_cloud.shape == (50, 3)
        assert sample.left_image is not None
        assert sample.annotation is not None

    def test_load_sample_missing_frame(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_sample(9999)

    def test_calibration_loaded(self, loader):
        calib = loader.calibration
        assert calib.camera_matrix.shape == (3, 3)

    def test_calibration_placeholder(self, tmp_path):
        """When no calib file exists, a placeholder identity calibration is used."""
        root = tmp_path / "no_calib_ds"
        root.mkdir()
        loader = FinnWoodlandsLoader(root)
        calib = loader.calibration
        np.testing.assert_array_equal(calib.camera_matrix, np.eye(3))

    def test_point_cloud_with_intensity(self, dataset_dir):
        """Point cloud with 4 columns should extract intensity."""
        pc = np.random.rand(30, 4).astype(np.float32)
        np.save(dataset_dir / "point_clouds" / "000001.npy", pc)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_dir / "images" / "left" / "000001.png"), img)
        loader = FinnWoodlandsLoader(dataset_dir)
        sample = loader.load_sample(1)
        assert sample.intensity is not None
        assert len(sample.intensity) == 30
