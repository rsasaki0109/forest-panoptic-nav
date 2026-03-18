"""Tests for the CLI module."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from forest_panoptic_nav.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def dataset_dir(tmp_path):
    """Minimal FinnWoodlands dataset for CLI smoke tests."""
    root = tmp_path / "dataset"
    (root / "calibration").mkdir(parents=True)
    (root / "point_clouds").mkdir(parents=True)
    (root / "images" / "left").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)

    np.savez(
        root / "calibration" / "calib.npz",
        camera_matrix=np.eye(3, dtype=np.float64),
        dist_coeffs=np.zeros(5, dtype=np.float64),
        extrinsic_lidar_to_cam=np.eye(4, dtype=np.float64),
    )

    pc = np.random.rand(50, 3).astype(np.float32)
    np.save(root / "point_clouds" / "000000.npy", pc)

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "images" / "left" / "000000.png"), img)

    np.savez(
        root / "annotations" / "000000.npz",
        semantic_labels=np.ones(50, dtype=np.int32),
        instance_ids=np.zeros(50, dtype=np.int32),
    )

    return root


class TestCli:
    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "segment" in result.output
        assert "traversability" in result.output
        assert "visualize" in result.output

    def test_segment_help(self, runner):
        result = runner.invoke(cli, ["segment", "--help"])
        assert result.exit_code == 0
        assert "DATA_DIR" in result.output

    def test_segment_no_fusion(self, runner, dataset_dir, tmp_path):
        """Run segmentation without fusion (no camera projection needed)."""
        output_dir = tmp_path / "seg_output"
        result = runner.invoke(cli, [
            "segment", str(dataset_dir),
            "--no-fusion",
            "-o", str(output_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Processing 1 frame" in result.output
        seg_files = list(output_dir.glob("frame_*.npz"))
        assert len(seg_files) == 1

    def test_segment_single_frame(self, runner, dataset_dir, tmp_path):
        output_dir = tmp_path / "seg_single"
        result = runner.invoke(cli, [
            "segment", str(dataset_dir),
            "--no-fusion",
            "-f", "0",
            "-o", str(output_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "1 frame" in result.output

    def test_traversability(self, runner, dataset_dir, tmp_path):
        """Run traversability on pre-generated segmentation output."""
        seg_dir = tmp_path / "seg"
        seg_dir.mkdir()
        # Create a fake segmentation result
        points = np.random.rand(50, 3).astype(np.float32)
        labels = np.ones(50, dtype=np.int32)
        np.savez(seg_dir / "frame_000000.npz",
                 points=points, semantic_labels=labels,
                 instance_ids=np.zeros(50, dtype=np.int32),
                 confidence=np.ones(50, dtype=np.float32))
        trav_dir = tmp_path / "trav_output"
        result = runner.invoke(cli, [
            "traversability", str(seg_dir),
            "-o", str(trav_dir),
            "-r", "0.5",
        ])
        assert result.exit_code == 0, result.output
        assert "traversable" in result.output.lower()

    def test_segment_nonexistent_dir(self, runner, tmp_path):
        result = runner.invoke(cli, [
            "segment", str(tmp_path / "nonexistent"),
        ])
        assert result.exit_code != 0

    def test_traversability_help(self, runner):
        result = runner.invoke(cli, ["traversability", "--help"])
        assert result.exit_code == 0
        assert "resolution" in result.output.lower()
