"""Data loader for the FinnWoodlands dataset.

FinnWoodlands (2023) contains:
- 5,170 stereo RGB frames from a ZED2 stereo camera
- Corresponding LiDAR point clouds from an Ouster OS1
- 300 annotated frames with panoptic labels:
    - Things (instance): Spruce, Birch, Pine tree trunks
    - Stuff (semantic): Lake, Ground, Track
- Backpack-mounted sensor setup
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# Semantic class definitions matching FinnWoodlands annotation scheme
SEMANTIC_CLASSES = {
    0: "unlabeled",
    1: "ground",
    2: "track",
    3: "lake",
    4: "spruce",
    5: "birch",
    6: "pine",
}

STUFF_CLASSES = {1, 2, 3}       # ground, track, lake
THING_CLASSES = {4, 5, 6}       # spruce, birch, pine


@dataclass
class CalibrationData:
    """Camera-LiDAR calibration parameters."""

    camera_matrix: np.ndarray           # 3x3 intrinsic matrix
    dist_coeffs: np.ndarray             # distortion coefficients
    extrinsic_lidar_to_cam: np.ndarray  # 4x4 LiDAR-to-camera transform
    baseline: float = 0.12              # ZED2 stereo baseline in meters

    @classmethod
    def from_file(cls, path: Path) -> "CalibrationData":
        """Load calibration from a NumPy .npz file."""
        data = np.load(path)
        return cls(
            camera_matrix=data["camera_matrix"],
            dist_coeffs=data["dist_coeffs"],
            extrinsic_lidar_to_cam=data["extrinsic_lidar_to_cam"],
            baseline=float(data.get("baseline", 0.12)),
        )


@dataclass
class AnnotationData:
    """Panoptic annotation for a single frame."""

    semantic_labels: np.ndarray     # (N,) int array — per-point semantic class
    instance_ids: np.ndarray        # (N,) int array — per-point instance id (0 = no instance)

    @classmethod
    def from_file(cls, path: Path) -> "AnnotationData":
        data = np.load(path)
        return cls(
            semantic_labels=data["semantic_labels"],
            instance_ids=data["instance_ids"],
        )


@dataclass
class Sample:
    """A single data sample from FinnWoodlands."""

    frame_id: int
    point_cloud: np.ndarray         # (N, 3) float32 XYZ points
    left_image: np.ndarray          # (H, W, 3) uint8 BGR image
    right_image: np.ndarray | None  # (H, W, 3) uint8 BGR image, may be absent
    calibration: CalibrationData
    annotation: AnnotationData | None = None  # only available for annotated frames
    intensity: np.ndarray | None = None       # (N,) float32 LiDAR intensity


class FinnWoodlandsLoader:
    """Load samples from a FinnWoodlands dataset directory.

    Expected directory layout::

        dataset_root/
            calibration/
                calib.npz
            point_clouds/
                000000.pcd  or .npy
                ...
            images/
                left/
                    000000.png
                    ...
                right/
                    000000.png
                    ...
            annotations/        (only for annotated subset)
                000000.npz
                ...
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._pc_dir = self.root / "point_clouds"
        self._img_left_dir = self.root / "images" / "left"
        self._img_right_dir = self.root / "images" / "right"
        self._ann_dir = self.root / "annotations"
        self._calib_path = self.root / "calibration" / "calib.npz"

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self._calibration: CalibrationData | None = None

    @property
    def calibration(self) -> CalibrationData:
        if self._calibration is None:
            if self._calib_path.exists():
                self._calibration = CalibrationData.from_file(self._calib_path)
            else:
                # Provide placeholder calibration for development
                self._calibration = CalibrationData(
                    camera_matrix=np.eye(3, dtype=np.float64),
                    dist_coeffs=np.zeros(5, dtype=np.float64),
                    extrinsic_lidar_to_cam=np.eye(4, dtype=np.float64),
                )
        return self._calibration

    def list_frame_ids(self) -> list[int]:
        """Return sorted list of available frame IDs based on point cloud files."""
        if not self._pc_dir.exists():
            return []
        ids = []
        for p in self._pc_dir.iterdir():
            if p.suffix in (".npy", ".pcd", ".ply"):
                try:
                    ids.append(int(p.stem))
                except ValueError:
                    continue
        return sorted(ids)

    def list_annotated_frame_ids(self) -> list[int]:
        """Return sorted list of frame IDs that have annotations."""
        if not self._ann_dir.exists():
            return []
        ids = []
        for p in self._ann_dir.glob("*.npz"):
            try:
                ids.append(int(p.stem))
            except ValueError:
                continue
        return sorted(ids)

    def _load_point_cloud(self, frame_id: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Load point cloud, return (points, intensity)."""
        npy_path = self._pc_dir / f"{frame_id:06d}.npy"
        if npy_path.exists():
            data = np.load(npy_path)
            if data.shape[1] >= 4:
                return data[:, :3].astype(np.float32), data[:, 3].astype(np.float32)
            return data[:, :3].astype(np.float32), None

        # Try Open3D for PCD/PLY files
        for ext in (".pcd", ".ply"):
            path = self._pc_dir / f"{frame_id:06d}{ext}"
            if path.exists():
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(str(path))
                points = np.asarray(pcd.points, dtype=np.float32)
                return points, None

        raise FileNotFoundError(f"No point cloud file found for frame {frame_id}")

    def _load_image(self, frame_id: int, side: str = "left") -> np.ndarray | None:
        img_dir = self._img_left_dir if side == "left" else self._img_right_dir
        for ext in (".png", ".jpg", ".jpeg"):
            path = img_dir / f"{frame_id:06d}{ext}"
            if path.exists():
                return cv2.imread(str(path))
        return None

    def _load_annotation(self, frame_id: int) -> AnnotationData | None:
        path = self._ann_dir / f"{frame_id:06d}.npz"
        if path.exists():
            return AnnotationData.from_file(path)
        return None

    def load_sample(self, frame_id: int) -> Sample:
        """Load a complete sample for the given frame ID."""
        points, intensity = self._load_point_cloud(frame_id)
        left = self._load_image(frame_id, "left")
        if left is None:
            raise FileNotFoundError(f"Left image not found for frame {frame_id}")
        right = self._load_image(frame_id, "right")
        annotation = self._load_annotation(frame_id)

        return Sample(
            frame_id=frame_id,
            point_cloud=points,
            left_image=left,
            right_image=right,
            calibration=self.calibration,
            annotation=annotation,
            intensity=intensity,
        )
