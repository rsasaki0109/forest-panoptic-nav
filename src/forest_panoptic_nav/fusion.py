"""LiDAR-RGB sensor fusion for the FinnWoodlands dataset.

Projects Ouster OS1 LiDAR points onto the ZED2 stereo camera image plane
to combine 3D geometry with RGB appearance features.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .loader import CalibrationData


@dataclass
class FusedData:
    """Point cloud enriched with camera-derived features."""

    point_cloud: np.ndarray     # (M, 3) points that have valid image projections
    features: np.ndarray        # (M, D) per-point feature vectors
    pixel_coords: np.ndarray    # (M, 2) corresponding pixel locations (u, v)
    valid_mask: np.ndarray      # (N,) bool mask over original points — True if point was fused


class LidarRgbFusion:
    """Project LiDAR points into camera images and extract per-point features."""

    def __init__(self, feature_mode: str = "rgb") -> None:
        """
        Args:
            feature_mode: Type of features to extract per point.
                - "rgb": raw RGB color (3-dim)
                - "rgbi": RGB + LiDAR intensity (4-dim)
                - "learned": placeholder for a learned feature extractor
        """
        if feature_mode not in ("rgb", "rgbi", "learned"):
            raise ValueError(f"Unknown feature_mode: {feature_mode}")
        self.feature_mode = feature_mode

    def fuse(
        self,
        points: np.ndarray,
        image: np.ndarray,
        calibration: CalibrationData,
        intensity: np.ndarray | None = None,
    ) -> FusedData:
        """Project LiDAR points onto the image and sample features.

        Args:
            points: (N, 3) float32 XYZ point cloud in LiDAR frame.
            image: (H, W, 3) uint8 BGR image.
            calibration: Camera-LiDAR calibration parameters.
            intensity: (N,) optional LiDAR intensity.

        Returns:
            FusedData containing only points visible in the image.
        """
        pixel_coords, valid_mask = self.project_points(points, calibration, image.shape[:2])

        fused_points = points[valid_mask]
        fused_pixels = pixel_coords[valid_mask]

        features = self._extract_features(image, fused_pixels, intensity, valid_mask)

        return FusedData(
            point_cloud=fused_points,
            features=features,
            pixel_coords=fused_pixels,
            valid_mask=valid_mask,
        )

    def project_points(
        self,
        points: np.ndarray,
        calibration: CalibrationData,
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project 3D LiDAR points to 2D image pixel coordinates.

        Args:
            points: (N, 3) points in LiDAR frame.
            calibration: Calibration data with extrinsic and intrinsic params.
            image_shape: (height, width) of the target image.

        Returns:
            pixel_coords: (N, 2) float array of (u, v) pixel coordinates.
            valid_mask: (N,) bool array — True for points in front of camera
                        and within image bounds.
        """
        n = len(points)
        # Homogeneous coordinates
        pts_h = np.hstack([points, np.ones((n, 1), dtype=np.float32)])

        # Transform to camera frame
        pts_cam = (calibration.extrinsic_lidar_to_cam @ pts_h.T).T  # (N, 4)
        pts_cam_3d = pts_cam[:, :3]

        # Points must be in front of the camera (positive Z in camera frame)
        in_front = pts_cam_3d[:, 2] > 0

        # Project to image plane
        fx = calibration.camera_matrix[0, 0]
        fy = calibration.camera_matrix[1, 1]
        cx = calibration.camera_matrix[0, 2]
        cy = calibration.camera_matrix[1, 2]

        z = pts_cam_3d[:, 2]
        z_safe = np.where(z > 0, z, 1.0)  # avoid division by zero

        u = fx * pts_cam_3d[:, 0] / z_safe + cx
        v = fy * pts_cam_3d[:, 1] / z_safe + cy

        pixel_coords = np.stack([u, v], axis=1)

        h, w = image_shape
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        valid_mask = in_front & in_bounds

        return pixel_coords, valid_mask

    def _extract_features(
        self,
        image: np.ndarray,
        pixel_coords: np.ndarray,
        intensity: np.ndarray | None,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Sample per-point features from the image.

        Args:
            image: (H, W, 3) BGR image.
            pixel_coords: (M, 2) pixel locations of fused points.
            intensity: (N,) full intensity array (before masking) or None.
            valid_mask: (N,) mask used to select fused points.

        Returns:
            (M, D) feature array.
        """
        u = pixel_coords[:, 0].astype(np.int32)
        v = pixel_coords[:, 1].astype(np.int32)

        # BGR -> RGB
        rgb = image[v, u, ::-1].astype(np.float32) / 255.0

        if self.feature_mode == "rgb":
            return rgb

        if self.feature_mode == "rgbi":
            if intensity is not None:
                i = intensity[valid_mask].reshape(-1, 1).astype(np.float32)
                # Normalize intensity to [0, 1]
                i_max = i.max() if i.max() > 0 else 1.0
                i = i / i_max
            else:
                i = np.zeros((len(rgb), 1), dtype=np.float32)
            return np.hstack([rgb, i])

        if self.feature_mode == "learned":
            # Stub: return RGB features as placeholder.
            # Replace with a CNN feature extractor (e.g., pretrained ResNet)
            # that produces per-pixel feature maps, then sample at (u, v).
            return rgb

        raise ValueError(f"Unknown feature_mode: {self.feature_mode}")

    def create_colored_cloud(
        self,
        points: np.ndarray,
        image: np.ndarray,
        calibration: CalibrationData,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience method: return points with RGB colors.

        Returns:
            points: (M, 3) visible points.
            colors: (M, 3) RGB float in [0, 1].
        """
        fused = self.fuse(points, image, calibration)
        return fused.point_cloud, fused.features[:, :3]
