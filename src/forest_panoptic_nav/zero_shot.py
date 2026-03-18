"""Zero-shot forest segmentation using geometric features.

No training required. Uses RANSAC ground estimation, DBSCAN clustering,
and geometric shape analysis to classify forest point clouds into
FinnWoodlands-compatible panoptic labels.

Pipeline:
    1. Ground plane estimation via RANSAC on lowest points
    2. Ground removal
    3. Height filtering for trunk detection (0.5m - 3.0m above ground)
    4. DBSCAN clustering of remaining points into individual objects
    5. Shape-based classification: tall+thin = tree trunk, wide+flat = obstacle, etc.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN

from .loader import SEMANTIC_CLASSES
from .segmentation import PanopticResult


# FinnWoodlands class IDs
CLASS_UNLABELED = 0
CLASS_GROUND = 1
CLASS_TRACK = 2
CLASS_LAKE = 3
CLASS_SPRUCE = 4
CLASS_BIRCH = 5
CLASS_PINE = 6

# Height thresholds (meters above estimated ground)
MIN_TRUNK_HEIGHT = 0.5
MAX_TRUNK_HEIGHT = 8.0
TRACK_MAX_HEIGHT = 0.3

# DBSCAN parameters
DBSCAN_EPS = 0.5            # max distance between points in a cluster (meters)
DBSCAN_MIN_SAMPLES = 5      # minimum points to form a cluster

# Shape classification thresholds
MAX_TRUNK_RADIUS = 0.5      # max XY radius for a tree trunk cluster (meters)
MIN_TRUNK_ASPECT_RATIO = 1.5  # height / diameter ratio for a trunk


@dataclass
class ClusterInfo:
    """Geometric properties of a point cluster."""

    center_xy: np.ndarray   # (2,) centroid in XY
    height_span: float      # max_z - min_z
    xy_radius: float        # max distance from centroid in XY
    num_points: int
    min_z: float
    max_z: float

    @property
    def aspect_ratio(self) -> float:
        """Height-to-diameter ratio."""
        diameter = max(2.0 * self.xy_radius, 0.01)
        return self.height_span / diameter


def _ransac_ground_plane(
    points: np.ndarray,
    n_iterations: int = 100,
    distance_threshold: float = 0.15,
    lowest_fraction: float = 0.3,
) -> tuple[np.ndarray, float]:
    """Estimate ground plane using RANSAC on the lowest points.

    Args:
        points: (N, 3) point cloud.
        n_iterations: Number of RANSAC iterations.
        distance_threshold: Inlier distance threshold in meters.
        lowest_fraction: Fraction of points (by Z) to consider as ground candidates.

    Returns:
        normal: (3,) unit normal of the ground plane.
        d: Plane offset (plane equation: normal . x + d = 0).
    """
    z = points[:, 2]
    z_thresh = np.percentile(z, lowest_fraction * 100)
    candidates = points[z <= z_thresh]

    if len(candidates) < 3:
        # Fallback: assume flat ground at min Z
        return np.array([0.0, 0.0, 1.0]), -z.min()

    rng = np.random.default_rng(42)
    best_inliers = 0
    best_normal = np.array([0.0, 0.0, 1.0])
    best_d = -z.min()

    for _ in range(n_iterations):
        idx = rng.choice(len(candidates), size=3, replace=False)
        p0, p1, p2 = candidates[idx]

        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm

        # Ensure normal points upward (positive Z component)
        if normal[2] < 0:
            normal = -normal

        d = -np.dot(normal, p0)
        distances = np.abs(points @ normal + d)
        n_inliers = (distances < distance_threshold).sum()

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_normal = normal
            best_d = d

    return best_normal, best_d


def _height_above_ground(
    points: np.ndarray,
    normal: np.ndarray,
    d: float,
) -> np.ndarray:
    """Compute signed height above the ground plane for each point.

    Args:
        points: (N, 3) point cloud.
        normal: (3,) ground plane normal (pointing up).
        d: Ground plane offset.

    Returns:
        (N,) heights above ground in meters.
    """
    return points @ normal + d


def _analyze_cluster(points: np.ndarray) -> ClusterInfo:
    """Compute geometric properties of a point cluster."""
    center_xy = points[:, :2].mean(axis=0)
    min_z = points[:, 2].min()
    max_z = points[:, 2].max()
    height_span = max_z - min_z

    dists = np.linalg.norm(points[:, :2] - center_xy, axis=1)
    xy_radius = dists.max() if len(dists) > 0 else 0.0

    return ClusterInfo(
        center_xy=center_xy,
        height_span=height_span,
        xy_radius=xy_radius,
        num_points=len(points),
        min_z=min_z,
        max_z=max_z,
    )


def _classify_cluster(info: ClusterInfo) -> int:
    """Classify a cluster into a FinnWoodlands semantic class based on shape.

    Rules:
    - Tall and thin (high aspect ratio, small radius) -> tree trunk
    - Wide and flat -> obstacle or unlabeled
    - Very small clusters -> unlabeled

    Tree species assignment is based on height heuristic:
    - Tallest trunks -> Pine (6) (pines tend to be tallest)
    - Medium -> Spruce (4)
    - Shortest -> Birch (5)

    Returns:
        Semantic class ID.
    """
    if info.num_points < DBSCAN_MIN_SAMPLES:
        return CLASS_UNLABELED

    if info.xy_radius <= MAX_TRUNK_RADIUS and info.aspect_ratio >= MIN_TRUNK_ASPECT_RATIO:
        # Tree trunk - assign species by height (rough heuristic)
        if info.height_span > 4.0:
            return CLASS_PINE
        elif info.height_span > 2.0:
            return CLASS_SPRUCE
        else:
            return CLASS_BIRCH

    # Large non-trunk cluster
    return CLASS_UNLABELED


class ZeroShotForestSegmenter:
    """Zero-shot forest point cloud segmenter using geometric features.

    No ML model or training data required. Uses RANSAC ground estimation,
    DBSCAN clustering, and shape-based classification to produce panoptic
    labels compatible with the FinnWoodlands dataset classes.

    Args:
        ground_distance_threshold: RANSAC inlier distance for ground plane (meters).
        ransac_iterations: Number of RANSAC iterations for ground estimation.
        dbscan_eps: DBSCAN epsilon parameter (meters).
        dbscan_min_samples: DBSCAN minimum cluster size.
        trunk_min_height: Minimum height above ground for trunk points (meters).
        trunk_max_height: Maximum height above ground for trunk points (meters).
        track_max_height: Maximum height above ground for track/path points (meters).
    """

    def __init__(
        self,
        ground_distance_threshold: float = 0.15,
        ransac_iterations: int = 100,
        dbscan_eps: float = DBSCAN_EPS,
        dbscan_min_samples: int = DBSCAN_MIN_SAMPLES,
        trunk_min_height: float = MIN_TRUNK_HEIGHT,
        trunk_max_height: float = MAX_TRUNK_HEIGHT,
        track_max_height: float = TRACK_MAX_HEIGHT,
    ) -> None:
        self.ground_distance_threshold = ground_distance_threshold
        self.ransac_iterations = ransac_iterations
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.trunk_min_height = trunk_min_height
        self.trunk_max_height = trunk_max_height
        self.track_max_height = track_max_height

    def predict(
        self,
        points: np.ndarray,
        features: np.ndarray | None = None,
    ) -> PanopticResult:
        """Run zero-shot panoptic segmentation on a forest point cloud.

        Args:
            points: (N, 3) float32 XYZ coordinates.
            features: Ignored (kept for API compatibility).

        Returns:
            PanopticResult with semantic labels and instance IDs.
        """
        n = len(points)
        semantic = np.full(n, CLASS_UNLABELED, dtype=np.int32)
        instance = np.zeros(n, dtype=np.int32)
        confidence = np.full(n, 0.3, dtype=np.float32)

        if n < 3:
            return PanopticResult(
                points=points,
                semantic_labels=semantic,
                instance_ids=instance,
                confidence=confidence,
            )

        # Step 1: Estimate ground plane via RANSAC
        normal, d = _ransac_ground_plane(
            points,
            n_iterations=self.ransac_iterations,
            distance_threshold=self.ground_distance_threshold,
        )

        # Step 2: Compute height above ground
        heights = _height_above_ground(points, normal, d)

        # Step 3: Classify ground and near-ground points
        ground_mask = heights < self.track_max_height
        track_mask = ground_mask & (np.abs(heights) < self.track_max_height * 0.5)

        # Heuristic: very flat areas with low height variation -> track
        # Everything else near ground -> ground
        semantic[ground_mask] = CLASS_GROUND
        confidence[ground_mask] = 0.7

        # Refine: detect track-like regions (flattest ground areas)
        if track_mask.any():
            track_points = points[track_mask]
            if len(track_points) > 10:
                # Use local height variance to distinguish track from rough ground
                semantic, confidence = self._detect_track(
                    points, semantic, confidence, heights, ground_mask,
                )

        # Step 4: Extract above-ground points for trunk detection
        above_ground = (heights >= self.trunk_min_height) & (heights <= self.trunk_max_height)

        if above_ground.sum() >= self.dbscan_min_samples:
            # Step 5: DBSCAN clustering
            above_points = points[above_ground]
            clustering = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
            ).fit(above_points)

            labels = clustering.labels_
            next_instance_id = 1

            # Step 6: Classify each cluster
            for cluster_id in set(labels):
                if cluster_id == -1:
                    continue  # noise

                cluster_mask_local = labels == cluster_id
                cluster_points = above_points[cluster_mask_local]
                info = _analyze_cluster(cluster_points)
                cls = _classify_cluster(info)

                # Map local mask back to global indices
                global_indices = np.where(above_ground)[0][cluster_mask_local]

                if cls in (CLASS_SPRUCE, CLASS_BIRCH, CLASS_PINE):
                    semantic[global_indices] = cls
                    instance[global_indices] = next_instance_id
                    confidence[global_indices] = min(0.8, 0.5 + info.aspect_ratio * 0.1)
                    next_instance_id += 1
                else:
                    semantic[global_indices] = cls
                    confidence[global_indices] = 0.4

        # Points above trunk range but not classified -> unlabeled (canopy, etc.)
        very_high = heights > self.trunk_max_height
        semantic[very_high & (semantic == CLASS_UNLABELED)] = CLASS_UNLABELED
        confidence[very_high & (semantic == CLASS_UNLABELED)] = 0.2

        return PanopticResult(
            points=points,
            semantic_labels=semantic,
            instance_ids=instance,
            confidence=confidence,
        )

    def _detect_track(
        self,
        points: np.ndarray,
        semantic: np.ndarray,
        confidence: np.ndarray,
        heights: np.ndarray,
        ground_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect track/path regions within ground points using local flatness.

        Divides ground points into a grid and marks cells with very low
        height variance as track.

        Returns:
            Updated (semantic, confidence) arrays.
        """
        ground_idx = np.where(ground_mask)[0]
        ground_pts = points[ground_idx]

        if len(ground_pts) < 10:
            return semantic, confidence

        # Grid-based flatness analysis
        grid_size = 1.0  # meters
        xy = ground_pts[:, :2]
        min_xy = xy.min(axis=0)
        cell_ids = np.floor((xy - min_xy) / grid_size).astype(np.int64)
        span = cell_ids.max(axis=0) + 1
        cell_keys = cell_ids[:, 0] * span[1] + cell_ids[:, 1]

        unique_cells = np.unique(cell_keys)
        for cell in unique_cells:
            cell_mask = cell_keys == cell
            cell_heights = heights[ground_idx[cell_mask]]
            if len(cell_heights) < 3:
                continue
            height_std = cell_heights.std()
            # Very flat cells -> track
            if height_std < 0.05:
                global_indices = ground_idx[cell_mask]
                semantic[global_indices] = CLASS_TRACK
                confidence[global_indices] = 0.6

        return semantic, confidence
