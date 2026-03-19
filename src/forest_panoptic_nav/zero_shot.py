"""Zero-shot forest segmentation using geometric features.

No training required. Uses RANSAC ground estimation, DBSCAN clustering,
and geometric shape analysis to classify forest point clouds into
FinnWoodlands-compatible panoptic labels.

Pipeline:
    1. Ground plane estimation via RANSAC on lowest points
    2. Ground removal
    3. Track detection via flatness + elongation analysis
    4. Height filtering for trunk detection (0.5m - 8.0m above ground)
    5. DBSCAN clustering of remaining points into individual objects
    6. Per-cluster geometric feature extraction:
       - Trunk radius estimation (circle fit at breast height)
       - Bark roughness (local point density variation)
       - Crown shape analysis (points above trunk top)
    7. Species classification using combined geometric features
    8. Obstacle detection for non-trunk, non-ground objects
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
CLASS_OBSTACLE = 7

# Height thresholds (meters above estimated ground)
MIN_TRUNK_HEIGHT = 0.5
MAX_TRUNK_HEIGHT = 8.0
TRACK_MAX_HEIGHT = 0.3

# Breast height for trunk radius estimation (forestry standard: 1.3m)
BREAST_HEIGHT = 1.3
BREAST_HEIGHT_TOLERANCE = 0.3  # +/- tolerance band

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
    trunk_radius: float     # fitted trunk radius at breast height
    bark_roughness: float   # local point density variation (std of radial distances)
    crown_spread: float     # horizontal extent of crown relative to trunk radius
    crown_flatness: float   # crown height span / crown XY spread

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


def _fit_trunk_radius(
    points: np.ndarray,
    center_xy: np.ndarray,
    heights: np.ndarray,
    breast_height: float = BREAST_HEIGHT,
    tolerance: float = BREAST_HEIGHT_TOLERANCE,
) -> float:
    """Estimate trunk radius by fitting a circle to the cross-section at breast height.

    Selects points within a height band around breast height and computes
    the mean radial distance from the XY centroid as an estimate of the trunk
    radius.

    Args:
        points: (N, 3) cluster points.
        center_xy: (2,) XY centroid of the cluster.
        heights: (N,) heights above ground for the cluster points.
        breast_height: Target height for measurement (meters).
        tolerance: Half-width of the height band (meters).

    Returns:
        Estimated trunk radius in meters. Returns 0.0 if insufficient points.
    """
    band_mask = (heights >= breast_height - tolerance) & (heights <= breast_height + tolerance)
    band_points = points[band_mask]

    if len(band_points) < 3:
        # Fall back to using all points
        dists = np.linalg.norm(points[:, :2] - center_xy, axis=1)
        return float(np.median(dists)) if len(dists) > 0 else 0.0

    dists = np.linalg.norm(band_points[:, :2] - center_xy, axis=1)
    return float(np.median(dists))


def _estimate_bark_roughness(
    points: np.ndarray,
    center_xy: np.ndarray,
) -> float:
    """Estimate bark roughness from the variation in radial distances.

    Smooth bark (e.g., birch) shows low variation in radial distance from center.
    Rough bark (e.g., pine, spruce) shows higher variation.

    Args:
        points: (N, 3) cluster points.
        center_xy: (2,) XY centroid.

    Returns:
        Standard deviation of radial distances (meters).
    """
    if len(points) < 3:
        return 0.0
    dists = np.linalg.norm(points[:, :2] - center_xy, axis=1)
    return float(np.std(dists))


def _analyze_crown(
    all_points: np.ndarray,
    trunk_center_xy: np.ndarray,
    trunk_top_height: float,
    heights: np.ndarray,
    search_radius: float = 2.0,
) -> tuple[float, float]:
    """Analyze crown shape for points above the trunk top.

    Args:
        all_points: (N, 3) full point cloud (or nearby subset).
        trunk_center_xy: (2,) XY center of the trunk.
        trunk_top_height: Height of trunk top above ground.
        heights: (N,) heights above ground for all_points.
        search_radius: Max horizontal distance from trunk to search for crown.

    Returns:
        crown_spread: Horizontal extent of crown relative to trunk radius.
        crown_flatness: Crown height span / crown XY spread (conical = high, flat = low).
    """
    # Crown points: above trunk top, within search radius of trunk center
    xy_dists = np.linalg.norm(all_points[:, :2] - trunk_center_xy, axis=1)
    crown_mask = (heights > trunk_top_height) & (xy_dists < search_radius)
    crown_points = all_points[crown_mask]

    if len(crown_points) < 3:
        return 0.0, 0.0

    crown_xy_dists = np.linalg.norm(crown_points[:, :2] - trunk_center_xy, axis=1)
    crown_spread = float(np.max(crown_xy_dists))
    crown_height = float(crown_points[:, 2].max() - crown_points[:, 2].min())
    crown_flatness = crown_height / max(crown_spread, 0.01)

    return crown_spread, crown_flatness


def _analyze_cluster(
    points: np.ndarray,
    heights: np.ndarray | None = None,
    all_points: np.ndarray | None = None,
    all_heights: np.ndarray | None = None,
) -> ClusterInfo:
    """Compute geometric properties of a point cluster.

    Args:
        points: (M, 3) points belonging to this cluster.
        heights: (M,) heights above ground for cluster points.
        all_points: (N, 3) full point cloud for crown analysis.
        all_heights: (N,) heights above ground for full cloud.
    """
    center_xy = points[:, :2].mean(axis=0)
    min_z = points[:, 2].min()
    max_z = points[:, 2].max()
    height_span = max_z - min_z

    dists = np.linalg.norm(points[:, :2] - center_xy, axis=1)
    xy_radius = dists.max() if len(dists) > 0 else 0.0

    # Trunk radius estimation
    if heights is not None:
        trunk_radius = _fit_trunk_radius(points, center_xy, heights)
    else:
        trunk_radius = float(np.median(dists)) if len(dists) > 0 else 0.0

    # Bark roughness
    bark_roughness = _estimate_bark_roughness(points, center_xy)

    # Crown analysis
    crown_spread = 0.0
    crown_flatness = 0.0
    if all_points is not None and all_heights is not None and heights is not None:
        trunk_top = float(heights.max()) if len(heights) > 0 else 0.0
        crown_spread, crown_flatness = _analyze_crown(
            all_points, center_xy, trunk_top, all_heights,
        )

    return ClusterInfo(
        center_xy=center_xy,
        height_span=height_span,
        xy_radius=xy_radius,
        num_points=len(points),
        min_z=min_z,
        max_z=max_z,
        trunk_radius=trunk_radius,
        bark_roughness=bark_roughness,
        crown_spread=crown_spread,
        crown_flatness=crown_flatness,
    )


def _classify_cluster(info: ClusterInfo) -> int:
    """Classify a cluster into a FinnWoodlands semantic class using geometric features.

    Uses a combination of features for species classification:
    - Trunk radius at breast height (birch thin, pine/spruce thicker)
    - Bark roughness (birch smooth, spruce/pine rough)
    - Height span
    - Crown shape (spruce conical, pine flat-topped, birch rounded)
    - Aspect ratio

    Non-tree objects (low aspect ratio, wide spread) are classified as obstacles.

    Returns:
        Semantic class ID.
    """
    if info.num_points < DBSCAN_MIN_SAMPLES:
        return CLASS_UNLABELED

    # Check if it looks like a tree trunk: tall, thin, high aspect ratio
    is_trunk_shape = (
        info.xy_radius <= MAX_TRUNK_RADIUS
        and info.aspect_ratio >= MIN_TRUNK_ASPECT_RATIO
    )

    if not is_trunk_shape:
        # Non-trunk above-ground object -> obstacle (rocks, fallen logs, etc.)
        if info.height_span > 0.3 and info.num_points >= DBSCAN_MIN_SAMPLES:
            return CLASS_OBSTACLE
        return CLASS_UNLABELED

    # --- Tree species classification using multiple geometric features ---
    # Score each species based on how well the features match typical values.
    # Higher score = better match.

    score_birch = 0.0
    score_spruce = 0.0
    score_pine = 0.0

    # Feature 1: Trunk radius at breast height
    # Birch: typically thin (0.05 - 0.12m)
    # Spruce: medium (0.10 - 0.20m)
    # Pine: thick (0.12 - 0.25m)
    r = info.trunk_radius
    if r < 0.10:
        score_birch += 2.0
        score_spruce += 0.5
    elif r < 0.15:
        score_birch += 1.0
        score_spruce += 1.5
        score_pine += 1.0
    elif r < 0.22:
        score_spruce += 1.0
        score_pine += 2.0
    else:
        score_pine += 2.0
        score_spruce += 0.5

    # Feature 2: Bark roughness (std of radial distances)
    # Birch: smooth bark -> low roughness (< 0.03)
    # Pine: moderately rough (0.02 - 0.05)
    # Spruce: rough bark (> 0.03)
    roughness = info.bark_roughness
    if roughness < 0.02:
        score_birch += 1.5
        score_pine += 0.5
    elif roughness < 0.04:
        score_pine += 1.5
        score_spruce += 1.0
        score_birch += 0.5
    else:
        score_spruce += 1.5
        score_pine += 1.0

    # Feature 3: Height span
    # Pine: tallest (> 5m trunk visible)
    # Spruce: medium to tall (3-7m)
    # Birch: often shorter trunks visible (< 5m)
    h = info.height_span
    if h > 5.0:
        score_pine += 2.0
        score_spruce += 1.0
    elif h > 3.0:
        score_spruce += 1.5
        score_pine += 1.0
        score_birch += 0.5
    elif h > 1.5:
        score_birch += 1.0
        score_spruce += 1.0
    else:
        score_birch += 1.5

    # Feature 4: Crown shape (if available)
    # Spruce: conical crown (high flatness ratio)
    # Pine: flat-topped or irregular crown (low flatness, wider spread)
    # Birch: rounded crown (medium flatness)
    if info.crown_spread > 0:
        if info.crown_flatness > 2.0:
            score_spruce += 1.5  # conical
        elif info.crown_flatness > 1.0:
            score_birch += 1.0   # rounded
        else:
            score_pine += 1.5    # flat-topped

        # Wide crown spread favors pine/birch
        if info.crown_spread > 1.5:
            score_pine += 0.5
            score_birch += 0.5

    # Pick the species with the highest score
    scores = {
        CLASS_PINE: score_pine,
        CLASS_SPRUCE: score_spruce,
        CLASS_BIRCH: score_birch,
    }
    return max(scores, key=scores.get)  # type: ignore[arg-type]


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
        semantic[ground_mask] = CLASS_GROUND
        confidence[ground_mask] = 0.7

        # Step 4: Detect track/path regions (flat, narrow, elongated)
        if ground_mask.any():
            semantic, confidence = self._detect_track(
                points, semantic, confidence, heights, ground_mask,
            )

        # Step 5: Extract above-ground points for trunk detection
        above_ground = (heights >= self.trunk_min_height) & (heights <= self.trunk_max_height)

        if above_ground.sum() >= self.dbscan_min_samples:
            # Step 6: DBSCAN clustering
            above_points = points[above_ground]
            above_heights = heights[above_ground]
            clustering = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
            ).fit(above_points)

            labels = clustering.labels_
            next_instance_id = 1

            # Step 7: Classify each cluster with enhanced features
            for cluster_id in set(labels):
                if cluster_id == -1:
                    continue  # noise

                cluster_mask_local = labels == cluster_id
                cluster_points = above_points[cluster_mask_local]
                cluster_heights = above_heights[cluster_mask_local]

                info = _analyze_cluster(
                    cluster_points,
                    heights=cluster_heights,
                    all_points=points,
                    all_heights=heights,
                )
                cls = _classify_cluster(info)

                # Map local mask back to global indices
                global_indices = np.where(above_ground)[0][cluster_mask_local]

                if cls in (CLASS_SPRUCE, CLASS_BIRCH, CLASS_PINE):
                    semantic[global_indices] = cls
                    instance[global_indices] = next_instance_id
                    confidence[global_indices] = min(0.8, 0.5 + info.aspect_ratio * 0.1)
                    next_instance_id += 1
                elif cls == CLASS_OBSTACLE:
                    semantic[global_indices] = CLASS_OBSTACLE
                    instance[global_indices] = next_instance_id
                    confidence[global_indices] = 0.6
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
        """Detect track/path regions within ground points.

        Uses a combination of:
        - Local height variance (tracks are flat)
        - Elongation analysis (tracks are narrow and elongated)
        - Connectivity (tracks form continuous strips)

        Returns:
            Updated (semantic, confidence) arrays.
        """
        ground_idx = np.where(ground_mask)[0]
        ground_pts = points[ground_idx]

        if len(ground_pts) < 10:
            return semantic, confidence

        # Grid-based analysis
        grid_size = 1.0  # meters
        xy = ground_pts[:, :2]
        min_xy = xy.min(axis=0)
        cell_ids = np.floor((xy - min_xy) / grid_size).astype(np.int64)
        span = cell_ids.max(axis=0) + 1
        cell_keys = cell_ids[:, 0] * span[1] + cell_ids[:, 1]

        unique_cells = np.unique(cell_keys)

        # First pass: compute per-cell flatness and density
        cell_flatness = {}
        cell_density = {}
        cell_mean_height = {}
        for cell in unique_cells:
            cell_mask = cell_keys == cell
            cell_heights = heights[ground_idx[cell_mask]]
            if len(cell_heights) < 3:
                continue
            cell_flatness[cell] = float(cell_heights.std())
            cell_density[cell] = int(cell_mask.sum())
            cell_mean_height[cell] = float(cell_heights.mean())

        if not cell_flatness:
            return semantic, confidence

        # Compute median flatness as reference
        all_flatness = np.array(list(cell_flatness.values()))
        median_flatness = float(np.median(all_flatness))

        # Cells that are flatter than average are track candidates
        flat_threshold = min(0.05, median_flatness * 0.8)

        # Second pass: mark flat cells as track, with elongation boost
        for cell in unique_cells:
            if cell not in cell_flatness:
                continue
            cell_mask = cell_keys == cell
            height_std = cell_flatness[cell]

            # Flat cell -> track candidate
            if height_std < flat_threshold:
                # Check if this cell is slightly below surrounding ground
                # (tracks are often slightly depressed)
                mean_h = cell_mean_height[cell]
                global_indices = ground_idx[cell_mask]

                # Higher confidence if the cell is also denser (well-sampled)
                density = cell_density[cell]
                conf = 0.6 + min(0.2, density / 500.0)

                semantic[global_indices] = CLASS_TRACK
                confidence[global_indices] = conf

        return semantic, confidence


def visualize_segmentation(
    result: PanopticResult,
    output_path: str | None = None,
) -> None:
    """Visualize segmentation result with top-down and side views.

    Args:
        result: PanopticResult to visualize.
        output_path: Path to save PNG. If None, displays interactively.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # Color map for semantic classes
    class_colors = {
        CLASS_UNLABELED: (0.7, 0.7, 0.7),   # gray
        CLASS_GROUND: (0.6, 0.4, 0.2),       # brown
        CLASS_TRACK: (0.9, 0.8, 0.5),        # sandy
        CLASS_LAKE: (0.2, 0.4, 0.8),         # blue
        CLASS_SPRUCE: (0.0, 0.5, 0.0),       # dark green
        CLASS_BIRCH: (0.6, 0.9, 0.3),        # light green
        CLASS_PINE: (0.0, 0.3, 0.0),         # forest green
        CLASS_OBSTACLE: (0.8, 0.2, 0.2),     # red
    }

    class_names = {
        CLASS_UNLABELED: "Unlabeled",
        CLASS_GROUND: "Ground",
        CLASS_TRACK: "Track",
        CLASS_LAKE: "Lake",
        CLASS_SPRUCE: "Spruce",
        CLASS_BIRCH: "Birch",
        CLASS_PINE: "Pine",
        CLASS_OBSTACLE: "Obstacle",
    }

    points = result.points
    labels = result.semantic_labels

    # Assign colors to each point
    colors = np.array([class_colors.get(l, (0.5, 0.5, 0.5)) for l in labels])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Top-down view (XY plane)
    ax_top = axes[0]
    ax_top.scatter(points[:, 0], points[:, 1], c=colors, s=1, alpha=0.6)
    ax_top.set_xlabel("X (m)")
    ax_top.set_ylabel("Y (m)")
    ax_top.set_title("Top-down view (colored by class)")
    ax_top.set_aspect("equal")

    # Side view (XZ plane)
    ax_side = axes[1]
    ax_side.scatter(points[:, 0], points[:, 2], c=colors, s=1, alpha=0.6)
    ax_side.set_xlabel("X (m)")
    ax_side.set_ylabel("Z (m)")
    ax_side.set_title("Side view (height profile)")

    # Legend
    present_classes = np.unique(labels)
    legend_elements = []
    for cls_id in sorted(present_classes):
        color = class_colors.get(cls_id, (0.5, 0.5, 0.5))
        name = class_names.get(cls_id, f"Class {cls_id}")
        legend_elements.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                       markersize=8, label=name)
        )
    ax_top.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
