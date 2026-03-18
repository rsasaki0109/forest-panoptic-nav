"""Panoptic segmentation module for forest environments.

Panoptic segmentation divides the scene into:
- Things (instance segmentation): individual tree trunks by species
    - Spruce (class 4)
    - Birch (class 5)
    - Pine (class 6)
- Stuff (semantic segmentation): terrain categories
    - Ground (class 1)
    - Track (class 2)
    - Lake (class 3)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .loader import SEMANTIC_CLASSES, STUFF_CLASSES, THING_CLASSES


@dataclass
class PanopticResult:
    """Result of panoptic segmentation for one frame."""

    points: np.ndarray              # (N, 3) XYZ coordinates
    semantic_labels: np.ndarray     # (N,) semantic class per point
    instance_ids: np.ndarray        # (N,) instance id per point (0 = stuff / unlabeled)
    confidence: np.ndarray          # (N,) confidence score per point [0, 1]

    @property
    def num_instances(self) -> int:
        unique = np.unique(self.instance_ids)
        return int((unique > 0).sum())

    @property
    def num_semantic_classes(self) -> int:
        return int(len(np.unique(self.semantic_labels)))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            points=self.points,
            semantic_labels=self.semantic_labels,
            instance_ids=self.instance_ids,
            confidence=self.confidence,
        )

    @classmethod
    def load(cls, path: Path) -> PanopticResult:
        data = np.load(path)
        return cls(
            points=data["points"],
            semantic_labels=data["semantic_labels"],
            instance_ids=data["instance_ids"],
            confidence=data["confidence"],
        )

    def get_instances(self, semantic_class: int | None = None) -> list[np.ndarray]:
        """Return list of point arrays, one per instance.

        Args:
            semantic_class: If given, only return instances of this class.
        """
        instances = []
        for iid in np.unique(self.instance_ids):
            if iid == 0:
                continue
            mask = self.instance_ids == iid
            if semantic_class is not None:
                cls_labels = self.semantic_labels[mask]
                majority = np.bincount(cls_labels).argmax()
                if majority != semantic_class:
                    continue
            instances.append(self.points[mask])
        return instances


class PanopticSegmenter:
    """Panoptic segmenter for forest point clouds.

    Supports three methods:
    - ``"zero_shot"`` (default): Geometric zero-shot segmentation using RANSAC
      ground estimation, DBSCAN clustering, and shape-based classification.
      No training data required.
    - ``"heuristic"``: Simple height-percentile-based fallback.
    - ``"ml"``: Trained ML model (requires a model checkpoint).

    The interface is stable: call ``predict`` with a point cloud and optional
    per-point features to get a ``PanopticResult``.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        device: str = "cpu",
        method: str = "zero_shot",
    ) -> None:
        """Initialize the segmenter.

        Args:
            model_path: Path to trained model weights. Only used when method="ml".
            device: Torch device string ('cpu', 'cuda', 'cuda:0', ...).
            method: Segmentation method - "zero_shot", "heuristic", or "ml".
        """
        if method not in ("zero_shot", "heuristic", "ml"):
            raise ValueError(f"Unknown method: {method!r}. Use 'zero_shot', 'heuristic', or 'ml'.")
        self.device = device
        self.method = method
        self.model = None
        self._zero_shot = None

        if method == "ml" and model_path is not None:
            self.model = self._load_model(model_path)
        elif method == "zero_shot":
            from .zero_shot import ZeroShotForestSegmenter
            self._zero_shot = ZeroShotForestSegmenter()

    def _load_model(self, path: Path):
        """Load a trained model from disk.

        Stub — replace with actual model loading (e.g., torch.load).
        """
        raise NotImplementedError(
            f"Model loading not yet implemented. Provide a trained checkpoint at {path}."
        )

    def predict(
        self,
        points: np.ndarray,
        features: np.ndarray | None = None,
    ) -> PanopticResult:
        """Run panoptic segmentation on a point cloud.

        Args:
            points: (N, 3) float32 array of XYZ coordinates.
            features: (N, D) optional per-point feature vectors
                      (e.g., RGB, intensity, fused features).

        Returns:
            PanopticResult with semantic labels and instance IDs.
        """
        if self.method == "ml" and self.model is not None:
            return self._run_model(points, features)
        if self.method == "zero_shot" and self._zero_shot is not None:
            return self._zero_shot.predict(points, features)
        return self._heuristic_predict(points)

    def _run_model(self, points: np.ndarray, features: np.ndarray | None) -> PanopticResult:
        """Run the trained ML model.

        Stub — implement forward pass here.
        Expected to return a PanopticResult.
        """
        raise NotImplementedError("ML model inference not yet implemented.")

    def _heuristic_predict(self, points: np.ndarray) -> PanopticResult:
        """Simple height-based heuristic for development and testing.

        Classification rules (rough approximation):
        - Points below the 20th height percentile -> Ground (1)
        - Points above the 80th height percentile -> tree trunks (assigned Pine=6)
        - Everything in between -> Track (2)

        This is NOT a real segmenter. It exists so the pipeline can run
        end-to-end without a trained model.
        """
        n = len(points)
        z = points[:, 2]

        semantic = np.zeros(n, dtype=np.int32)
        instance = np.zeros(n, dtype=np.int32)
        confidence = np.full(n, 0.5, dtype=np.float32)

        p20, p80 = np.percentile(z, [20, 80])

        ground_mask = z < p20
        mid_mask = (z >= p20) & (z < p80)
        high_mask = z >= p80

        semantic[ground_mask] = 1   # ground
        semantic[mid_mask] = 2      # track
        semantic[high_mask] = 6     # pine (default tree species)

        # Cluster high points into rough instances via simple XY grid
        if high_mask.any():
            high_xy = points[high_mask, :2]
            grid_size = 0.5  # meters
            grid_ids = np.floor(high_xy / grid_size).astype(np.int64)
            # Encode grid cell as unique integer
            offset = grid_ids.min(axis=0)
            grid_ids -= offset
            span = grid_ids.max(axis=0) + 1
            cell_keys = grid_ids[:, 0] * span[1] + grid_ids[:, 1]
            unique_cells, cell_labels = np.unique(cell_keys, return_inverse=True)
            instance[high_mask] = cell_labels + 1  # 1-based instance IDs

        return PanopticResult(
            points=points,
            semantic_labels=semantic,
            instance_ids=instance,
            confidence=confidence,
        )
