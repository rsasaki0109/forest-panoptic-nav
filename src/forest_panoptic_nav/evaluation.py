"""Evaluation metrics for panoptic segmentation.

Computes mean Intersection over Union (mIoU) and per-class IoU between
predicted and ground-truth semantic labels, compatible with FinnWoodlands
panoptic annotation format.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .loader import SEMANTIC_CLASSES


@dataclass
class EvaluationResult:
    """Evaluation metrics for a segmentation prediction."""

    per_class_iou: dict[int, float]     # class_id -> IoU
    miou: float                         # mean IoU across present classes
    per_class_accuracy: dict[int, float]  # class_id -> accuracy
    overall_accuracy: float             # total correct / total points
    confusion_matrix: np.ndarray        # (num_classes, num_classes) counts

    def summary(self) -> str:
        """Human-readable summary of evaluation results."""
        lines = [
            f"Overall accuracy: {self.overall_accuracy:.3f}",
            f"mIoU: {self.miou:.3f}",
            "",
            "Per-class results:",
            f"  {'Class':<12s} {'IoU':>6s}  {'Acc':>6s}",
            f"  {'-'*12} {'-'*6}  {'-'*6}",
        ]
        for cls_id in sorted(self.per_class_iou.keys()):
            name = SEMANTIC_CLASSES.get(cls_id, f"cls_{cls_id}")
            iou = self.per_class_iou[cls_id]
            acc = self.per_class_accuracy.get(cls_id, 0.0)
            lines.append(f"  {name:<12s} {iou:>6.3f}  {acc:>6.3f}")
        return "\n".join(lines)


def compute_confusion_matrix(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    num_classes: int = 7,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        predictions: (N,) int array of predicted class labels.
        ground_truth: (N,) int array of ground-truth class labels.
        num_classes: Total number of classes (0 through num_classes-1).

    Returns:
        (num_classes, num_classes) confusion matrix where
        cm[gt, pred] = count of points with true label gt predicted as pred.
    """
    assert len(predictions) == len(ground_truth), (
        f"Length mismatch: predictions={len(predictions)}, ground_truth={len(ground_truth)}"
    )
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for gt, pred in zip(ground_truth, predictions):
        if 0 <= gt < num_classes and 0 <= pred < num_classes:
            cm[gt, pred] += 1
    return cm


def compute_iou_from_confusion(cm: np.ndarray) -> dict[int, float]:
    """Compute per-class IoU from a confusion matrix.

    IoU(c) = TP(c) / (TP(c) + FP(c) + FN(c))

    Only classes present in ground truth or predictions are included.

    Args:
        cm: (C, C) confusion matrix.

    Returns:
        Dict mapping class_id to IoU value.
    """
    num_classes = cm.shape[0]
    iou = {}
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        if denom == 0:
            continue  # class not present
        iou[c] = float(tp / denom)
    return iou


def evaluate_segmentation(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    num_classes: int = 7,
) -> EvaluationResult:
    """Evaluate semantic segmentation predictions against ground truth.

    Args:
        predictions: (N,) int array of predicted semantic labels.
        ground_truth: (N,) int array of ground-truth semantic labels.
        num_classes: Number of semantic classes (0 through num_classes-1).

    Returns:
        EvaluationResult with mIoU, per-class IoU, accuracy, and confusion matrix.
    """
    predictions = np.asarray(predictions, dtype=np.int32)
    ground_truth = np.asarray(ground_truth, dtype=np.int32)

    cm = compute_confusion_matrix(predictions, ground_truth, num_classes)
    per_class_iou = compute_iou_from_confusion(cm)

    # Mean IoU over classes that are present
    if per_class_iou:
        miou = float(np.mean(list(per_class_iou.values())))
    else:
        miou = 0.0

    # Per-class accuracy: TP / (TP + FN) for each class
    per_class_accuracy = {}
    for c in range(num_classes):
        total = cm[c, :].sum()
        if total > 0:
            per_class_accuracy[c] = float(cm[c, c] / total)

    # Overall accuracy
    total_points = cm.sum()
    overall_accuracy = float(np.trace(cm) / total_points) if total_points > 0 else 0.0

    return EvaluationResult(
        per_class_iou=per_class_iou,
        miou=miou,
        per_class_accuracy=per_class_accuracy,
        overall_accuracy=overall_accuracy,
        confusion_matrix=cm,
    )
