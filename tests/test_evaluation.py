"""Tests for evaluation metrics."""

import numpy as np
import pytest

from forest_panoptic_nav.evaluation import (
    EvaluationResult,
    compute_confusion_matrix,
    compute_iou_from_confusion,
    evaluate_segmentation,
)


class TestConfusionMatrix:
    def test_perfect_predictions(self):
        gt = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([0, 0, 1, 1, 2, 2])
        cm = compute_confusion_matrix(pred, gt, num_classes=3)
        expected = np.diag([2, 2, 2])
        np.testing.assert_array_equal(cm, expected)

    def test_all_wrong(self):
        gt = np.array([0, 0, 1, 1])
        pred = np.array([1, 1, 0, 0])
        cm = compute_confusion_matrix(pred, gt, num_classes=2)
        assert cm[0, 0] == 0
        assert cm[0, 1] == 2  # gt=0 predicted as 1
        assert cm[1, 0] == 2  # gt=1 predicted as 0

    def test_shape(self):
        gt = np.array([0, 1, 2])
        pred = np.array([0, 1, 2])
        cm = compute_confusion_matrix(pred, gt, num_classes=7)
        assert cm.shape == (7, 7)

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            compute_confusion_matrix(np.array([0, 1]), np.array([0]))


class TestIoUFromConfusion:
    def test_perfect(self):
        cm = np.diag([10, 20, 30]).astype(np.int64)
        iou = compute_iou_from_confusion(cm)
        for c in iou:
            assert iou[c] == pytest.approx(1.0)

    def test_no_overlap(self):
        cm = np.array([[0, 10], [10, 0]], dtype=np.int64)
        iou = compute_iou_from_confusion(cm)
        assert iou[0] == pytest.approx(0.0)
        assert iou[1] == pytest.approx(0.0)

    def test_partial_overlap(self):
        # Class 0: TP=5, FP=5, FN=5 -> IoU = 5/15 = 1/3
        cm = np.array([[5, 5], [5, 5]], dtype=np.int64)
        iou = compute_iou_from_confusion(cm)
        assert iou[0] == pytest.approx(5.0 / 15.0)
        assert iou[1] == pytest.approx(5.0 / 15.0)

    def test_empty_class_excluded(self):
        cm = np.zeros((3, 3), dtype=np.int64)
        cm[0, 0] = 10
        iou = compute_iou_from_confusion(cm)
        assert 0 in iou
        assert 1 not in iou
        assert 2 not in iou


class TestEvaluateSegmentation:
    def test_perfect_segmentation(self):
        gt = np.array([1, 1, 2, 2, 6, 6])
        result = evaluate_segmentation(gt, gt)
        assert result.miou == pytest.approx(1.0)
        assert result.overall_accuracy == pytest.approx(1.0)

    def test_overall_accuracy(self):
        gt = np.array([0, 0, 1, 1])
        pred = np.array([0, 1, 1, 1])  # 3/4 correct
        result = evaluate_segmentation(pred, gt)
        assert result.overall_accuracy == pytest.approx(0.75)

    def test_miou_partial(self):
        gt = np.array([0, 0, 0, 1, 1, 1])
        pred = np.array([0, 0, 1, 1, 1, 1])
        result = evaluate_segmentation(pred, gt)
        # Class 0: TP=2, FP=0, FN=1 -> IoU=2/3
        # Class 1: TP=3, FP=1, FN=0 -> IoU=3/4
        # mIoU = (2/3 + 3/4) / 2
        expected_miou = (2.0 / 3.0 + 3.0 / 4.0) / 2.0
        assert result.miou == pytest.approx(expected_miou)

    def test_summary_string(self):
        gt = np.array([1, 1, 2, 2])
        result = evaluate_segmentation(gt, gt)
        summary = result.summary()
        assert "mIoU" in summary
        assert "accuracy" in summary.lower()

    def test_returns_confusion_matrix(self):
        gt = np.array([0, 1, 2])
        pred = np.array([0, 1, 2])
        result = evaluate_segmentation(pred, gt)
        assert isinstance(result.confusion_matrix, np.ndarray)
        assert result.confusion_matrix.shape == (7, 7)

    def test_empty_arrays(self):
        gt = np.array([], dtype=np.int32)
        pred = np.array([], dtype=np.int32)
        result = evaluate_segmentation(pred, gt)
        assert result.miou == 0.0
        assert result.overall_accuracy == 0.0
