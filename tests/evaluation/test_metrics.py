from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.metrics import ClassificationReportMetric, RulesJaccardMetric
from src.evaluation.models import DocumentClassificationResult


def test_classification_report_metric_replaces_invalid_preds():
    records = cast(
        list[DocumentClassificationResult], [
            MagicMock(label_true="A", label_pred="A"),
            MagicMock(label_true="B", label_pred=None),
            MagicMock(label_true="A", label_pred="nan"),
            MagicMock(label_true="B", label_pred=float("nan")),
        ]
    )

    metric = ClassificationReportMetric()

    with patch(
        "src.evaluation.metrics.classification_report"
    ) as mock_sklearn_report:
        mock_sklearn_report.return_value = {
            "accuracy": 0.5,
            "macro avg": {"precision": 0.5, "recall": 0.5, "f1-score": 0.5},
            "weighted avg": {"precision": 0.5, "recall": 0.5, "f1-score": 0.5},
        }

        metric.compute(records)

        mock_sklearn_report.assert_called_once_with(
            ["A", "B", "A", "B"],
            ["A", "ERROR", "ERROR", "ERROR"],
            output_dict=True,
            zero_division=0,
        )


def test_classification_report_metric_structure():
    records = cast(
        list[DocumentClassificationResult],
        [
            MagicMock(label_true="cat", label_pred="cat"),
            MagicMock(label_true="dog", label_pred="cat"),
        ],
    )
    
    metric = ClassificationReportMetric()
    result = metric.compute(records)

    expected_keys = {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "classification_report",
    }
    assert set(result.keys()) == expected_keys

    assert isinstance(result["accuracy"], float)
    assert isinstance(result["macro_f1"], float)
    assert isinstance(result["classification_report"], dict)
    
@pytest.fixture
def metric():
    return RulesJaccardMetric()


def test_rules_jaccard_empty_records(metric):
    result = metric.compute([])
    assert result == {"iou": 0.0}


def test_rules_jaccard_both_rules_empty(metric):
    records = cast(
        list[DocumentClassificationResult],
        [MagicMock(rules_true=[], rules_pred=[])],
    )
    result = metric.compute(records)
    assert result == {"iou": 1.0}


def test_rules_jaccard_pred_is_none(metric):
    records = cast(
        list[DocumentClassificationResult],
        [MagicMock(rules_true=["rule1", "rule2"], rules_pred=None)],
    )
    result = metric.compute(records)
    assert result == {"iou": 0.0}


def test_rules_jaccard_calculation(metric):
    records = cast(
        list[DocumentClassificationResult],
        [
            MagicMock(rules_true=["r1", "r2"], rules_pred=["r1", "r2"]),
            MagicMock(rules_true=["r1", "r2"], rules_pred=["r2", "r3"]),
            MagicMock(rules_true=["r1"], rules_pred=["r2"]),
        ],
    )
    # (1.0 + 1/3 + 0.0) / 3 = 1.3333333... / 3 ≈ 0.4444444...
    result = metric.compute(records)
    
    assert isinstance(result["iou"], float)
    assert result["iou"] == pytest.approx(4 / 9)


def test_rules_jaccard_calculation_empty_pred_rules(metric):
    records = cast(
        list[DocumentClassificationResult],
        [
            MagicMock(rules_true=["r1", "r2"], rules_pred=[]),
        ],
    )
    result = metric.compute(records)
    
    assert isinstance(result["iou"], float)
    assert result["iou"] == 0.0