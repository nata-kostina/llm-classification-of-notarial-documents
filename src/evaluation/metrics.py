from abc import ABC, abstractmethod
from typing import Any

from sklearn.metrics import classification_report

from src.evaluation.models import DocumentClassificationResult


class BaseMetric(ABC):
    name: str

    @abstractmethod
    def compute(self, records: list[DocumentClassificationResult]) -> dict[str, Any]:
        pass


class ClassificationReportMetric(BaseMetric):
    name = "classification_report"

    def compute(self, records: list[DocumentClassificationResult]) -> dict[str, Any]:
        y_true = [r.label_true for r in records]

        y_pred = [(r.label_pred if (r.label_pred is not None and str(r.label_pred) != "nan") else "ERROR") for r in records]

        dict_report: dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)  # type: ignore

        values = {
            "accuracy": float(dict_report["accuracy"]),
            "macro_precision": float(dict_report["macro avg"]["precision"]),
            "macro_recall": float(dict_report["macro avg"]["recall"]),
            "macro_f1": float(dict_report["macro avg"]["f1-score"]),
            "weighted_precision": float(dict_report["weighted avg"]["precision"]),
            "weighted_recall": float(dict_report["weighted avg"]["recall"]),
            "weighted_f1": float(dict_report["weighted avg"]["f1-score"]),
            "classification_report": dict_report,
        }
        return values


class RulesJaccardMetric(BaseMetric):
    name = "rules_jaccard"

    def compute(self, records: list[DocumentClassificationResult]) -> dict[str, Any]:
        if not records:
            return {"iou": 0.0}

        scores = []
        for r in records:
            set_true = set(r.rules_true)
            set_pred = set(r.rules_pred) if r.rules_pred is not None else set()

            union = set_true | set_pred
            intersection = set_true & set_pred

            score = len(intersection) / len(union) if union else 1.0
            scores.append(score)

        mean_iou = sum(scores) / len(records)
        return {"iou": float(mean_iou)}
