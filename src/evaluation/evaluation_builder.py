from typing import Any

from src.evaluation.metrics import (
    BaseMetric,
    ClassificationReportMetric,
    RulesJaccardMetric,
)
from src.evaluation.models import DocumentClassificationResult, EvaluationMetrics


class EvaluationSuiteBuilder:
    def __init__(self):
        self._metrics: list[BaseMetric] = []

    def add_classification_metrics(self) -> EvaluationSuiteBuilder:
        self._metrics.append(ClassificationReportMetric())
        return self

    def add_rules_metrics(self) -> EvaluationSuiteBuilder:
        self._metrics.append(RulesJaccardMetric())
        return self

    def build(self) -> EvaluationRunner:
        return EvaluationRunner(metrics=self._metrics)


class EvaluationRunner:
    def __init__(self, metrics: list[BaseMetric]):
        self.metrics = metrics

    def run(self, records: list[DocumentClassificationResult]) -> EvaluationMetrics:
        collected_data: dict[str, Any] = {}

        for metric in self.metrics:
            res = metric.compute(records)
            collected_data.update(res)

        return EvaluationMetrics(**collected_data)
