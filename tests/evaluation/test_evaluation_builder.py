from typing import Any

import pytest
from pydantic import ValidationError

from src.evaluation.evaluation_builder import EvaluationRunner, EvaluationSuiteBuilder
from src.evaluation.metrics import BaseMetric, ClassificationReportMetric, RulesJaccardMetric
from src.evaluation.models import EvaluationMetrics


class DummyMetric(BaseMetric):
    def __init__(self, return_data: dict[str, Any]):
        self.return_data = return_data

    def compute(self, records: list) -> dict[str, Any]:
        return self.return_data


def test_builder_adds_metrics_and_builds():
    builder = EvaluationSuiteBuilder()
    runner = (
        builder
        .add_classification_metrics()
        .add_rules_metrics()
        .build()
    )

    assert isinstance(runner, EvaluationRunner)
    assert len(runner.metrics) == 2
    assert isinstance(runner.metrics[0], ClassificationReportMetric)
    assert isinstance(runner.metrics[1], RulesJaccardMetric)


def test_runner_collects_metrics_properly():
    metric_1 = DummyMetric({"accuracy": 0.95, "macro_f1": 0.90})
    metric_2 = DummyMetric({"iou": 0.80})

    runner = EvaluationRunner(metrics=[metric_1, metric_2])
    
    result = runner.run(records=[])

    assert isinstance(result, EvaluationMetrics)
    assert result.accuracy == 0.95
    assert result.macro_f1 == 0.90
    assert result.iou == 0.80



def test_runner_raises_validation_error_on_invalid_data():
    bad_metric = DummyMetric({"accuracy": "not_a_number_str"})
    runner = EvaluationRunner(metrics=[bad_metric])

    with pytest.raises(ValidationError):
        runner.run(records=[])


def test_runner_ignores_unknown_fields_by_default():
    unknown_field_metric = DummyMetric({"non_existent_metric_key": 123})
    runner = EvaluationRunner(metrics=[unknown_field_metric])

    result = runner.run(records=[])

    assert isinstance(result, EvaluationMetrics)
    
    assert not hasattr(result, "non_existent_metric_key")