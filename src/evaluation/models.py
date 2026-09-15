from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    accuracy: float | None = Field(default=None, json_schema_extra={"mlflow_type": "metrics"})
    macro_precision: float | None = Field(default=None, json_schema_extra={"mlflow_type": "metrics"})
    macro_recall: float | None = Field(default=None, json_schema_extra={"mlflow_type": "metrics"})
    macro_f1: float | None = Field(default=None, json_schema_extra={"mlflow_type": "metrics"})
    weighted_precision: float | None = Field(default=None, json_schema_extra={"mlflow_type": "metrics"})
    weighted_recall: float | None = Field(default=None, json_schema_extra={"mlflow_type": "metrics"})
    weighted_f1: float | None = Field(default=None, json_schema_extra={"mlflow_type": "metrics"})
    classification_report: dict | None = Field(default=None, json_schema_extra={"mlflow_type": "dictionary"})

    iou: float | None = Field(default=None, json_schema_extra={"mlflow_type": "metrics"})


class DocumentClassificationResult(BaseModel):
    act_id: str
    label_true: str
    rules_true: list[str]
    label_pred: str | None
    rules_pred: list[str] | None


class EvaluationResult(BaseModel):
    metrics: EvaluationMetrics
    data: list[DocumentClassificationResult] = Field(json_schema_extra={"mlflow_type": "csv"})


def parse_rules_list(val: Any) -> list[str] | None:
    if not isinstance(val, list) and (val is None or pd.isna(val)):
        return []

    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]

    if isinstance(val, str):
        rules = [item.strip() for item in val.split(",") if item.strip()]
        return rules

    return []


def parse_optional_str(val: Any) -> str | None:
    if isinstance(val, (list, tuple, set)):
        return str(val)

    if val is None or pd.isna(val):
        return None

    return str(val)
