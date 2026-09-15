from pydantic import BaseModel, Field


class PerDocumentRetrievalResult(BaseModel):
    source_doc_id: str
    target_label: str
    total_relevant: int

    retrieved_labels: list[str]
    retrieved_doc_ids: list[str]


class EvaluationMetrics(BaseModel):
    k: int
    mean_precision: float
    mean_hit_rate: float
    mean_reciprocal_rank: float
    mean_average_precision: float
    total_source_docs: int


class DocumentRetrievalResult(BaseModel):
    retrieved_docs: list[PerDocumentRetrievalResult] = Field(json_schema_extra={"mlflow_type": "dictionary"})
    metrics: EvaluationMetrics = Field(json_schema_extra={"mlflow_type": "metrics"})
