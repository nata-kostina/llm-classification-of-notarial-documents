from pydantic import BaseModel, ConfigDict, Field


class DocumentEmbeddingError(RuntimeError):
    pass


class DocumentCount(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    processed: int = 0
    skipped: int = 0
    written: int = 0
    failed: int = 0


class DocumentIngestionResult(BaseModel):
    document_count: DocumentCount = Field(json_schema_extra={"mlflow_type": "dictionary"})
