from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from src.classification.models import DocumentClassificationResult
from src.evaluation.models import EvaluationResult
from src.ingestion.models import DocumentIngestionResult
from src.retrieval.models import DocumentRetrievalResult
from src.tools.loggers import logger


class PipelineContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents_path: Path

    ingestion: DocumentIngestionResult | None = None
    retrieval: DocumentRetrievalResult | None = None
    classification: DocumentClassificationResult | None = None
    evaluation: EvaluationResult | None = None


class PipelineStep(Protocol):
    name: str

    def run(self, ctx: PipelineContext) -> PipelineContext: ...


class Pipeline:
    def __init__(self, steps: Sequence[PipelineStep]) -> None:
        self.steps = tuple(steps)

    def run(self, documents_path: Path) -> PipelineContext:
        logger.info(f"Pipeline started for {documents_path.name} ({len(self.steps)} steps)")
        ctx = PipelineContext(documents_path=documents_path)
        for index, step in enumerate(self.steps, start=1):
            logger.info(f"[{index}/{len(self.steps)}] Starting step: {step.name}")
            ctx = step.run(ctx)
            logger.info(f"[{index}/{len(self.steps)}] Finished step: {step.name}")
        logger.info(f"Pipeline completed for {documents_path.name}")
        return ctx
