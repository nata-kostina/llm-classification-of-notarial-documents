from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from tqdm import tqdm

from src.config import Settings, get_settings
from src.database.models.act import Act
from src.database.models.label import LabelEnum
from src.database.session import get_session
from src.providers.models import EmbeddingProvider
from src.retrieval.models import (
    DocumentRetrievalResult,
    PerDocumentRetrievalResult,
)
from src.retrieval.retrieval_evaluation import evaluate_retrieval
from src.services.rag.dynamic_rag import DynamicRagService
from src.tools.chunked import chunked
from src.tools.file_utils import get_files_to_process
from src.tools.loggers import logger

if TYPE_CHECKING:
    from src.pipeline.base import PipelineContext


class DocumentRetriever:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        rag_service: DynamicRagService,
        settings: Settings | None = None,
    ):
        resolved_settings = settings or get_settings()
        self.retrieval_top_k = resolved_settings.retrieval_top_k
        self._embedding_provider = embedding_provider
        self._rag_service = rag_service

    def run(self, *, documents_path: Path, batch_size: int = 16) -> DocumentRetrievalResult:
        files = get_files_to_process(documents_path)

        retrieved_docs = []

        with get_session() as session:
            counts_by_label = dict(session.execute(select(Act.label, func.count(Act.id)).group_by(Act.label)).tuples().all())

        batches = list(chunked(files, batch_size))
        pbar = tqdm(
            enumerate(batches, start=1),
            total=len(batches),
            desc="Retrieving document",
            unit="Batch",
        )

        for batch_idx, batch in pbar:
            try:
                with get_session() as session:
                    batch_similar_documents = self._rag_service.get_similar_documents(batch, session)

                    for act in batch:
                        act_dir = act.resolve().parent
                        act_id = act_dir.name
                        target_label_str = act_dir.parent.name.lower()
                        target_label_enum = LabelEnum(target_label_str)

                        similar_acts = batch_similar_documents.get(act_id, [])

                        item = PerDocumentRetrievalResult(
                            source_doc_id=act_id,
                            target_label=target_label_str,
                            retrieved_labels=[d.label for d in similar_acts],
                            retrieved_doc_ids=[d.act_id for d in similar_acts],
                            total_relevant=counts_by_label.get(target_label_enum, 0),
                        )

                        retrieved_docs.append(item)
            except ConnectionError, OperationalError:
                raise
            except Exception as e:
                logger.error(
                    f"Failed to process retrieval batch #{batch_idx}: {e}",
                    exc_info=True,
                )
                continue

        metrics = evaluate_retrieval(retrieved_docs, k=self.retrieval_top_k)

        return DocumentRetrievalResult(retrieved_docs=retrieved_docs, metrics=metrics)


class RetrievalStep:
    name = "retrieval"

    def __init__(self, document_retriever: DocumentRetriever) -> None:
        self._retriever = document_retriever

    def run(self, ctx: PipelineContext) -> PipelineContext:

        retrieval = self._retriever.run(documents_path=ctx.documents_path)

        return ctx.model_copy(update={"retrieval": retrieval})
