from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from tqdm import tqdm

from src.database.models.act import Act
from src.database.models.label import LabelEnum
from src.database.session import get_session
from src.ingestion.models import DocumentCount, DocumentIngestionResult
from src.pipeline.base import PipelineContext
from src.providers.models import EmbeddingProvider
from src.tools.chunked import chunked
from src.tools.file_utils import get_files_to_process
from src.tools.loggers import logger


class DocumentIngestor:
    def __init__(self, embedding_provider: EmbeddingProvider, skip_existing: bool) -> None:
        self._embedding_provider = embedding_provider
        self.skip_existing = skip_existing

    def run(self, *, documents_path: Path, batch_size: int = 16) -> DocumentIngestionResult:
        files = get_files_to_process(documents_path)
        document_count = DocumentCount()

        batches = list(chunked(files, batch_size))
        pbar = tqdm(enumerate(batches, start=1), total=len(batches), desc="Ingesting documents", unit="batch")

        for batch_idx, batch in pbar:
            try:
                with get_session() as session:
                    skipped_in_batch = 0

                    if self.skip_existing:
                        batch_act_ids = [f.resolve().parent.name for f in batch]
                        existing_act_ids = set(session.scalars(select(Act.act_id).where(Act.act_id.in_(batch_act_ids))).all())
                        filtered_batch = [f for f in batch if f.resolve().parent.name not in existing_act_ids]
                        skipped_in_batch = len(batch) - len(filtered_batch)
                        batch = filtered_batch

                        if skipped_in_batch > 0:
                            document_count = document_count.model_copy(
                                update={
                                    "processed": document_count.processed + skipped_in_batch,
                                    "skipped": document_count.skipped + skipped_in_batch,
                                }
                            )
                    if not batch:
                        continue

                    self.ingest_batch(session, batch)
                    session.commit()

                    document_count = document_count.model_copy(
                        update={
                            "processed": document_count.processed + len(batch),
                            "written": document_count.written + len(batch),
                        }
                    )
            except ConnectionError, SQLAlchemyError:
                raise

            except Exception as e:
                logger.error(f"Failed to ingest batch #{batch_idx} ({len(batch)} files): {e}", exc_info=True)
                document_count = document_count.model_copy(
                    update={
                        "processed": document_count.processed + len(batch),
                        "failed": document_count.failed + len(batch),
                    }
                )

        return DocumentIngestionResult(document_count=document_count)

    def ingest_batch(self, session: Session, batch: list[Path]) -> None:
        if not batch:
            return

        texts: list[str] = []
        records_metadata: list[dict[str, Any]] = []

        for act in batch:
            text = act.read_text(encoding="utf-8")
            act_dir = act.resolve().parent
            act_id = act_dir.name
            label = LabelEnum(act_dir.parent.name.lower())

            texts.append(text)
            records_metadata.append({"act_id": act_id, "label": label, "text": text})

        embeddings = self._embedding_provider.embed(texts)

        values_to_insert = [{**meta, "embedding": emb} for meta, emb in zip(records_metadata, embeddings, strict=True)]

        stmt = insert(Act)

        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=["act_id"], set_={"embedding": stmt.excluded.embedding, "label": stmt.excluded.label, "text": stmt.excluded.text}
        )

        session.execute(upsert_stmt, values_to_insert)


class IngestionStep:
    name = "ingestion"

    def __init__(self, ingestor: DocumentIngestor) -> None:
        self._ingestor = ingestor

    def run(self, ctx: PipelineContext) -> PipelineContext:
        ingestion = self._ingestor.run(documents_path=ctx.documents_path)

        return ctx.model_copy(update={"ingestion": ingestion})
