from abc import ABC, abstractmethod
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models.act import Act
from src.database.session import get_session
from src.tools.loggers import logger


class BaseRagService(ABC):
    name: str

    @abstractmethod
    def get_similar_documents_ids(self, batch: list[Path], session: Session | None = None) -> dict[str, list[str]]: ...

    def get_batch_context(self, batch: list[Path]) -> list[str]:
        batch_similar_results = self.get_similar_documents_ids(batch)

        all_doc_ids = {doc_id for doc_ids in batch_similar_results.values() if doc_ids for doc_id in doc_ids}

        with get_session() as session:
            stmt = select(Act).where(Act.act_id.in_(all_doc_ids))
            db_docs = session.scalars(stmt).all()

            db_docs_map = {doc.act_id: doc for doc in db_docs}

        batch_contexts: list[str] = []

        for file in batch:
            act_id = file.parent.name
            similar_doc_ids = batch_similar_results.get(act_id, [])

            db_similar_docs = [db_docs_map[doc_id] for doc_id in similar_doc_ids if doc_id in db_docs_map]
            if not db_similar_docs:
                batch_contexts.append("")
                logger.warning(f"Retrieved documents missing for doc_id='{act_id}'. RAG prompt generation skipped for this document.")
                continue

            context_blocks = []
            for i, doc in enumerate(db_similar_docs, start=1):
                label = getattr(doc, "label", "")
                text_content = doc.text.strip() if doc.text else ""
                block = f"### Exemple {i}\n**Catégorie :** `{label}`\n\n```text\n{text_content}\n```"
                context_blocks.append(block)

            full_file_context = "\n\n".join(context_blocks)
            batch_contexts.append(full_file_context)

        return batch_contexts
