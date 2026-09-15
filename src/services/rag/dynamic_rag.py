from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import Settings, get_settings
from src.database.models.act import Act
from src.providers.models import EmbeddingProvider
from src.services.rag.base import BaseRagService


class DynamicRagService(BaseRagService):
    name = "dynamic"

    def __init__(self, embedder: EmbeddingProvider, settings: Settings | None = None) -> None:
        self._embedder = embedder
        resolved_settings = settings or get_settings()
        self.retrieval_top_k = resolved_settings.retrieval_top_k

    def get_similar_documents(self, batch: list[Path], session: Session | None) -> dict[str, list[Act]]:
        if not session:
            raise ValueError("Database session must be provided.")

        texts = [f.read_text(encoding="utf-8") for f in batch]
        act_ids = [f.resolve().parent.name.lower() for f in batch]

        query_embeddings = self._embedder.embed(texts)

        batch_results: list[list[Act]] = []

        for current_act_id, query_embedding in zip(act_ids, query_embeddings, strict=True):
            similar_acts = list(
                session.scalars(
                    select(Act)
                    .where(Act.act_id != current_act_id)
                    .order_by(Act.embedding.cosine_distance(query_embedding))
                    .limit(self.retrieval_top_k)
                ).all()
            )
            batch_results.append(similar_acts)

        result: dict[str, list[Act]] = {}
        for source_act_id, similar_acts in zip(act_ids, batch_results, strict=True):
            result[source_act_id] = similar_acts

        return result

    def get_similar_documents_ids(self, batch: list[Path], session: Session | None) -> dict[str, list[str]]:
        result = self.get_similar_documents(batch, session)

        return {act_id: [similar_act.act_id for similar_act in similar_acts] for act_id, similar_acts in result.items()}
