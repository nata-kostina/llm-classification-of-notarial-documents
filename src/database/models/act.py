import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import UUID, Enum, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.database.models.base import Base, TimestampMixin
from src.database.models.constants import OPENAI_EMBEDDING_DIMENSIONS
from src.database.models.label import LabelEnum


class Act(Base, TimestampMixin):
    __tablename__ = "acts"
    __table_args__ = (
        Index("ix_acts_act_id", "act_id"),
        Index(
            "ix_acts_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    act_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    label: Mapped[LabelEnum] = mapped_column(Enum(LabelEnum, name="act_label", native_enum=False), nullable=False)

    text: Mapped[str] = mapped_column(Text, nullable=False)

    embedding: Mapped[list[float]] = mapped_column(Vector(OPENAI_EMBEDDING_DIMENSIONS), nullable=False)
