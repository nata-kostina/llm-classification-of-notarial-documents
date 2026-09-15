from openai import OpenAI

from src.config import Settings, get_settings
from src.ingestion.models import DocumentEmbeddingError


class OpenAIDocEmbedder:
    name = "openai_embedder"

    def __init__(self, *, client: OpenAI, settings: Settings | None = None):
        self._client = client

        resolved_settings = settings or get_settings()

        self.model = resolved_settings.openai_embedding_model
        self.temperature = resolved_settings.openai_chat_model_temperature
        self.dimensions = resolved_settings.openai_embedding_dimensions

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return [[]]

        try:
            response = self._client.embeddings.create(input=texts, model=self.model, dimensions=self.dimensions)

            return [list(item.embedding) for item in response.data]

        except Exception as error:
            raise DocumentEmbeddingError("OpenAI document embeddings failed.") from error
