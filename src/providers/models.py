from collections.abc import Iterable
from typing import Protocol

from openai.types.chat import ChatCompletionMessageParam

from src.classification.models import LlmDocumentClassificationResult


class EmbeddingProvider(Protocol):
    name: str

    def embed(self, texts: str | list[str]) -> list[list[float]]: ...


class ClassificationProvider(Protocol):
    name: str

    def classify(self, prompt: Iterable[ChatCompletionMessageParam]) -> LlmDocumentClassificationResult: ...
