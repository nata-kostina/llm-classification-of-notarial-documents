import json
from collections.abc import Iterable
from time import perf_counter

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError

from src.classification.models import (
    DocumentClassificationError,
    LlmDocumentClassification,
    LlmDocumentClassificationMetrics,
    LlmDocumentClassificationResult,
)
from src.config import Settings, get_settings


class OpenAIDocClassifier:
    name = "openai_classifier"

    def __init__(self, *, client: OpenAI, settings: Settings | None = None):
        self._client = client

        resolved_settings = settings or get_settings()
        self.model = resolved_settings.openai_chat_model
        self.temperature = resolved_settings.openai_chat_model_temperature

    def classify(self, prompt: Iterable[ChatCompletionMessageParam]) -> LlmDocumentClassificationResult:

        schema = LlmDocumentClassification.model_json_schema()

        try:
            start = perf_counter()

            response = self._client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=self.temperature,
                seed=42,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "document_classification", "strict": True, "schema": schema},
                },
            )

            latency = perf_counter() - start

            usage = response.usage
            choice = response.choices[0]

            metrics = LlmDocumentClassificationMetrics(
                latency=latency,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                finish_reason=choice.finish_reason if choice else None,
            )

        except Exception as error:
            raise DocumentClassificationError("OpenAI document classification failed.") from error

        message = response.choices[0].message.content
        if not message:
            raise DocumentClassificationError("OpenAI returned empty response.")

        try:
            parsed_data = LlmDocumentClassification.model_validate_json(message)
            return LlmDocumentClassificationResult(data=parsed_data, metrics=metrics)
        except (json.JSONDecodeError, ValidationError, TypeError) as error:
            raise DocumentClassificationError("OpenAI did not return valid structured document classification output.") from error
