import re
from typing import Annotated

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from src.database.models.label import LabelEnum

RuleString = Annotated[str, StringConstraints(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")]


class LlmDocumentClassification(BaseModel):
    label: LabelEnum = Field(description="La qualification juridique exacte du bien immobilier (la catégorie).")
    rules: list[RuleString] = Field(
        description=(
            "La liste des numéros de règles ayant permis de classer le document. "
            "IMPORTANT: Écrire UNIQUEMENT les chiffres séparés par des points (ex: '1.1.1'). "
            "NE PAS ajouter de mots, de lettres, de préfixes comme 'RULE'."
        ),
        examples=[["1.1.1", "3.2.1"]],
        min_length=1,
        max_length=3,
    )

    @field_validator("rules", mode="before")
    @classmethod
    def clean_rule_strings(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list):
            return value

        cleaned_rules = []
        for item in value:
            if isinstance(item, str):
                match = re.search(r"^[0-9]+\.[0-9]+\.[0-9]+$", item)
                if match:
                    cleaned_rules.append(match.group(0))
                else:
                    cleaned_rules.append(item)
            else:
                cleaned_rules.append(item)
        return cleaned_rules


class BuiltPrompt(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    system_prompt: str
    rag_prompt: str
    user_prompt: str
    messages: list[ChatCompletionMessageParam]


class LlmDocumentClassificationMetrics(BaseModel):
    latency: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str | None


class LlmDocumentClassificationResult(BaseModel):
    data: LlmDocumentClassification
    metrics: LlmDocumentClassificationMetrics


class DocumentClassificationResult(BaseModel):
    act_id: str
    file_path: str
    label_true: str

    system_prompt: str
    rag_prompt: str
    user_prompt: str

    data: LlmDocumentClassification | None
    metrics: LlmDocumentClassificationMetrics | None

    error: str | None = None


class ClassificationPipelineResults(BaseModel):
    records: list[DocumentClassificationResult] = Field(
        default_factory=list,
        json_schema_extra={"mlflow_type": "csv"},
    )


class DocumentClassificationError(RuntimeError):
    pass
