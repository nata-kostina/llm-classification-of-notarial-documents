from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.classification.classification import DocumentClassifier
from src.classification.models import (
    LlmDocumentClassification,
    LlmDocumentClassificationMetrics,
    LlmDocumentClassificationResult,
)
from src.database.models.label import LabelEnum

any_label = list(LabelEnum)[0]

@pytest.fixture
def mock_provider() -> MagicMock:
    provider = MagicMock()
    
    provider.classify.return_value = LlmDocumentClassificationResult(
        data=LlmDocumentClassification(
            label=any_label,
            rules=["1.1.1"]
        ),
        metrics=LlmDocumentClassificationMetrics(
            latency=0.5,
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            finish_reason="stop"
        )
    )
    return provider


def test_run_success_without_rag(mocker, mock_provider: MagicMock, tmp_path: Path):
    fake_files = [
        Path("dataset/category_a/act_001/doc_1.txt"),
        Path("dataset/category_a/act_002/doc_2.txt"),
    ]
    mocker.patch(
        "src.classification.classification.get_files_to_process", 
        return_value=fake_files
    )

    mocker.patch("src.classification.classification.JsonlLogger")

    mock_built_prompt = MagicMock()
    mock_built_prompt.system_prompt = "System Prompt"
    mock_built_prompt.rag_prompt = ""
    mock_built_prompt.user_prompt = "User Prompt"
    mock_built_prompt.messages = [
        {"role": "system", "content": "System Prompt. Some rules."}, 
        {"role": "user", "content": "User Prompt."}
    ]

    mocker.patch(
        "src.classification.classification.PromptBuilder.build_for_file",
        return_value=mock_built_prompt
    )

    classifier = DocumentClassifier(
        classification_provider=mock_provider,
        rag_service=None
    )

    results = classifier.run(documents_path=tmp_path, batch_size=2)

    assert len(results.records) == 2

    assert mock_provider.classify.call_count == 2

    record_1 = results.records[0]
    assert record_1.act_id == "act_001"
    assert record_1.label_true == "category_a"
    assert record_1.error is None
    assert record_1.data is not None
    assert record_1.data.label == any_label
    assert record_1.data.rules == ["1.1.1"]
    assert record_1.metrics is not None
    assert record_1.metrics.latency == 0.5
    assert record_1.metrics.prompt_tokens == 100
    assert record_1.metrics.completion_tokens == 20
    assert record_1.metrics.total_tokens == 120
    assert record_1.metrics.finish_reason == "stop"