import json
from pathlib import Path

from jsonschema import ValidationError

from src.config import Settings, get_settings
from src.retrieval.models import PerDocumentRetrievalResult
from src.services.rag.base import BaseRagService
from src.services.rag.models import RagArtifactNotFoundError, RagModelValidationError


class StaticRagService(BaseRagService):
    name = "static"

    def __init__(self, rag_run_id: str, settings: Settings | None = None, artifact_name: str = "retrieved_docs.json"):
        super().__init__()
        self.rag_run_id = rag_run_id
        self.artifact_name = artifact_name

        resolved_settings = settings or get_settings()
        self.mlflow_artifact_store = Path(resolved_settings.mlflow_artifact_store)

        self._cached_results: dict[str, PerDocumentRetrievalResult] | None = None

    def _load_all_results(self) -> dict[str, PerDocumentRetrievalResult]:
        if self._cached_results is not None:
            return self._cached_results

        matches = list(self.mlflow_artifact_store.rglob(f"{self.rag_run_id}/**/{self.artifact_name}"))

        if not matches:
            raise RagArtifactNotFoundError(f"File {self.artifact_name} not found for run ID {self.rag_run_id}")

        artifact_file = matches[0]

        with open(artifact_file, encoding="utf-8") as f:
            rag_context = json.load(f)

        try:
            parsed_list = [PerDocumentRetrievalResult.model_validate(res) for res in rag_context]

            self._cached_results = {item.source_doc_id: item for item in parsed_list}
            return self._cached_results

        except json.JSONDecodeError as exc:
            raise RagModelValidationError(f"Failed to parse JSON in artifact '{self.artifact_name}' for run '{self.rag_run_id}'") from exc

        except ValidationError as exc:
            raise RagModelValidationError(
                f"Schema validation failed for artifact '{self.artifact_name}' (run '{self.rag_run_id}'):\n{exc}"
            ) from exc

    def get_similar_documents_ids(self, batch: list[Path]) -> dict[str, list[str]]:
        all_results = self._load_all_results()

        result = {}
        for file_path in batch:
            act_id = file_path.parent.name
            doc_result = all_results.get(act_id)
            result[act_id] = doc_result.retrieved_doc_ids if doc_result else []

        return result
