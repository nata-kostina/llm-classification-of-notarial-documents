from pathlib import Path

from sqlalchemy.exc import OperationalError
from tqdm import tqdm

from src.classification.models import ClassificationPipelineResults, DocumentClassificationResult
from src.classification.prompt_builder import PromptBuilder
from src.pipeline.base import PipelineContext
from src.providers.models import ClassificationProvider
from src.services.rag.base import BaseRagService
from src.tools.chunked import chunked
from src.tools.file_utils import get_files_to_process
from src.tools.loggers import JsonlLogger, logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DocumentClassifier:
    def __init__(self, classification_provider: ClassificationProvider, rag_service: BaseRagService | None) -> None:
        self._classification_provider = classification_provider
        self._rag_service = rag_service

    def run(self, *, documents_path: Path, batch_size: int = 10) -> ClassificationPipelineResults:
        files = get_files_to_process(documents_path)

        records: list[DocumentClassificationResult] = []

        prompt_builder = PromptBuilder(PROJECT_ROOT / "prompts")

        batches = list(chunked(files, batch_size))
        pbar = tqdm(enumerate(batches, start=1), total=len(batches), desc="Classifying documents", unit="batch")

        with JsonlLogger() as json_logger:
            for batch_idx, batch in pbar:
                try:
                    if self._rag_service is not None:
                        batch_contexts = self._rag_service.get_batch_context(batch)
                    else:
                        batch_contexts = [""] * len(batch)
                except ConnectionError, OperationalError:
                    raise
                except Exception as e:
                    logger.error(f"Failed to fetch RAG context for batch #{batch_idx}: {e}", exc_info=True)
                    continue

                for file, rag_context in zip(batch, batch_contexts, strict=True):
                    act_id = file.parent.name
                    label_true = file.parent.parent.name.lower()

                    try:
                        built_prompt = prompt_builder.build_for_file(file, rag_context)

                        result = self._classification_provider.classify(built_prompt.messages)

                        record = DocumentClassificationResult(
                            act_id=act_id,
                            file_path=str(file),
                            label_true=label_true,
                            system_prompt=built_prompt.system_prompt,
                            rag_prompt=built_prompt.rag_prompt,
                            user_prompt=built_prompt.user_prompt,
                            data=result.data,
                            metrics=result.metrics,
                        )
                    except ConnectionError, OperationalError:
                        raise

                    except Exception as error:
                        logger.error(f"Failed to classify file '{file.name}' in batch #{batch_idx}: {error}", exc_info=True)
                        record = DocumentClassificationResult(
                            act_id=file.parent.name,
                            file_path=str(file),
                            label_true=file.parent.parent.name.lower(),
                            system_prompt="",
                            rag_prompt="",
                            user_prompt="",
                            data=None,
                            metrics=None,
                            error=f"Unexpected error in batch #{batch_idx}: {error}",
                        )

                    json_logger.log(record)
                    records.append(record)

        return ClassificationPipelineResults(records=records)


class ClassificationStep:
    name = "classification"

    def __init__(self, classifier: DocumentClassifier) -> None:
        self._classifier = classifier

    def run(self, ctx: PipelineContext) -> PipelineContext:
        classification = self._classifier.run(documents_path=ctx.documents_path)

        return ctx.model_copy(update={"classification": classification})
