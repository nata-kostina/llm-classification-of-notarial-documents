import argparse
from pathlib import Path

from src.classification.classification import ClassificationStep, DocumentClassifier
from src.config import get_settings
from src.pipeline.base import Pipeline
from src.providers.openai.openai_client import build_openai_client
from src.providers.openai.openai_doc_classification import OpenAIDocClassifier
from src.services.rag.static_rag import StaticRagService
from src.tools.loggers import logger
from src.tools.tracker import log_mlflow_results, log_run_info, track_run


def _build_parser():
    parser = argparse.ArgumentParser(description="Classify documents within the provided dataset directory.")
    parser.add_argument(
        "--data-src",
        type=Path,
        required=True,
        help="Path to the file or directory containing data to process.",
    )
    parser.add_argument(
        "--rag-run-id",
        default=None,
        type=str,
        help="Optional RAG run ID to use for classification.",
    )
    parser.add_argument(
        "--log-mlflow",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to log results to MLflow",
    )

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    data_src, rag_run_id, log_mlflow = args.data_src, args.rag_run_id, args.log_mlflow

    settings = get_settings()

    openai_client = build_openai_client(settings)
    classification_provider = OpenAIDocClassifier(
        client=openai_client,
        settings=settings,
    )
    rag_service = StaticRagService(rag_run_id=rag_run_id) if rag_run_id else None

    document_classifier = DocumentClassifier(classification_provider=classification_provider, rag_service=rag_service)

    try:
        pipeline = Pipeline([ClassificationStep(classifier=document_classifier)])

        if log_mlflow:
            params = {
                "test_data": str(data_src),
                "rag_run_id": rag_run_id,
                "model": settings.openai_chat_model,
                "temperature": settings.openai_chat_model_temperature,
            }

            with track_run(experiment_name="Classification", params=params) as run:
                context = pipeline.run(documents_path=data_src)

                log_mlflow_results(context)

                log_run_info(run)
        else:
            context = pipeline.run(documents_path=data_src)

    except Exception as error:
        message = str(error) or error.__class__.__name__
        logger.error(message, exc_info=True)


if __name__ == "__main__":
    main()
