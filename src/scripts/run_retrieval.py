import argparse
from pathlib import Path

from src.config import get_settings
from src.pipeline.base import Pipeline
from src.providers.openai.openai_client import build_openai_client
from src.providers.openai.openai_doc_embeddings import OpenAIDocEmbedder
from src.retrieval.retrieval import DocumentRetriever, RetrievalStep
from src.services.rag.dynamic_rag import DynamicRagService
from src.tools.loggers import logger
from src.tools.tracker import log_mlflow_results, log_run_info, track_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--data-src",
        type=Path,
        required=True,
        help="Path to the file or directory containing data to process.",
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
    data_src, log_mlflow = args.data_src, args.log_mlflow

    settings = get_settings()

    openai_client = build_openai_client(settings)
    embeddings_provider = OpenAIDocEmbedder(client=openai_client, settings=settings)
    rag_service = DynamicRagService(embeddings_provider, settings)

    document_retriever = DocumentRetriever(embedding_provider=embeddings_provider, rag_service=rag_service)
    try:
        pipeline = Pipeline([RetrievalStep(document_retriever)])

        if log_mlflow:
            params = {
                "test_data": data_src,
                "k": settings.retrieval_top_k,
                "model": settings.openai_embedding_model,
                "dimensions": settings.openai_embedding_dimensions,
            }

            with track_run(experiment_name="Retrieval", params=params) as run:
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
