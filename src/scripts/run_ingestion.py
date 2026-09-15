import argparse
from pathlib import Path

from src.config import get_settings
from src.ingestion.ingestion import DocumentIngestor, IngestionStep
from src.pipeline.base import Pipeline
from src.providers.openai.openai_client import build_openai_client
from src.providers.openai.openai_doc_embeddings import OpenAIDocEmbedder
from src.tools.loggers import logger
from src.tools.tracker import log_mlflow_results, log_run_info, track_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Indexing notarial acts.")
    parser.add_argument(
        "--data-src",
        type=Path,
        required=True,
        help="Path to the file or directory containing data to ingest.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip documents that already were embedded",
    )
    parser.add_argument(
        "--log-mlflow",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to log results to MLflow",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    data_src, log_mlflow = args.data_src, args.log_mlflow

    settings = get_settings()

    openai_client = build_openai_client(settings)
    embedding_provider = OpenAIDocEmbedder(client=openai_client, settings=settings)

    document_ingestor = DocumentIngestor(embedding_provider=embedding_provider, skip_existing=args.skip_existing)

    try:
        pipeline = Pipeline([IngestionStep(document_ingestor)])

        if log_mlflow:
            with track_run(experiment_name="Ingestion", params={}) as run:
                context = pipeline.run(documents_path=data_src)

                log_mlflow_results(context)

                log_run_info(run)
        else:
            context = pipeline.run(documents_path=data_src)

    except Exception as error:
        message = str(error) or error.__class__.__name__
        logger.exception(message, exc_info=False)


if __name__ == "__main__":
    main()
