from pathlib import Path

from rich import print

from src.config import get_settings
from src.pipeline.base import Pipeline
from src.providers.openai.openai_client import build_openai_client
from src.providers.openai.openai_doc_embeddings import OpenAIDocEmbedder
from src.retrieval.retrieval import DocumentRetriever, RetrievalStep
from src.services.rag.dynamic_rag import DynamicRagService

DOCUMENTS_PATH = Path(r".\corpus_v2\test\GAR")


def main():
    settings = get_settings()

    openai_client = build_openai_client(settings)
    embeddings_provider = OpenAIDocEmbedder(client=openai_client, settings=settings)
    rag_service = DynamicRagService(embeddings_provider, settings)

    document_retriever = DocumentRetriever(
        embedding_provider=embeddings_provider, rag_service=rag_service
    )

    pipeline = Pipeline([RetrievalStep(document_retriever)])
    context = pipeline.run(documents_path=DOCUMENTS_PATH)
    print(context.model_dump(mode="json"))


if __name__ == "__main__":
    main()
