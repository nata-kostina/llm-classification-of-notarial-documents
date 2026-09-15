from pathlib import Path

from src.config import get_settings
from src.services.rag.static_rag import StaticRagService

RAG_RUN_ID = "13e607359ae84f1d8a0e090824b9d6b6"
BATCH = [Path(r".\GAR\292773057\designation.txt")]


def main():
    settings = get_settings()

    rag_service = StaticRagService(settings=settings, rag_run_id=RAG_RUN_ID)

    batch_similar_results = rag_service.get_similar_documents_ids(BATCH)

    batch_context = rag_service.get_batch_context(BATCH)

    print(batch_similar_results)
    print(batch_context)


if __name__ == "__main__":
    main()
