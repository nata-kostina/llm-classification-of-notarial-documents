from src.retrieval.models import EvaluationMetrics, PerDocumentRetrievalResult
from src.retrieval.retrieval_metrics import (
    mean_average_precision,
    mean_hit_rate_at_k,
    mean_precision_at_k,
    mean_reciprocal_rank,
)


def evaluate_retrieval(dataset: list[PerDocumentRetrievalResult], *, k: int) -> EvaluationMetrics:
    all_retrieved_labels = [d.retrieved_labels for d in dataset]
    all_target_labels = [d.target_label for d in dataset]
    all_totals = [d.total_relevant for d in dataset]

    mean_precision = mean_precision_at_k(all_retrieved_labels, all_target_labels, k)
    mean_hit_rate = mean_hit_rate_at_k(all_retrieved_labels, all_target_labels, k)
    mean_rr = mean_reciprocal_rank(all_retrieved_labels, all_target_labels, k)
    mean_ap = mean_average_precision(all_retrieved_labels, all_target_labels, k, all_totals)

    return EvaluationMetrics(
        k=k,
        mean_precision=mean_precision,
        mean_hit_rate=mean_hit_rate,
        mean_reciprocal_rank=mean_rr,
        mean_average_precision=mean_ap,
        total_source_docs=len(dataset),
    )
