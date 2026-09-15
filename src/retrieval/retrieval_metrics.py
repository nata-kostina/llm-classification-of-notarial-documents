# --- Single Query Evaluation Metrics ---


def precision_at_k(retrieved_labels: list[str], target_label: str, k: int) -> float:
    if not retrieved_labels or k <= 0:
        return 0.0

    top_k_labels = retrieved_labels[:k]

    relevant_count = sum(1 for label in top_k_labels if label == target_label)

    return relevant_count / k


def hit_rate_at_k(retrieved_labels: list[str], target_label: str, k: int) -> float:
    if not retrieved_labels or k <= 0:
        return 0.0

    top_k_labels = retrieved_labels[:k]

    return 1.0 if target_label in top_k_labels else 0.0


def reciprocal_rank(retrieved_labels: list[str], target_label: str, k: int) -> float:
    if not retrieved_labels or k <= 0:
        return 0.0

    top_k_labels = retrieved_labels[:k]

    try:
        rank = top_k_labels.index(target_label) + 1
        return 1 / rank
    except ValueError:
        return 0.0


def average_precision_at_k(retrieved_labels: list[str], target_label: str, k: int, total_relevant: int) -> float:
    if not retrieved_labels or k <= 0 or total_relevant <= 0:
        return 0.0

    top_k_labels = retrieved_labels[:k]

    num_relevant = 0
    scores = 0.0

    for i, label in enumerate(top_k_labels):
        if label == target_label:
            num_relevant += 1
            scores += num_relevant / (i + 1)

    if num_relevant == 0:
        return 0.0

    max_possible_relevant = min(k, total_relevant)

    return scores / max_possible_relevant


# --- Global Evaluation Metrics ---


def mean_precision_at_k(all_retrieved_labels: list[list[str]], all_target_labels: list[str], k: int) -> float:
    if not all_retrieved_labels or len(all_retrieved_labels) != len(all_target_labels):
        return 0.0

    total_precision = 0.0

    for retrieved, target in zip(all_retrieved_labels, all_target_labels, strict=False):
        total_precision += precision_at_k(retrieved, target, k)

    return total_precision / len(all_retrieved_labels)


def mean_hit_rate_at_k(all_retrieved_labels: list[list[str]], all_target_labels: list[str], k: int) -> float:
    if not all_retrieved_labels or len(all_retrieved_labels) != len(all_target_labels):
        return 0.0

    total_rate = 0.0

    for retrieved, target in zip(all_retrieved_labels, all_target_labels, strict=False):
        total_rate += hit_rate_at_k(retrieved, target, k)

    return total_rate / len(all_retrieved_labels)


def mean_reciprocal_rank(all_retrieved_labels: list[list[str]], all_target_labels: list[str], k: int) -> float:
    if not all_retrieved_labels or len(all_retrieved_labels) != len(all_target_labels):
        return 0.0

    total_reciprocal_rank = 0.0

    for retrieved, target in zip(all_retrieved_labels, all_target_labels, strict=False):
        total_reciprocal_rank += reciprocal_rank(retrieved, target, k)

    return total_reciprocal_rank / len(all_retrieved_labels)


def mean_average_precision(all_retrieved_labels: list[list[str]], all_target_labels: list[str], k: int, total_relevant: list[int]) -> float:
    if not all_retrieved_labels or len(all_retrieved_labels) != len(all_target_labels):
        return 0.0

    total_average_precision = 0.0

    for retrieved, target, total in zip(all_retrieved_labels, all_target_labels, total_relevant, strict=False):
        total_average_precision += average_precision_at_k(retrieved, target, k, total)

    return total_average_precision / len(all_retrieved_labels)
