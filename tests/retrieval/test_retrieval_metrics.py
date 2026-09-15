import pytest

from src.retrieval.retrieval_metrics import average_precision_at_k, hit_rate_at_k, precision_at_k, reciprocal_rank


def test_precision_at_k_basic_cases():
    labels = ["A", "A", "B", "A", "C"]
    assert precision_at_k(labels, "A", 3) == 2 / 3
    assert precision_at_k(labels, "B", 3) == 1 / 3
    assert precision_at_k(labels, "C", 3) == 0.0


def test_precision_at_k_perfect_and_zero_score():
    labels = ["A", "A", "A"]
    assert precision_at_k(labels, "A", 3) == 1.0
    assert precision_at_k(labels, "B", 3) == 0.0


def test_precision_at_k_empty_input():
    assert precision_at_k([], "A", 5) == 0.0


def test_precision_at_k_invalid_k():
    labels = ["A", "B"]
    assert precision_at_k(labels, "A", 0) == 0.0
    assert precision_at_k(labels, "A", -1) == 0.0


def test_precision_at_k_when_k_exceeds_list_length():
    labels = ["A", "A"]
    assert precision_at_k(labels, "A", 5) == 2 / 5


def test_precision_at_k_case_sensitivity():
    labels = ["A", "a"]
    assert precision_at_k(labels, "a", 2) == 0.5


def test_precision_at_k_floating_point_precision():
    labels = ["A", "B", "A"]
    assert pytest.approx(precision_at_k(labels, "A", 3), rel=1e-9) == 2 / 3
    

def test_hit_rate_at_k_hit():
    labels = ["A", "B", "C"]
    assert hit_rate_at_k(labels, "A", 1) == 1.0
    assert hit_rate_at_k(labels, "B", 2) == 1.0


def test_hit_rate_at_k_miss():
    labels = ["A", "B", "C"]
    assert hit_rate_at_k(labels, "C", 2) == 0.0
    assert hit_rate_at_k(labels, "D", 3) == 0.0


def test_hit_rate_at_k_empty_and_invalid_k():
    labels = ["A", "B"]
    assert hit_rate_at_k([], "A", 3) == 0.0
    assert hit_rate_at_k(labels, "A", 0) == 0.0
    assert hit_rate_at_k(labels, "A", -1) == 0.0


def test_reciprocal_rank_ranks():
    labels = ["A", "B", "C"]
    assert reciprocal_rank(labels, "A", 3) == 1.0
    assert reciprocal_rank(labels, "B", 3) == 0.5
    assert reciprocal_rank(labels, "C", 3) == 1 / 3


def test_reciprocal_rank_not_found_or_outside_k():
    labels = ["A", "B", "C"]
    assert reciprocal_rank(labels, "D", 3) == 0.0
    assert reciprocal_rank(labels, "C", 2) == 0.0


def test_reciprocal_rank_empty_and_invalid_k():
    labels = ["A", "B"]
    assert reciprocal_rank([], "A", 3) == 0.0
    assert reciprocal_rank(labels, "A", 0) == 0.0
    assert reciprocal_rank(labels, "A", -1) == 0.0


def test_average_precision_at_k_perfect():
    labels = ["A", "A", "B"]
    assert average_precision_at_k(labels, "A", 2, 2) == 1.0


def test_average_precision_at_k_partial_hit():
    labels = ["B", "A", "A"]
    # i=1: 1/2, i=2: 2/3. sum = 0.5 + 0.6666... max_possible = min(3, 2) = 2
    expected = (1 / 2 + 2 / 3) / 2
    assert pytest.approx(average_precision_at_k(labels, "A", 3, 2), rel=1e-9) == expected


def test_average_precision_at_k_zero_hits():
    labels = ["B", "C", "D"]
    assert average_precision_at_k(labels, "A", 3, 2) == 0.0


def test_average_precision_at_k_edge_cases():
    labels = ["A", "B"]
    assert average_precision_at_k([], "A", 2, 2) == 0.0
    assert average_precision_at_k(labels, "A", 0, 2) == 0.0
    assert average_precision_at_k(labels, "A", 2, 0) == 0.0
    assert average_precision_at_k(labels, "A", 2, -1) == 0.0


def test_average_precision_at_k_total_relevant_exceeds_k():
    labels = ["A", "B", "A"]
    # i=0: 1/1, i=2: 2/3. sum = 1.0 + 0.6666... max_possible = min(3, 5) = 3
    expected = (1.0 + 2 / 3) / 3
    assert pytest.approx(average_precision_at_k(labels, "A", 3, 5), rel=1e-9) == expected