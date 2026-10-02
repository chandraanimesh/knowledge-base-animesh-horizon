from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    recall: float
    precision: float
    top_k_accuracy: float
    no_result_correct: bool | None = None


def calculate_recall(
    expected_ids: set[str],
    retrieved_ids: list[str],
) -> float:
    if not expected_ids:
        return 1.0 if not retrieved_ids else 0.0

    retrieved_set = set(retrieved_ids)

    relevant_retrieved = (
        expected_ids & retrieved_set
    )

    return (
        len(relevant_retrieved)
        / len(expected_ids)
    )


def calculate_precision(
    expected_ids: set[str],
    retrieved_ids: list[str],
) -> float:
    if not retrieved_ids:
        return 1.0 if not expected_ids else 0.0

    retrieved_set = set(retrieved_ids)

    relevant_retrieved = (
        expected_ids & retrieved_set
    )

    return (
        len(relevant_retrieved)
        / len(retrieved_ids)
    )


def calculate_top_k_accuracy(
    expected_ids: set[str],
    retrieved_ids: list[str],
) -> float:
    if not expected_ids:
        return 1.0 if not retrieved_ids else 0.0

    return float(
        any(
            company_id in expected_ids
            for company_id in retrieved_ids
        )
    )


def calculate_no_result_correctness(
    expect_no_results: bool,
    retrieved_ids: list[str],
) -> bool:
    actual_no_results = not retrieved_ids

    return (
        actual_no_results
        == expect_no_results
    )


def calculate_metrics(
    expected_ids: set[str],
    retrieved_ids: list[str],
    *,
    expect_no_results: bool = False,
) -> RetrievalMetrics:

    recall = calculate_recall(
        expected_ids,
        retrieved_ids,
    )

    precision = calculate_precision(
        expected_ids,
        retrieved_ids,
    )

    top_k_accuracy = calculate_top_k_accuracy(
        expected_ids,
        retrieved_ids,
    )

    no_result_correct = (
        calculate_no_result_correctness(
            expect_no_results,
            retrieved_ids,
        )
        if expect_no_results
        else None
    )

    return RetrievalMetrics(
        recall=recall,
        precision=precision,
        top_k_accuracy=top_k_accuracy,
        no_result_correct=no_result_correct,
    )