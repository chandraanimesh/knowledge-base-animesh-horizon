import time
from pathlib import Path

import httpx

from evaluation.data import EVAL_CASES
from evaluation.metrics import calculate_metrics


API_URL = "http://127.0.0.1:8000/chat/ask"

DATA_FILE = Path(
    "C:\\ragDemo\\ragdemo1\\sample_docs\\companies_1000_synthetic.csv"
)


def load_records() -> list[dict[str, str]]:
    import csv

    with DATA_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        return list(
            csv.DictReader(file)
        )


def get_expected_ids(
    records: list[dict[str, str]],
    expected: dict[str, str],
) -> set[str]:

    result: set[str] = set()

    for record in records:

        if (
            "company_id" in expected
            and record["company_id"]
            != expected["company_id"]
        ):
            continue

        if (
            "city" in expected
            and record["location"].strip().lower()
            != expected["city"].strip().lower()
        ):
            continue

        if (
            "country" in expected
            and record["country"].strip().lower()
            != expected["country"].strip().lower()
        ):
            continue

        if (
            "funding_status" in expected
            and record["funding_status"].strip().lower()
            != expected["funding_status"].strip().lower()
        ):
            continue

        if (
            "company_name_contains" in expected
            and expected["company_name_contains"].lower()
            not in record["company_name"].lower()
        ):
            continue

        result.add(
            record["company_id"]
        )

    return result


def extract_company_ids(
    response_json: dict,
) -> list[str]:

    company_ids: list[str] = []

    for chunk in response_json.get(
        "retrieved_chunks",
        [],
    ):
        text = chunk.get("text", "")

        for line in text.splitlines():

            if line.startswith("company_id:"):

                company_id = (
                    line.split(
                        ":",
                        1,
                    )[1]
                    .strip()
                )

                company_ids.append(
                    company_id
                )

                break

    return company_ids


async def evaluate_case(
    client: httpx.AsyncClient,
    records: list[dict[str, str]],
    case: dict,
) -> dict:

    expected_ids = get_expected_ids(
        records,
        case["expected"],
    )

    start = time.perf_counter()

    response = await client.post(
        API_URL,
        json={
            "query": case["query"],
            "top_k": 5,
            "rerank": False,
        },
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    response.raise_for_status()

    response_json = response.json()

    retrieved_ids = extract_company_ids(
        response_json
    )

    metrics = calculate_metrics(
        expected_ids,
        retrieved_ids,
        expect_no_results=case.get(
            "expect_no_results",
            False,
        ),
    )

    return {
        "name": case["name"],
        "query": case["query"],
        "expected_count": len(expected_ids),
        "retrieved_count": len(retrieved_ids),
        "expected_ids": sorted(
            expected_ids,
        ),
        "retrieved_ids": retrieved_ids,
        "recall": metrics.recall,
        "precision": metrics.precision,
        "top_k_accuracy": (
            metrics.top_k_accuracy
        ),
        "no_result_correct": (
            metrics.no_result_correct
        ),
        "latency_ms": round(
            latency_ms,
            2,
        ),
        "answer": response_json.get(
            "answer",
            "",
        ),
    }


async def main():

    records = load_records()

    print(
        f"Loaded {len(records)} records"
    )

    results = []

    async with httpx.AsyncClient(
        timeout=60.0
    ) as client:

        for case in EVAL_CASES:

            print(
                f"\nRunning: {case['name']}"
            )

            result = await evaluate_case(
                client,
                records,
                case,
            )

            results.append(result)

            print(
                f"Recall: "
                f"{result['recall']:.2f}"
            )

            print(
                f"Precision: "
                f"{result['precision']:.2f}"
            )

            print(
                f"Top-K Accuracy: "
                f"{result['top_k_accuracy']:.2f}"
            )

            print(
                f"Latency: "
                f"{result['latency_ms']} ms"
            )

            print(
                "Expected:",
                result["expected_ids"],
            )

            print(
                "Retrieved:",
                result["retrieved_ids"],
            )

    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    total = len(results)

    avg_recall = (
        sum(
            r["recall"]
            for r in results
        )
        / total
    )

    avg_precision = (
        sum(
            r["precision"]
            for r in results
        )
        / total
    )

    avg_top_k = (
        sum(
            r["top_k_accuracy"]
            for r in results
        )
        / total
    )

    avg_latency = (
        sum(
            r["latency_ms"]
            for r in results
        )
        / total
    )

    no_result_cases = [
        r
        for r in results
        if r["no_result_correct"]
        is not None
    ]

    no_result_accuracy = (
        sum(
            r["no_result_correct"]
            for r in no_result_cases
        )
        / len(no_result_cases)
        if no_result_cases
        else None
    )

    print(
        f"Recall@5:       "
        f"{avg_recall:.2%}"
    )

    print(
        f"Precision@5:    "
        f"{avg_precision:.2%}"
    )

    print(
        f"Top-K Accuracy: "
        f"{avg_top_k:.2%}"
    )

    if no_result_accuracy is not None:
        print(
            f"No-result:      "
            f"{no_result_accuracy:.2%}"
        )

    print(
        f"Avg Latency:    "
        f"{avg_latency:.2f} ms"
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())