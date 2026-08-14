import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.retriever import retrieve_with_trace


def matches(chunk, expected):
    metadata = chunk.metadata
    return all(
        (
            key == "act" and metadata.get("act_short_name") == value
            or key == "section" and metadata.get("section_number") == value
            or key == "article" and chunk.article == value
            or key == "document_type" and metadata.get("document_type") == value
        )
        for key, value in expected.items()
    )


def metrics(records):
    expected_total = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    exact_hits = []
    for case, result in records:
        expected = case["expected"]
        expected_total += len(expected)
        hits_at_3 += sum(any(matches(chunk, target) for chunk in result[:3]) for target in expected)
        hits_at_5 += sum(any(matches(chunk, target) for chunk in result[:5]) for target in expected)
        ranks = [
            rank
            for rank, chunk in enumerate(result, 1)
            if any(matches(chunk, target) for target in expected)
        ]
        reciprocal_ranks.append(1 / min(ranks) if ranks else 0)
        if case["kind"] == "exact":
            exact_hits.append(all(any(matches(chunk, target) for chunk in result[:5]) for target in expected))
    return {
        "recall@3": hits_at_3 / expected_total if expected_total else 0,
        "recall@5": hits_at_5 / expected_total if expected_total else 0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0,
        "exact_hit": sum(exact_hits) / len(exact_hits) if exact_hits else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True)
    parser.add_argument("--document-id", action="append", default=[])
    parser.add_argument("--cases", default="evals/retrieval_cases.json")
    parser.add_argument("--kind")
    args = parser.parse_args()
    cases = json.loads(Path(args.cases).read_text())
    records = []
    skipped = 0
    for index, case in enumerate(cases, 1):
        if args.kind and case["kind"] != args.kind:
            continue
        if case.get("requires_upload") and not args.document_id:
            skipped += 1
            print(f"SKIP {index}: upload fixture not supplied")
            continue
        trace = retrieve_with_trace(args.email, args.document_id, case["question"])
        records.append((case, trace.original_vector, trace.final))
        print(f"DONE {index}/{len(cases)}: {case['question']}")
        missing = [
            target for target in case["expected"]
            if not any(matches(chunk, target) for chunk in trace.final[:5])
        ]
        if missing:
            print(f"  V2 MISS: {missing}")

    baseline = metrics([(case, old) for case, old, _ in records])
    v2 = metrics([(case, new) for case, _, new in records])
    print(f"\nEvaluated: {len(records)} | Skipped: {skipped}")
    print(f"{'Metric':<16} {'Vector':>10} {'V2':>10}")
    for key in ("recall@3", "recall@5", "mrr", "exact_hit"):
        print(f"{key:<16} {baseline[key]:>10.3f} {v2[key]:>10.3f}")


if __name__ == "__main__":
    main()
