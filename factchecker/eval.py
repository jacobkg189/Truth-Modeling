"""Measure accuracy, coverage and speed on labelled claims.

Input is JSON Lines with a ``claim`` and a ``label`` per line. Labels may use
this project's names or FEVER's (SUPPORTS / REFUTES / NOT ENOUGH INFO), so a
FEVER dev split can be evaluated directly:

    factcheck-eval data/sample_claims.jsonl
    factcheck-eval fever_dev.jsonl --limit 500 --out results.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from .cli import claim_to_dict
from .config import Config, load_dotenv
from .pipeline import ClaimResult, FactChecker
from .verdict import Label

_LABEL_ALIASES = {
    "SUPPORTS": Label.SUPPORTED,
    "SUPPORTED": Label.SUPPORTED,
    "TRUE": Label.SUPPORTED,
    "REFUTES": Label.REFUTED,
    "REFUTED": Label.REFUTED,
    "FALSE": Label.REFUTED,
    "NOT ENOUGH INFO": Label.NOT_ENOUGH_EVIDENCE,
    "NOT_ENOUGH_INFO": Label.NOT_ENOUGH_EVIDENCE,
    "NOT_ENOUGH_EVIDENCE": Label.NOT_ENOUGH_EVIDENCE,
    "NEI": Label.NOT_ENOUGH_EVIDENCE,
}
_GOLD_LABELS = [Label.SUPPORTED, Label.REFUTED, Label.NOT_ENOUGH_EVIDENCE]


def load_examples(path: str | Path, limit: int | None = None) -> list[tuple[str, Label]]:
    examples = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            label = _LABEL_ALIASES.get(str(row["label"]).strip().upper())
            if label is None:
                raise ValueError(f"{path}:{line_number}: unknown label {row['label']!r}")
            examples.append((row["claim"], label))
            if limit and len(examples) >= limit:
                break
    return examples


def summarize(golds: list[Label], results: list[ClaimResult]) -> dict:
    predictions = [r.verdict.label for r in results]
    n = len(golds)
    correct = [g == p for g, p in zip(golds, predictions)]
    auto = [not r.verdict.needs_review for r in results]
    auto_correct = [c for c, a in zip(correct, auto) if a]

    per_class = {}
    f1s = []
    for label in _GOLD_LABELS:
        tp = sum(g == label and p == label for g, p in zip(golds, predictions))
        fp = sum(g != label and p == label for g, p in zip(golds, predictions))
        fn = sum(g == label and p != label for g, p in zip(golds, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if tp + fn:
            f1s.append(f1)
        per_class[label.value] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    decided = [(g, p) for g, p in zip(golds, predictions) if g is not Label.NOT_ENOUGH_EVIDENCE]
    latencies = sorted(r.seconds for r in results)
    return {
        "examples": n,
        "accuracy": sum(correct) / n if n else 0.0,
        "macro_f1": statistics.fmean(f1s) if f1s else 0.0,
        # Accuracy on claims whose gold label is SUPPORTED or REFUTED. Benchmark "not enough
        # info" labels are often stale (the encyclopedia has grown since labelling).
        "accuracy_supported_refuted": (
            sum(g == p for g, p in decided) / len(decided) if decided else 0.0
        ),
        # The number that matters for automation: how often the verdicts that were NOT
        # flagged for review are right, and what share of claims that covers.
        "auto_accepted_share": sum(auto) / n if n else 0.0,
        "auto_accepted_accuracy": sum(auto_correct) / len(auto_correct) if auto_correct else 0.0,
        "per_class": per_class,
        "confusion": {
            g.value: dict(Counter(p.value for gg, p in zip(golds, predictions) if gg == g))
            for g in _GOLD_LABELS
        },
        "errors": sum(r.error is not None for r in results),
        "latency_p50_s": latencies[len(latencies) // 2] if latencies else 0.0,
        "latency_p95_s": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0,
    }


async def evaluate(
    examples: list[tuple[str, Label]], checker: FactChecker, concurrency: int = 8
) -> list[ClaimResult]:
    limit = asyncio.Semaphore(concurrency)

    async def one(claim: str) -> ClaimResult:
        async with limit:
            return await checker.check_claim(claim)

    return list(await asyncio.gather(*(one(claim) for claim, _ in examples)))


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="factcheck-eval", description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", help="JSON Lines file with 'claim' and 'label' fields.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N examples.")
    parser.add_argument("--concurrency", type=int, default=8, help="Claims checked at once.")
    parser.add_argument("--source", choices=["wikipedia", "local"], default="wikipedia")
    parser.add_argument("--corpus", help="Directory of .txt/.md files for --source local.")
    parser.add_argument("--model", help="Jev model ID or alias (default: jev-latest).")
    parser.add_argument("--out", help="Write one JSON line per claim with prediction and evidence.")
    args = parser.parse_args(argv)
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("Set TYPESAFE_API_KEY (in the environment or a .env file).", file=sys.stderr)
        return 2

    examples = load_examples(args.dataset, args.limit)

    async def run() -> tuple[list[ClaimResult], FactChecker]:
        checker = FactChecker.create(source=args.source, corpus_dir=args.corpus, model=args.model, config=Config())
        async with checker:
            return await evaluate(examples, checker, args.concurrency), checker

    started = time.perf_counter()
    results, checker = asyncio.run(run())
    wall = time.perf_counter() - started
    golds = [label for _, label in examples]
    summary = summarize(golds, results)
    usage = checker.judge.usage
    summary.update(
        wall_seconds=round(wall, 2),
        claims_per_second=round(len(examples) / wall, 2) if wall else None,
        jev_requests=usage.requests,
        input_tokens=usage.input_tokens,
        estimated_cost_usd=round(usage.cost_usd, 5),
        models=sorted(usage.models),
    )

    if args.out:
        with open(args.out, "w") as handle:
            for (claim, gold), result in zip(examples, results):
                row = claim_to_dict(result)
                row["gold"] = gold.value
                row["correct"] = gold == result.verdict.label
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for (claim, gold), result in zip(examples, results):
        mark = "✓" if gold == result.verdict.label else "✗"
        review = " (review)" if result.verdict.needs_review else ""
        print(f"{mark} {gold.value:<20} → {result.verdict.label.value:<20} {result.verdict.confidence:.2f}{review}  {claim}")
    print()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
