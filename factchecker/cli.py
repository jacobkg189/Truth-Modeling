"""Command-line interface: ``factcheck "text to check"``."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import replace
from typing import Any

from .config import Config, load_dotenv
from .pipeline import ClaimResult, FactChecker, Report
from .verdict import Label

_COLORS = {
    Label.SUPPORTED: "\033[32m",
    Label.REFUTED: "\033[31m",
    Label.CONFLICTING: "\033[33m",
    Label.NOT_ENOUGH_EVIDENCE: "\033[90m",
}
_RESET, _BOLD, _DIM = "\033[0m", "\033[1m", "\033[2m"


def claim_to_dict(result: ClaimResult, max_evidence: int = 3) -> dict[str, Any]:
    v = result.verdict
    return {
        "claim": result.claim,
        "verdict": v.label.value,
        "confidence": round(v.confidence, 3),
        "needs_review": v.needs_review,
        "support": round(v.support, 3),
        "refute": round(v.refute, 3),
        "reasons": v.reasons,
        "check_worthiness": None if result.check_worthiness is None else round(result.check_worthiness, 3),
        "seconds": round(result.seconds, 3),
        "error": result.error,
        "evidence": [
            {
                "stance": j.stance.label,
                "supports": round(j.stance.supports, 3),
                "contradicts": round(j.stance.contradicts, 3),
                "says_nothing": round(j.stance.says_nothing, 3),
                "order_agreement": j.stance.order_agreement,
                "title": j.passage.title,
                "section": j.passage.section,
                "url": j.passage.url,
                "text": j.passage.text,
            }
            for j in result.evidence[:max_evidence]
        ],
    }


def report_to_dict(report: Report) -> dict[str, Any]:
    return {
        "claims": [claim_to_dict(c) for c in report.claims],
        "skipped_sentences": [{"sentence": s, "check_worthiness": round(p, 3)} for s, p in report.skipped],
        "seconds": round(report.seconds, 3),
        "usage": {
            "jev_requests": report.usage.requests,
            "input_tokens": report.usage.input_tokens,
            "output_tokens": report.usage.output_tokens,
            "estimated_cost_usd": round(report.usage.cost_usd, 6),
            "models": sorted(report.usage.models),
        },
    }


def render_text(report: Report, color: bool, verbose: bool) -> str:
    def paint(code: str, text: str) -> str:
        return f"{code}{text}{_RESET}" if color else text

    lines: list[str] = []
    if not report.claims:
        lines.append("No checkable factual claims found.")
    for result in report.claims:
        v = result.verdict
        flag = "  ⚑ review" if v.needs_review else ""
        lines.append(
            f"{paint(_COLORS[v.label] + _BOLD, v.label.value)} "
            f"{paint(_DIM, f'({v.confidence:.0%} confidence){flag}')}"
        )
        lines.append(f"  “{result.claim}”")
        for reason in v.reasons:
            lines.append(paint(_DIM, f"  · {reason}"))
        shown = result.evidence if verbose else result.evidence[:1]
        for j in shown:
            if not verbose and j.stance.label == "says_nothing":
                continue
            snippet = j.passage.text if verbose else _truncate(j.passage.text, 220)
            lines.append(
                f"  {paint(_DIM, j.stance.label + f' {j.weight:.2f}')} {j.passage.title}"
                + (f" › {j.passage.section}" if j.passage.section else "")
            )
            lines.append(paint(_DIM, f"    {snippet}"))
            if j.passage.url:
                lines.append(paint(_DIM, f"    {j.passage.url}"))
        lines.append("")
    if verbose and report.skipped:
        lines.append(paint(_DIM, "Not checked (not a factual claim):"))
        for sentence, score in report.skipped:
            lines.append(paint(_DIM, f"  {score:.2f}  {sentence}"))
        lines.append("")
    u = report.usage
    lines.append(
        paint(
            _DIM,
            f"{len(report.claims)} claim(s) in {report.seconds:.2f}s · {u.requests} Jev requests · "
            f"{u.input_tokens:,} input tokens (~${u.cost_usd:.5f})",
        )
    )
    return "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="factcheck",
        description="Fact-check text against retrieved evidence, using Jev to make each judgment.",
    )
    parser.add_argument("text", nargs="?", help="Text to check. Omit or pass '-' to read stdin.")
    parser.add_argument("-f", "--file", help="Read the text from a file.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show all evidence and skipped sentences.")
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Treat every sentence as a claim instead of letting Jev pick the factual ones.",
    )
    parser.add_argument("--source", choices=["wikipedia", "local"], default="wikipedia")
    parser.add_argument("--corpus", help="Directory of .txt/.md files for --source local.")
    parser.add_argument("--lang", default="en", help="Wikipedia language code (default: en).")
    parser.add_argument("--model", help="Jev model ID or alias (default: jev-latest).")
    parser.add_argument("--max-passages", type=int, help="Passages judged per claim (default: 6).")
    parser.add_argument("--no-cache", action="store_true", help="Do not use the on-disk Wikipedia cache.")
    return parser


async def _run(args: argparse.Namespace, text: str) -> Report:
    config = Config()
    if args.max_passages:
        config = replace(config, max_passages=args.max_passages)
    checker = FactChecker.create(
        source=args.source,
        corpus_dir=args.corpus,
        lang=args.lang,
        model=args.model,
        config=config,
        cache=not args.no_cache,
    )
    async with checker:
        return await checker.check_text(text, detect_claims=not args.no_detect)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.file:
        with open(args.file) as handle:
            text = handle.read()
    elif args.text and args.text != "-":
        text = args.text
    elif not sys.stdin.isatty() or args.text == "-":
        text = sys.stdin.read()
    else:
        build_parser().print_help()
        return 2
    if not text.strip():
        print("Nothing to check.", file=sys.stderr)
        return 2
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("Set TYPESAFE_API_KEY (in the environment or a .env file).", file=sys.stderr)
        return 2

    report = asyncio.run(_run(args, text))
    if args.json:
        print(json.dumps(report_to_dict(report), indent=2, ensure_ascii=False))
    else:
        print(render_text(report, color=sys.stdout.isatty(), verbose=args.verbose))
    return 0


if __name__ == "__main__":
    sys.exit(main())
