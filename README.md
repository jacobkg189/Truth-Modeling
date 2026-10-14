# Jev Fact Checker

Fast, evidence-grounded fact checking, with [TypeSafe's Jev](https://docs.typesafe.ai) making every judgment.

```
$ factcheck "Facebook was founded in 2004 by Mark Zuckerberg. I think it is the best website ever. The company is headquartered in Seattle."

SUPPORTED (100% confidence)
  “Facebook was founded in 2004 by Mark Zuckerberg.”
  supports 1.00 Facebook
    It was founded in 2004 by Mark Zuckerberg, along with his Harvard College roommates…

REFUTED (99% confidence)
  “The company is headquartered in Seattle.”
  contradicts 0.99 History of Facebook › 2006–2012: Public access, Microsoft alliance, and rapid growth
    In February 2011, Facebook announced plans to move its headquarters to … Menlo Park, California.

2 claim(s) in 0.82s · 16 Jev requests · 13,127 input tokens (~$0.00055)
```

The opinion ("I think it is the best website ever") is skipped automatically, and "The company" is resolved from the earlier sentence.

## How it works

Jev is a *System One* model. It does not generate text, and it is not a knowledge base. It takes a state and typed questions and returns calibrated probabilities. That makes it a good judge of evidence, so this project pairs it with retrieval and keeps everything else in plain code:

```
text ──split (code)──▶ sentences ──Jev Noul: checkable factual claim?──▶ claims
claim ──Wikipedia full-text search + exact-title entity lookup (code)──▶ candidate articles
      ──Jev Noul: is this article about what the claim names?──▶ ≤3 articles
      ──chunk + BM25 (code)──▶ ≤7 passages
      ──Jev Choice ×2 orderings: supports / contradicts / says_nothing──▶ judgments
      ──aggregate (code)──▶ SUPPORTED | REFUTED | CONFLICTING | NOT_ENOUGH_EVIDENCE
                            + confidence + needs_review flag + cited passages
```

Design choices, each taken from TypeSafe's guidance for Jev 1.13:

| Choice | Why |
|---|---|
| Verdicts come only from retrieved evidence | Every verdict cites its source. Jev never "remembers" a fact. |
| One narrow question per judgment | Jev reads literally; atomic questions are the most accurate. |
| Questions about the same state share one request | Fan-out runs them in parallel at no extra latency. |
| Stance asked twice with the options reversed, then averaged | Cancels Jev's lean toward the first-listed option. Disagreement between the two orderings flags the verdict for review. |
| Small states (one passage per stance call) | Jev's accuracy drops as irrelevant context grows. |
| Numbers checked in code | Jev is weak at numeric precision. If a claim's numbers don't appear in its supporting evidence, the verdict is flagged for review. |
| "Same subject" wording in the questions | Stops replicas and namesakes (e.g. *Eiffel Tower (Six Flags)*) from counting as evidence. |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Create a `.env` file in the project directory (it is git-ignored):

```
TYPESAFE_API_KEY=your-key-from-console.typesafe.ai
FACTCHECK_USER_AGENT=jev-factchecker/0.1 (you@example.com)
```

Wikimedia requires a User-Agent with contact info and throttles anonymous clients heavily.

## Usage

```bash
factcheck "The Great Wall of China is located in Japan."
factcheck -f article.txt -v            # all evidence, plus the sentences that were skipped
echo "..." | factcheck --json          # machine-readable output
factcheck --no-detect "..."            # check every sentence, not only the factual ones
factcheck --source local --corpus ./docs "Refunds are available for 90 days."  # your own documents
```

From Python:

```python
import asyncio
from factchecker import FactChecker

async def main():
    async with FactChecker.create() as checker:
        report = await checker.check_text("Marie Curie was born in Germany.")
        for claim in report.claims:
            print(claim.verdict.label, claim.verdict.confidence, claim.evidence[0].passage.url)

asyncio.run(main())
```

`needs_review` is the important field for automation. When it is false, the verdict was decisive, the two option orderings agreed, and the numbers checked out. When it is true, a person should look before acting.

## Measuring accuracy

```bash
factcheck-eval data/sample_claims.jsonl              # 24 hand-written smoke-test claims
factcheck-eval data/fever_sample_300.jsonl --out results.jsonl
```

`factcheck-eval` reads JSON Lines with `claim` and `label` fields. FEVER labels (`SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO`) are accepted as-is. It reports:

- accuracy and macro-F1, plus per-class precision and recall
- **auto-accepted accuracy**: the accuracy of verdicts *not* flagged for review, and the share of claims they cover
- p50/p95 latency, Jev token usage and cost

`data/fever_sample_300.jsonl` is a seeded, stratified sample (100 per label) of the [FEVER](https://fever.ai) paper-dev set. FEVER was labelled against 2017 Wikipedia, so some "not enough info" claims are verifiable against today's Wikipedia.

All thresholds live in `factchecker/config.py`. Tune them on labelled claims from your own domain: because Jev's probabilities are calibrated, raising `review_below` buys precision at a predictable cost in coverage.

## Tests

```bash
.venv/bin/python -m pytest        # offline; uses a scripted fake Jev and mocked Wikipedia
```

## Limits

- **Evidence-bound.** If Wikipedia (or your corpus) doesn't cover something, the answer is NOT_ENOUGH_EVIDENCE, never a guess. Breaking news and niche topics need a fresher evidence source; implement the `EvidenceSource` protocol in `sources.py` to add one (e.g. a news search API).
- **Whole sentences are claims.** Jev can't rewrite text, so a sentence holding several claims is judged as one. "Supports" requires every part to be supported.
- **English first.** Jev's accuracy is best in English.
- **Adversarial text** written to steer the model can move answers. Treat `needs_review` verdicts accordingly.

## Cost and speed

Jev 1.13 costs $0.042 per million input tokens, and output tokens are free. A typical claim uses ~8 Jev requests and ~700 input tokens per request, so about **$0.0003 per claim**. A single claim takes about 0.5–1s end to end, and most of that is Wikipedia. Claims within a text are checked concurrently.
