# Jev Fact Checker

Fast, evidence-grounded fact checking. Evidence comes from Wikipedia and the open web, every judgment is made by [TypeSafe's Jev](https://docs.typesafe.ai), and each website's trustworthiness is learned over time.

```
$ factcheck "Facebook was founded in 2004 by Mark Zuckerberg. I think it is the best website ever.
             The company is headquartered in Seattle. Marie Curie was born in Germany."

SUPPORTED (98% confidence)
  “Facebook was founded in 2004 by Mark Zuckerberg.”
  supports 1.00 britannica.com (weight 0.92) Facebook | Overview, History, Controversies, & Facts
  supports 1.00 wikipedia.org (weight 0.85) List of Facebook features

REFUTED (99% confidence)
  “The company is headquartered in Seattle.”
  contradicts 1.00 britannica.com (weight 0.92) … The company’s headquarters are in Menlo Park, California.
  contradicts 0.99 wikipedia.org (weight 0.86) History of Facebook › 2006–2012 …

REFUTED (100% confidence)
  “Marie Curie was born in Germany.”
  contradicts 1.00 nobelprize.org (weight 0.93) Marie Curie was born Marya Skłodowska in 1867 in Warsaw…
  contradicts 1.00 nih.gov (weight 0.90) …born Maria Salomea Skłodowska on November 7, 1867, in Warsaw, Poland…

3 claim(s) in 2.91s · 29 Jev requests · 31,487 input tokens (~$0.00132)
```

The opinion is skipped, "The company" is resolved from the earlier sentence, and every verdict cites independent sources.

## How it works

Jev is a *System One* model. It doesn't generate text, and it isn't a knowledge base. It takes a state and typed questions and returns calibrated probabilities, which makes it a good judge of evidence. This project pairs it with retrieval and keeps everything else in plain code:

```
text ──split (code)──▶ sentences ──Jev Noul: checkable factual claim?──▶ claims
claim ──Wikipedia (full-text + exact-title lookup) ∥ web search (Tavily)──▶ candidate documents
      ──Jev Noul: is this document about what the claim names?──▶ ≤3 documents per source
      ──chunk + BM25 (code), one best passage per website first──▶ ≤8 passages
      ──Jev Choice ×2 orderings: supports / contradicts / says_nothing──▶ judgments
      ──pool per website, weighted by reliability (code)──▶ SUPPORTED | REFUTED | CONFLICTING
                                                            | NOT_ENOUGH_EVIDENCE + review flag
      ──leave-one-out agreement (code)──▶ updated website weights
```

Each design choice below comes from TypeSafe's guidance for Jev 1.13:

| Choice | Why |
|---|---|
| Verdicts come only from retrieved evidence | Every verdict cites its sources. Jev never "remembers" a fact. |
| One narrow question per judgment | Jev reads literally; atomic questions are the most accurate. |
| Questions about the same state share one request | Fan-out runs them in parallel at no extra latency. |
| Stance asked twice with options reversed, then averaged | Cancels Jev's lean toward the first-listed option. If the two orderings disagree, the verdict is flagged. |
| Small states (one passage per stance call) | Jev's accuracy drops as irrelevant context grows. |
| Numbers checked in code | Jev is weak at numeric precision. If a claim's numbers aren't in the supporting evidence, the verdict is flagged. |
| "Same subject" wording in the questions | Keeps replicas and namesakes (*Eiffel Tower (Six Flags)*) from counting as evidence. |
| Claims like "It was…" with no earlier sentence aren't checked | There's no way to know what "it" is; searching anyway would "verify" it against any subject. |

## Source weights

Every website has a **weight** between 0 and 1: how much its evidence counts.

**Starting point (priors, in `factchecker/reliability.py`).** Wire services, reference works, scientific publishers and fact-checkers (Reuters, AP, Britannica, Nature, Snopes…), `.gov` and `.edu` sites start around 0.85–0.93. Wikipedia starts at 0.85, an unknown site at 0.6, and user-generated or social sites (Reddit, Quora, SlideShare, TikTok, X…) at 0.2–0.35.

**Combining evidence.** Each site counts **once** per claim (its strongest passage), scaled by its weight, and independent sites combine like independent witnesses:

```
support = 1 − ∏ over sites (1 − weight × site's support)
```

One trusted site can carry a verdict. Two agreeing make it stronger. A low-weight site alone can't decide anything, and a trusted refutation outweighs low-weight support (the dissent is flagged for review).

**Learning.** After a confident verdict, each site that took a side is scored against the verdict reached *without its own evidence* (leave-one-out), and its weight moves:

```
weight = (prior × 20 + agreements) / (20 + agreements + disagreements)
```

Guardrails against the usual failure modes of "agree with the majority" learning:

- **No self-confirmation.** A site's own evidence is excluded when scoring it.
- **No punishment for being alone.** A source with no independent corroboration isn't scored either way.
- **No outvoting by weak sites.** Consensus is pooled by weight, so a handful of low-weight sites never reaches the confidence needed to penalize a trusted one.
- **No echo chambers.** Copies on the same site count once per claim.
- **Gradual change.** The prior is worth 20 observations, so one claim only nudges a weight.

Ground truth beats consensus: `factcheck-eval --learn` updates weights from labelled claims. Weights persist in `~/.cache/jev-factchecker/source_reliability.json`. View them with `factcheck --weights`, turn off learning for a run with `--no-learn`, or delete the file to reset.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Create a `.env` file in the project directory (it is git-ignored):

```
TYPESAFE_API_KEY=your-key-from-console.typesafe.ai
FACTCHECK_USER_AGENT=jev-factchecker/0.1 (you@example.com)
TAVILY_API_KEY=tvly-...        # optional
```

- Wikimedia requires a User-Agent with contact info and throttles anonymous clients heavily.
- Without `TAVILY_API_KEY`, web search uses Tavily's free, rate-limited keyless mode. A [free key](https://app.tavily.com) gives 1,000 searches/month (about 500–1,000 claims).

## Usage

```bash
factcheck "The Great Wall of China is located in Japan."
factcheck -f article.txt -v                    # all evidence, plus the sentences that were skipped
echo "..." | factcheck --json                  # machine-readable output
factcheck --sources wikipedia "..."            # encyclopedia only
factcheck --corpus ./docs "Refunds are available for 90 days."   # also search your own documents
factcheck --weights                            # learned website weights
```

From Python:

```python
import asyncio
from factchecker import FactChecker

async def main():
    async with FactChecker.create() as checker:
        report = await checker.check_text("Marie Curie was born in Germany.")
        for claim in report.claims:
            print(claim.verdict.label, claim.verdict.confidence)
            for j in claim.evidence[:2]:
                print(" ", j.stance.label, j.site, round(j.reliability, 2), j.passage.url)

asyncio.run(main())
```

`needs_review` is the important field for automation. False means the verdict was decisive, no trusted source dissented, the two option orderings agreed, and the numbers checked out. True means a person should look before acting on it.

## Measuring accuracy

```bash
factcheck-eval data/sample_claims.jsonl                     # 24 hand-written smoke-test claims
factcheck-eval data/fever_heldout_300.jsonl --out out.jsonl # held-out benchmark sample
factcheck-eval train.jsonl --learn --weights-file w.json    # learn weights from gold labels
```

`factcheck-eval` reads JSON Lines with `claim` and `label` fields, and FEVER labels are accepted as-is. It reports accuracy, accuracy on supported/refuted claims, macro-F1, per-class precision/recall, **auto-accepted accuracy** (accuracy of verdicts *not* flagged for review, and their share), latency, and Jev cost. Site-weight learning is off while measuring, so scores don't depend on claim order.

The FEVER files are seeded, stratified samples (100 per label) of the [FEVER](https://fever.ai) paper-dev set:

- `fever_sample_300.jsonl` was used for development.
- `fever_heldout_300.jsonl` is disjoint from it and is reserved for reporting.

FEVER was labelled against 2017 Wikipedia, so many of its "not enough info" claims are now answerable. Accuracy on supported/refuted claims is the cleaner number. Download `paper_dev.jsonl` from fever.ai to draw new samples.

## Tests

```bash
.venv/bin/python -m pytest        # offline: scripted fake Jev, mocked Wikipedia and Tavily
```

## Limits

- **Evidence-bound.** If no source covers something, the answer is NOT_ENOUGH_EVIDENCE, never a guess.
- **Weights are per website, not per article or author.** A good outlet's bad article borrows the outlet's weight. `blogspot.com` blogs share one weight.
- **Consensus learning reflects what's corroborated, not what's true.** On contested topics, prefer learning from labelled data (`--learn`), and review `factcheck --weights` from time to time.
- **Whole sentences are claims.** Jev can't rewrite text, so a sentence holding several claims is judged as one. "Supports" requires every part to be supported.
- **English first.** Jev's accuracy is best in English.
- **Adversarial text** written to steer the model can move answers. Treat `needs_review` verdicts accordingly.

## Cost and speed

Jev 1.13 costs $0.042 per million input tokens, and output tokens are free. A claim uses about 10 Jev requests (~$0.0004 per claim). End to end, a claim takes about 1–2.5s, mostly spent in web search. Claims in a text are checked concurrently, and repeated lookups are served from an on-disk cache.
