import json

import pytest

from factchecker.eval import load_examples, summarize
from factchecker.pipeline import ClaimResult
from factchecker.verdict import Label, Verdict


def result(label: Label, needs_review: bool = False, error: str | None = None) -> ClaimResult:
    return ClaimResult("c", Verdict(label, 0.9, 0, 0, needs_review), [], [], 0.5, error=error)


def test_load_examples_accepts_fever_labels(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"claim": c, "label": l})
            for c, l in [("a", "SUPPORTS"), ("b", "REFUTES"), ("c", "NOT ENOUGH INFO")]
        )
        + "\n\n"
    )
    assert [l for _, l in load_examples(path)] == [
        Label.SUPPORTED, Label.REFUTED, Label.NOT_ENOUGH_EVIDENCE,
    ]
    assert len(load_examples(path, limit=2)) == 2
    path.write_text(json.dumps({"claim": "x", "label": "MAYBE"}))
    with pytest.raises(ValueError):
        load_examples(path)


def test_summarize_metrics():
    golds = [Label.SUPPORTED, Label.REFUTED, Label.NOT_ENOUGH_EVIDENCE, Label.SUPPORTED]
    results = [
        result(Label.SUPPORTED),
        result(Label.NOT_ENOUGH_EVIDENCE, needs_review=True),
        result(Label.NOT_ENOUGH_EVIDENCE),
        result(Label.REFUTED, error="boom"),
    ]
    s = summarize(golds, results)
    assert s["accuracy"] == 0.5
    assert s["accuracy_supported_refuted"] == pytest.approx(1 / 3)
    assert s["auto_accepted_share"] == 0.75
    assert s["auto_accepted_accuracy"] == pytest.approx(2 / 3)
    assert s["errors"] == 1
    assert s["confusion"]["SUPPORTED"] == {"SUPPORTED": 1, "REFUTED": 1}
