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


def test_binary_metrics_match_hand_computation():
    from factchecker.eval import binary_metrics

    S, R, U = Label.SUPPORTED, Label.REFUTED, Label.NOT_ENOUGH_EVIDENCE
    golds = [S, S, S, R, R, R, R, R]
    preds = [S, S, U, R, R, R, S, U]
    m = binary_metrics(golds, preds)
    assert m["unknown_rate"] == pytest.approx(2 / 8)
    assert m["accuracy_when_answered"] == pytest.approx(5 / 6)
    # SUPPORTED: tp 2, fp 1, fn 1 -> F1 2/3. REFUTED: tp 3, fp 0, fn 2 -> F1 0.75.
    assert m["f1_macro"] == pytest.approx((2 / 3 + 0.75) / 2)
    # Unknowns forced wrong: preds -> S S R R R R S S; tp 2, tn 3, fp 2, fn 1.
    assert m["mcc"] == pytest.approx((2 * 3 - 2 * 1) / (4 * 3 * 5 * 4) ** 0.5)
    assert binary_metrics([S, R], [S, R])["mcc"] == pytest.approx(1.0)


def test_combine_report_false_if_any_part_refuted():
    from factchecker.eval import combine_report
    from factchecker.pipeline import Report

    def part(label):
        return ClaimResult("p", Verdict(label, 0.9, 0, 0, False), [], [], 0.1)

    supported, refuted, unknown = part(Label.SUPPORTED), part(Label.REFUTED), part(Label.NOT_ENOUGH_EVIDENCE)
    assert combine_report("t", Report("t", [supported, refuted])).verdict.label is Label.REFUTED
    assert combine_report("t", Report("t", [supported, supported])).verdict.label is Label.SUPPORTED
    assert combine_report("t", Report("t", [supported, unknown])).verdict.label is Label.NOT_ENOUGH_EVIDENCE
    assert combine_report("t", Report("t", [])).verdict.label is Label.NOT_ENOUGH_EVIDENCE


def test_summary_includes_binary_metrics_only_for_true_false_data():
    golds = [Label.SUPPORTED, Label.REFUTED]
    results = [result(Label.SUPPORTED), result(Label.REFUTED)]
    assert summarize(golds, results)["binary"]["f1_macro"] == 1.0
    assert "binary" not in summarize(golds + [Label.NOT_ENOUGH_EVIDENCE], results + [result(Label.REFUTED)])
