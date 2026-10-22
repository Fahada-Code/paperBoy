"""Tests for the judge eval harness (no network: a fake generate fn)."""

from paperboy.training.evaluate import confusion, evaluate_judge


def _ex(aid, label, title, source="gate"):
    return {
        "arxiv_id": aid,
        "label": label,
        "source": source,
        "topics": ["gpu serving"],
        "title": title,
        "abstract": "abs",
    }


def test_confusion_metrics():
    m = confusion([(1, 1), (1, 0), (0, 1), (0, 0), (0, 0)])
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 2)
    assert m["accuracy"] == 3 / 5
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["majority"] == 3 / 5


def test_confusion_empty_is_safe():
    assert confusion([])["accuracy"] == 0.0


def test_evaluate_scores_judge_and_splits_by_source():
    def fake_generate(prompt: str, *, system=None, temperature: float = 0.0) -> str:
        # Keeps anything with "serving" in the title, drops the rest.
        last = prompt.rsplit("Title:", 1)[1]
        return "VERDICT: KEEP\nREASON: ok" if "serving" in last.split("\n")[0] else "VERDICT: DROP\nREASON: no"

    examples = [
        _ex("1", 1, "LLM serving tricks"),  # right keep
        _ex("2", 0, "Sourdough kinetics"),  # right drop
        _ex("3", 1, "Quantized kernels", source="rating"),  # miss (judge drops a gold keep)
    ]
    m = evaluate_judge(examples, fake_generate)
    assert m["n"] == 3 and m["tp"] == 1 and m["tn"] == 1 and m["fn"] == 1
    assert m["by_source"]["gate"]["accuracy"] == 1.0
    assert m["by_source"]["rating"]["accuracy"] == 0.0


def test_evaluate_tuned_mode_uses_short_prompt():
    seen = []

    def fake_generate(prompt: str, *, system=None, temperature: float = 0.0) -> str:
        seen.append(prompt)
        return "VERDICT: KEEP\nREASON: ok"

    evaluate_judge([_ex("1", 1, "x")], fake_generate, tuned=True)
    assert "Example A" not in seen[0]
