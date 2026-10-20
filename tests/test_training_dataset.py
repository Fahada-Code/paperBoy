"""Tests for the judge fine-tuning dataset builder (pure stdlib)."""

from paperboy.training.dataset import (
    build_completion,
    build_judge_examples,
    build_prompt,
    split_examples,
)


def _gate(aid, verdict, topics=("t1", "t2"), title="A title", abstract="An abstract", reason="because"):
    return {
        "kind": "gate",
        "arxiv_id": aid,
        "title": title,
        "abstract": abstract,
        "topics": list(topics),
        "verdict": verdict,
        "reason": reason,
    }


def test_build_prompt_and_completion():
    p = build_prompt(["gpu serving"], "T", "A")
    assert "gpu serving" in p and "VERDICT:" in p and "few-shot" not in p.lower()
    assert build_completion("keep", "on topic") == "VERDICT: KEEP\nREASON: on topic"
    assert build_completion("drop", "").startswith("VERDICT: DROP")  # falls back to a default reason


def test_build_examples_shapes_and_labels():
    events = [_gate("1", "keep"), _gate("2", "drop")]
    ex = build_judge_examples(events)
    assert {e["arxiv_id"] for e in ex} == {"1", "2"}
    by_id = {e["arxiv_id"]: e for e in ex}
    assert by_id["1"]["label"] == 1 and by_id["2"]["label"] == 0
    assert "VERDICT: KEEP" in by_id["1"]["completion"]


def test_rating_overrides_gate_label():
    events = [
        _gate("1", "keep"),  # weak label says keep
        {"kind": "rating", "arxiv_id": "1", "verdict": "bad"},  # human says bad -> drop
    ]
    ex = build_judge_examples(events)
    assert len(ex) == 1
    assert ex[0]["label"] == 0 and ex[0]["source"] == "rating"
    assert "VERDICT: DROP" in ex[0]["completion"]


def test_examples_skip_when_missing_topics_or_text():
    events = [
        _gate("1", "keep", topics=()),  # no topics
        _gate("2", "keep", title="", abstract=""),  # no text
        _gate("3", "keep"),  # fine
    ]
    ex = build_judge_examples(events)
    assert [e["arxiv_id"] for e in ex] == ["3"]


def test_rating_without_gate_is_ignored():
    # A rating with no matching gate event has no features -> no example.
    ex = build_judge_examples([{"kind": "rating", "arxiv_id": "x", "verdict": "good"}])
    assert ex == []


def test_split_is_deterministic_and_disjoint():
    examples = [_gate(str(i), "keep") for i in range(20)]
    built = build_judge_examples(examples)
    tr1, va1 = split_examples(built, val_frac=0.15, seed=0)
    tr2, va2 = split_examples(built, val_frac=0.15, seed=0)
    assert (tr1, va1) == (tr2, va2)
    assert len(va1) == 3 and len(tr1) == 17
    ids = {e["arxiv_id"] for e in va1}
    assert not any(e["arxiv_id"] in ids for e in tr1)
