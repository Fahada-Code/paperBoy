"""Tests for the label store (feedback capture - the training data foundation)."""

from paperboy.feedback import (
    LabelEvent,
    append_events,
    load_events,
    read_manifest,
    summarize_events,
    write_manifest,
)


def test_append_and_load_roundtrip(tmp_path):
    path = tmp_path / "data" / "labels.jsonl"  # parent dir does not exist yet
    n = append_events(
        path,
        [
            LabelEvent(kind="gate", arxiv_id="1", title="A", verdict="keep", score=0.7),
            LabelEvent(kind="gate", arxiv_id="2", title="B", verdict="drop", score=0.6),
        ],
    )
    assert n == 2
    append_events(path, [LabelEvent(kind="rating", arxiv_id="1", verdict="good")])

    events = load_events(path)
    assert len(events) == 3
    assert events[0]["arxiv_id"] == "1"
    assert events[0]["verdict"] == "keep"
    assert events[2]["kind"] == "rating"


def test_load_missing_is_empty(tmp_path):
    assert load_events(tmp_path / "nope.jsonl") == []


def test_load_skips_corrupt_lines(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text('{"kind":"gate","arxiv_id":"1"}\nnot json\n\n', encoding="utf-8")
    events = load_events(path)
    assert len(events) == 1
    assert events[0]["arxiv_id"] == "1"


def test_summarize_counts_by_type():
    events = [
        {"kind": "gate", "verdict": "keep"},
        {"kind": "gate", "verdict": "drop"},
        {"kind": "gate", "verdict": "drop"},
        {"kind": "rating", "verdict": "good"},
        {"kind": "rating", "verdict": "bad"},
    ]
    counts = summarize_events(events)
    assert counts == {
        "total": 5,
        "gate_keep": 1,
        "gate_drop": 2,
        "rating_good": 1,
        "rating_bad": 1,
    }


def test_manifest_roundtrip(tmp_path):
    path = tmp_path / "data" / "last_digest.json"
    items = [{"arxiv_id": "1", "title": "A", "url": "u"}]
    write_manifest(path, items)
    assert read_manifest(path) == items
    assert read_manifest(tmp_path / "missing.json") == []


def _g(aid, verdict, score):
    return {"kind": "gate", "arxiv_id": aid, "verdict": verdict, "score": score, "title": aid}


def test_review_queue_puts_margin_cases_first():
    from paperboy.feedback import review_queue

    events = [
        _g("easy_keep", "keep", 0.80),
        _g("easy_drop", "drop", 0.38),
        _g("other_drop", "drop", 0.42),
        _g("near_cut", "keep", 0.62),
        _g("conflict", "keep", 0.30),  # kept despite a low score
        _g("flip", "drop", 0.75),
        _g("flip", "keep", 0.75),  # two runs disagreed
        _g("done", "drop", 0.41),
        {"kind": "rating", "arxiv_id": "done", "verdict": "bad"},  # already rated -> skipped
    ]
    queue, t = review_queue(events, 10)
    ids = [e["arxiv_id"] for e in queue]
    assert ids[0] == "flip" and ids[1] == "conflict"
    assert "done" not in ids
    assert ids.index("near_cut") < ids.index("easy_drop")
    assert 0.42 < t < 0.62


def test_review_queue_respects_n_and_empty():
    from paperboy.feedback import review_queue

    assert review_queue([], 5) == ([], 0.5)
    events = [_g(str(i), "keep", 0.5 + i / 100) for i in range(10)]
    assert len(review_queue(events, 3)[0]) == 3
