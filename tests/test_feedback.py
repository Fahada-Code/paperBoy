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
