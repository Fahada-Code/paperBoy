"""Tests for the delivered-papers history (no repeats across digests)."""

from datetime import date

from paperboy.history import base_id, drop_seen, load_seen, mark_seen
from paperboy.sources import Paper


def _paper(pid: str) -> Paper:
    return Paper(arxiv_id=pid, title=pid, abstract="", url="", published="")


def test_base_id_strips_version():
    assert base_id("2610.12345v3") == "2610.12345"


def test_mark_then_drop_ignores_versions(tmp_path):
    path = tmp_path / "seen.json"
    mark_seen(path, [_paper("2610.1v1")], today=date(2026, 10, 22))
    left = drop_seen([_paper("2610.1v2"), _paper("2610.2v1")], load_seen(path))
    assert [p.arxiv_id for p in left] == ["2610.2v1"]


def test_first_delivery_date_kept_and_old_entries_pruned(tmp_path):
    path = tmp_path / "seen.json"
    mark_seen(path, [_paper("old"), _paper("a")], today=date(2026, 9, 1))
    mark_seen(path, [_paper("a"), _paper("b")], today=date(2026, 10, 22))
    seen = load_seen(path)
    assert "old" not in seen  # older than 30 days
    assert seen == {"b": "2026-10-22"}  # "a" was first seen 2026-09-01, so it aged out too


def test_missing_or_corrupt_store_is_empty(tmp_path):
    assert load_seen(tmp_path / "nope.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    assert load_seen(bad) == {}
