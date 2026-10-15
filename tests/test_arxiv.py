"""Tests for arXiv feed parsing and fetch retry logic (no network)."""

import httpx
import pytest

from paperboy.sources import arxiv
from paperboy.sources.arxiv import _clean, fetch_arxiv, parse_feed

SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2610.12345v1</id>
    <title>A   Great   Paper
    On Batching</title>
    <summary>We   show   things.</summary>
    <published>2026-10-08T12:00:00Z</published>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <category term="cs.LG"/>
    <category term="cs.DC"/>
  </entry>
</feed>
"""


def test_clean_collapses_whitespace():
    assert _clean("  a\n  b   c ") == "a b c"
    assert _clean(None) == ""


def test_parse_feed_extracts_fields():
    papers = parse_feed(SAMPLE_FEED)
    assert len(papers) == 1
    p = papers[0]
    assert p.arxiv_id == "2610.12345v1"
    assert p.title == "A Great Paper On Batching"
    assert p.abstract == "We show things."
    assert p.published == "2026-10-08"
    assert p.authors == ["Ada Lovelace", "Alan Turing"]
    assert p.primary_category == "cs.LG"  # first category is primary
    assert p.url == "http://arxiv.org/abs/2610.12345v1"


def test_parse_feed_empty():
    assert parse_feed('<feed xmlns="http://www.w3.org/2005/Atom"></feed>') == []


def _patch_get(monkeypatch, responses):
    """Make httpx.get yield each item in `responses` (an Exception is raised,
    anything else is returned). Also no-op out the backoff sleep."""
    calls = {"n": 0}

    def fake_get(*args, **kwargs):
        item = responses[calls["n"]]
        calls["n"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(arxiv.httpx, "get", fake_get)
    monkeypatch.setattr(arxiv.time, "sleep", lambda _s: None)
    return calls


class _FakeResp:
    def __init__(self, text=SAMPLE_FEED):
        self.text = text

    def raise_for_status(self):
        return None


def test_fetch_retries_then_succeeds(monkeypatch):
    calls = _patch_get(
        monkeypatch,
        [httpx.ReadTimeout("timed out"), httpx.ConnectError("refused"), _FakeResp()],
    )
    papers = fetch_arxiv(["cs.LG"], retries=3)
    assert len(papers) == 1
    assert calls["n"] == 3  # failed twice, succeeded on the third


def test_fetch_raises_after_exhausting_retries(monkeypatch):
    _patch_get(monkeypatch, [httpx.ReadTimeout("t")] * 3)
    with pytest.raises(httpx.ReadTimeout):
        fetch_arxiv(["cs.LG"], retries=3)


def test_fetch_does_not_retry_4xx(monkeypatch):
    err = httpx.HTTPStatusError(
        "bad request", request=httpx.Request("GET", arxiv.ARXIV_API),
        response=httpx.Response(400),
    )
    calls = _patch_get(monkeypatch, [err, _FakeResp()])
    with pytest.raises(httpx.HTTPStatusError):
        fetch_arxiv(["cs.LG"], retries=3)
    assert calls["n"] == 1  # 4xx fails fast, no retry
