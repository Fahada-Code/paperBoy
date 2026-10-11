"""Tests for arXiv feed parsing (no network)."""

from paperboy.sources.arxiv import _clean, parse_feed

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
