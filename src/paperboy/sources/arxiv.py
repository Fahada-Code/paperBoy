"""Fetch recent papers from the arXiv API (free, no auth).

arXiv returns an Atom feed; we parse it with the stdlib. We ask for the most
recently submitted papers in the configured categories and hand back plain
`Paper` objects for ranking and summarization.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import httpx

ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"

# arXiv occasionally times out or returns a transient 5xx. A single hiccup
# should not kill a scheduled digest run, so we retry these with backoff.
_RETRY_STATUS = frozenset({500, 502, 503, 504})


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    url: str
    published: str
    authors: list[str] = field(default_factory=list)
    primary_category: str = ""
    score: float = 0.0  # relevance, filled in by the ranker
    matched_topic: str = ""  # the topic that scored highest, filled in by the ranker

    @property
    def rank_text(self) -> str:
        """Text used for relevance embedding: title carries the most signal."""
        return f"{self.title}\n\n{self.abstract}"


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def fetch_arxiv(
    categories: list[str],
    max_results: int = 60,
    timeout: int = 30,
    retries: int = 3,
    backoff: float = 2.0,
) -> list[Paper]:
    """Return the most recently submitted papers across `categories`.

    Transient failures (timeouts, connection errors, 5xx) are retried up to
    `retries` times with exponential backoff (`backoff`, 2*backoff, ...). A
    non-transient error (e.g. a 4xx) is raised immediately.
    """
    if not categories:
        return []

    # Join with " OR " (not "+OR+"): httpx encodes the spaces to "+", which is
    # arXiv's expected separator. A literal "+" would encode to "%2B" and break.
    search_query = " OR ".join(f"cat:{c}" for c in categories)
    params = {
        "search_query": search_query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": 0,
        "max_results": max_results,
    }

    last_exc: httpx.HTTPError | None = None
    for attempt in range(retries):
        try:
            resp = httpx.get(ARXIV_API, params=params, timeout=timeout, follow_redirects=True)
            resp.raise_for_status()
            return parse_feed(resp.text)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _RETRY_STATUS:
                raise  # 4xx etc. won't fix themselves - fail fast
            last_exc = exc
        except httpx.TransportError as exc:
            last_exc = exc  # timeouts, connection errors - worth retrying
        if attempt < retries - 1:
            time.sleep(backoff * (2 ** attempt))

    assert last_exc is not None  # loop only exits here after a caught error
    raise last_exc


def parse_feed(xml_text: str) -> list[Paper]:
    """Parse an arXiv Atom feed into Paper objects. Split out so it's testable
    without hitting the network."""
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    for entry in root.findall(f"{_ATOM}entry"):
        raw_id = _clean(entry.findtext(f"{_ATOM}id"))
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1]  # e.g. 2401.01234v1

        authors = [
            _clean(a.findtext(f"{_ATOM}name"))
            for a in entry.findall(f"{_ATOM}author")
        ]
        primary = entry.find(f"{_ATOM}category")
        primary_category = primary.get("term", "") if primary is not None else ""

        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=_clean(entry.findtext(f"{_ATOM}title")),
                abstract=_clean(entry.findtext(f"{_ATOM}summary")),
                url=raw_id,
                published=_clean(entry.findtext(f"{_ATOM}published"))[:10],
                authors=authors,
                primary_category=primary_category,
            )
        )
    return papers
