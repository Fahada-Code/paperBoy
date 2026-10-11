"""Relevance ranking.

Embed the user's topics and each paper, then score papers by cosine similarity
to the topic centroid. No numpy — the vectors are small and there are only a few
dozen papers per run, so plain Python is plenty and keeps dependencies light.
"""

from __future__ import annotations

import math
from typing import Callable

from .sources import Paper

# Retrieval models like nomic-embed-text expect these task prefixes; without them
# all similarities cluster together and ranking loses its discrimination. Backends
# that don't need prefixes ignore them.
QUERY_PREFIX = "search_query: "
DOC_PREFIX = "search_document: "

EmbedFn = Callable[..., list[float]]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]


def _dedup(papers: list[Paper]) -> list[Paper]:
    """Drop duplicate papers, keyed by arXiv id without the version suffix."""
    seen: set[str] = set()
    out: list[Paper] = []
    for p in papers:
        key = p.arxiv_id.split("v")[0]
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def rank_papers(
    papers: list[Paper],
    topics: list[str],
    embed: EmbedFn,
    top_n: int,
) -> list[Paper]:
    """Return the `top_n` papers most relevant to `topics`, scored in place.

    With no topics configured, relevance is undefined, so we just return the most
    recent papers (the arXiv fetch is already sorted newest-first).
    """
    papers = _dedup(papers)
    if not topics:
        return papers[:top_n]

    topic_centroid = _centroid([embed(t, prefix=QUERY_PREFIX) for t in topics])
    for p in papers:
        p.score = _cosine(embed(p.rank_text, prefix=DOC_PREFIX), topic_centroid)

    papers.sort(key=lambda p: p.score, reverse=True)
    return papers[:top_n]
