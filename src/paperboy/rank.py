"""Relevance ranking.

Embed the user's topics and each paper, then score each paper by its best
cosine match across the topics. No numpy — the vectors are small and there are only a few
dozen papers per run, so plain Python is plenty and keeps dependencies light.
"""

from __future__ import annotations

import math
from typing import Callable

from .reranker import Reranker, extract_features
from .sources import Paper

# Retrieval models like nomic-embed-text expect these task prefixes; without them
# all similarities cluster together and ranking loses its discrimination. Backends
# that don't need prefixes ignore them.
QUERY_PREFIX = "search_query: "
DOC_PREFIX = "search_document: "

EmbedFn = Callable[..., list[float]]
EmbedManyFn = Callable[..., list[list[float]]]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


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
    min_score: float = 0.0,
    reranker: Reranker | None = None,
    embed_many: EmbedManyFn | None = None,
) -> list[Paper]:
    """Return up to `top_n` papers most relevant to `topics`, scored in place.

    Each paper is matched by its *best* topic (the max cosine over the individual
    topic embeddings), not the average. Averaging the topics into one centroid
    blurs distinct interests together: a paper that nails one topic scores poorly
    against the mush of all of them. Max-over-topics instead lets a single strong
    match surface a paper, which is what you actually want.

    The relevance *score* is the max cosine by default. If a trained `reranker` is
    given, the score is instead its learned probability over the full distribution
    of per-topic similarities (see reranker.py) - which can tell a sharp single
    match from broad vocabulary overlap. The matched topic is unchanged either way.

    Papers scoring below `min_score` are dropped entirely, so a slow day yields a
    short (or empty) digest rather than irrelevant filler. (Note: with a reranker,
    scores are probabilities in [0, 1], so calibrate min_score accordingly.)

    With no topics configured, relevance is undefined, so we just return the most
    recent papers (the arXiv fetch is already sorted newest-first).

    Pass `embed_many` (a batch version of `embed`) to embed all papers in a few
    requests instead of one each - much faster on Ollama.
    """
    papers = _dedup(papers)
    if not topics:
        return papers[:top_n]

    if embed_many is None:

        def embed_many(texts: list[str], *, prefix: str = "") -> list[list[float]]:
            return [embed(t, prefix=prefix) for t in texts]

    topic_vecs = list(zip(topics, embed_many(topics, prefix=QUERY_PREFIX)))
    doc_vecs = embed_many([p.rank_text for p in papers], prefix=DOC_PREFIX)
    for p, doc_vec in zip(papers, doc_vecs):
        sims = [(_cosine(doc_vec, tv), t) for t, tv in topic_vecs]
        best_sim, p.matched_topic = max(sims, key=lambda st: st[0])
        if reranker is not None:
            p.score = reranker.score(extract_features([s for s, _ in sims]))
        else:
            p.score = best_sim

    papers.sort(key=lambda p: p.score, reverse=True)
    ranked = [p for p in papers if p.score >= min_score]
    return ranked[:top_n]
