"""LLM relevance gate.

Embeddings are cheap but blunt: at the margin they can't tell a genuinely
on-topic paper from one that merely shares machine-learning vocabulary (a
neural-PDE paper and a speculative-decoding paper can score identically). After
embeddings narrow the field, we ask the LLM to judge each shortlisted paper -
keep or drop, with a one-line reason. This is classification and explanation of
relevance, not advice or prediction, so it stays within the project's guardrail.
"""

from __future__ import annotations

from typing import Callable

from .backends import BackendError
from .sources import Paper

GenerateFn = Callable[..., str]

JUDGE_SYSTEM = (
    "You are a strict relevance filter for a technical research digest. You decide "
    "whether a paper genuinely matches at least one of the reader's topics. A paper "
    "that only shares generic machine-learning vocabulary is NOT a match - be willing "
    "to drop it. You never give advice or predictions; you only classify relevance."
)


# Phrases that mean "not relevant". Small models sometimes emit VERDICT: KEEP
# with a reason that plainly says the opposite - when they contradict, trust the
# reason and drop, since a stated irrelevance is the clearer signal.
_IRRELEVANT_MARKERS = (
    "irrelevant",
    "not relevant",
    "no relevance",
    "no direct relevance",
    "unrelated",
    "no match",
)


def _verdict(reply: str) -> tuple[bool, str]:
    """Parse 'VERDICT: KEEP|DROP' and 'REASON: ...' out of the model reply.

    Fail open: if the reply has no clear verdict, treat it as KEEP so a confused
    model trims nothing rather than silently discarding real papers. But if a KEEP
    verdict carries a reason that plainly says "irrelevant", trust the reason.
    """
    keep: bool | None = None
    reason = ""
    for line in reply.splitlines():
        low = line.strip().lower()
        if low.startswith("verdict:"):
            if "drop" in low:
                keep = False
            elif "keep" in low:
                keep = True
        elif low.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
    if keep is None:
        # No explicit verdict line: only drop if the model clearly said so.
        low = reply.lower()
        keep = not ("drop" in low and "keep" not in low)
    if keep and reason and any(m in reason.lower() for m in _IRRELEVANT_MARKERS):
        keep = False  # verdict/reason contradiction -> believe the reason
    return keep, reason


def filter_relevant(
    papers: list[Paper],
    topics: list[str],
    generate: GenerateFn,
    *,
    on_verdict: Callable[[Paper, bool, str], None] | None = None,
) -> list[Paper]:
    """Return the subset of `papers` the LLM judges genuinely relevant to `topics`.

    Input order is preserved (callers pass papers already sorted by embedding
    score). Each kept paper gets its `.relevance_reason` set. If the backend
    errors on a paper we keep it (fail open - a borderline include beats a crash).
    `on_verdict` is called for every paper with (paper, kept, reason) for progress.
    """
    if not topics:
        return papers

    topic_list = "; ".join(topics)
    kept: list[Paper] = []
    for p in papers:
        prompt = (
            f"Reader's topics: {topic_list}.\n\n"
            f"Paper:\nTitle: {p.title}\nAbstract: {p.abstract}\n\n"
            "Is this paper genuinely relevant to at least one topic? Reply in exactly "
            "two lines:\nVERDICT: KEEP or DROP\nREASON: (15 words max)"
        )
        try:
            reply = generate(prompt, system=JUDGE_SYSTEM, temperature=0.0)
            keep, reason = _verdict(reply)
        except BackendError:
            keep, reason = True, "kept (judge unavailable)"

        if on_verdict is not None:
            on_verdict(p, keep, reason)
        if keep:
            p.relevance_reason = reason
            kept.append(p)
    return kept
