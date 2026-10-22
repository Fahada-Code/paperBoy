"""LLM relevance gate.

Embeddings are cheap but blunt: at the margin they can't tell a genuinely
on-topic paper from one that merely shares machine-learning vocabulary (a
neural-PDE paper and a speculative-decoding paper can score identically). After
embeddings narrow the field, we ask the LLM to judge each shortlisted paper -
keep or drop, with a one-line reason. This is classification and explanation of
relevance, not advice or prediction, so it stays within the project's guardrail.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .backends import BackendError
from .sources import Paper
from .training.dataset import build_prompt

GenerateFn = Callable[..., str]

JUDGE_SYSTEM = (
    "You are a strict relevance filter for a technical research digest. You decide "
    "whether a paper genuinely matches at least one of the reader's topics. Sharing "
    "the broad field (machine learning) is NOT enough - a paper that only shares "
    "generic ML vocabulary must be dropped. Judge the paper's actual subject against "
    "the specific topics. You never give advice or predictions; you only classify."
)

# Two worked examples steady a small model at the margin (where embeddings and a
# 7B judge both wobble): one clear keep, one paper that shares ML words but is
# off-topic. The reasoning line forces the model to commit to a subject before
# voting, which cuts the "share vocabulary -> keep" failure mode.
_FEWSHOT = (
    "Example A\n"
    "Reader's topics: LLM inference optimization and serving; quantization.\n"
    "Title: FOLD: Far Memory for Long-Context Speculative Decoding\n"
    "Abstract: Speeds up LLM decoding by treating distant KV-cache entries as far memory.\n"
    "THOUGHT: Speculative decoding is squarely LLM inference optimization.\n"
    "VERDICT: KEEP\n"
    "REASON: directly optimizes LLM inference/serving\n"
    "\n"
    "Example B\n"
    "Reader's topics: LLM inference optimization and serving; quantization.\n"
    "Title: ReAl: Refine neural operators via aligning dynamical flow\n"
    "Abstract: Improves neural-PDE surrogates for physical simulation stability.\n"
    "THOUGHT: Neural-PDE solvers only share ML words; unrelated to LLM serving or quantization.\n"
    "VERDICT: DROP\n"
    "REASON: neural-PDE work, no link to the topics\n"
    "\n"
    "Example C\n"
    "Reader's topics: LLM inference optimization and serving; quantization.\n"
    "Title: Accelerated Newton Methods for Bilevel Optimization in Meta-Learning\n"
    "Abstract: Faster convergence for bilevel optimization and hyperparameter tuning.\n"
    "THOUGHT: Optimization theory for meta-learning is a different ML subfield - it is not "
    "one of the listed topics, even though it is adjacent.\n"
    "VERDICT: DROP\n"
    "REASON: optimization/meta-learning, not a listed topic\n"
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
    tuned: bool = False,
    workers: int = 1,
) -> list[Paper]:
    """Return the subset of `papers` the LLM judges genuinely relevant to `topics`.

    Input order is preserved (callers pass papers already sorted by embedding
    score). Each kept paper gets its `.relevance_reason` set. If the backend
    errors on a paper we keep it (fail open - a borderline include beats a crash).
    `on_verdict` is called for every paper with (paper, kept, reason) for progress.

    `tuned=True` is for a fine-tuned judge: it sends the exact short prompt the
    model was trained on (training/dataset.py) with no few-shot and no system
    prompt, so inference matches training and each call costs fewer tokens.

    `workers > 1` sends that many judge requests at once. Ollama batches parallel
    requests on the GPU: 4 workers judged 12 papers in 22s vs 49s one at a time,
    with identical verdicts (temperature 0). Results still come back in order.
    """
    if not topics:
        return papers

    topic_list = "; ".join(topics)

    def _judge_one(p: Paper) -> tuple[bool, str]:
        if tuned:
            prompt, system = build_prompt(topics, p.title, p.abstract), None
        else:
            prompt, system = _fewshot_prompt(topic_list, p), JUDGE_SYSTEM
        try:
            return _verdict(generate(prompt, system=system, temperature=0.0))
        except BackendError:
            return True, "kept (judge unavailable)"

    # Both map()s are lazy and ordered, so progress still streams paper by paper.
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 and len(papers) > 1 else None
    verdicts = pool.map(_judge_one, papers) if pool else map(_judge_one, papers)
    kept: list[Paper] = []
    try:
        for p, (keep, reason) in zip(papers, verdicts):
            if on_verdict is not None:
                on_verdict(p, keep, reason)
            if keep:
                p.relevance_reason = reason
                kept.append(p)
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    return kept


def _fewshot_prompt(topic_list: str, p: Paper) -> str:
    return (
        f"{_FEWSHOT}\n"
        "Now judge this paper the same way.\n"
        f"Reader's topics: {topic_list}.\n"
        f"Title: {p.title}\nAbstract: {p.abstract}\n\n"
        "Reply in exactly three lines:\n"
        "THOUGHT: (one sentence naming the paper's actual subject vs the topics)\n"
        "VERDICT: KEEP or DROP\n"
        "REASON: (15 words max)"
    )
