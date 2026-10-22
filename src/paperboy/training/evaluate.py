"""Score a relevance judge against labeled examples.

Pure stdlib, like dataset.py. Runs the real gate (filter.filter_relevant) over
each example so the eval exercises exactly what `run` does, then compares the
verdicts to the labels. This is the before/after check for a fine-tuned judge,
and works today for comparing stock models (e.g. llama3.2:3b vs qwen2.5:7b).

Caveat worth printing: weak labels were produced by the gate itself, so scoring
the same model on them measures self-agreement, not correctness. Gold (human
`rate`) labels are the real test, so metrics are also broken out per source.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from ..filter import filter_relevant
from ..sources import Paper


def confusion(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """Metrics for (label, predicted) pairs, where 1 = KEEP. KEEP is the positive class."""
    tp = sum(1 for y, p in pairs if y == 1 and p == 1)
    fp = sum(1 for y, p in pairs if y == 0 and p == 1)
    fn = sum(1 for y, p in pairs if y == 1 and p == 0)
    tn = sum(1 for y, p in pairs if y == 0 and p == 0)
    n = len(pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    keeps = tp + fn
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "majority": max(keeps, n - keeps) / n if n else 0.0,  # always-guess-the-common-class
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_judge(
    examples: list[dict[str, Any]],
    generate: Callable[..., str],
    *,
    tuned: bool = False,
    workers: int = 1,
    on_result: Callable[[dict[str, Any], int], None] | None = None,
) -> dict[str, Any]:
    """Run the gate on each example and score it. Returns overall metrics plus
    a per-source breakdown ("gate" weak vs "rating" gold) under "by_source"."""

    def _predict(ex: dict[str, Any]) -> int:
        paper = Paper(
            arxiv_id=ex["arxiv_id"],
            title=ex.get("title", ""),
            abstract=ex.get("abstract", ""),
            url="",
            published="",
        )
        verdicts: list[bool] = []
        filter_relevant(
            [paper],
            ex.get("topics") or [],
            generate,
            on_verdict=lambda _p, keep, _r: verdicts.append(keep),
            tuned=tuned,
        )
        return 1 if (verdicts and verdicts[0]) else 0

    # Examples can carry different topic lists, so parallelize per example.
    results: list[tuple[dict[str, Any], int]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for ex, pred in zip(examples, pool.map(_predict, examples)):
            results.append((ex, pred))
            if on_result is not None:
                on_result(ex, pred)

    out = confusion([(ex["label"], pred) for ex, pred in results])
    out["by_source"] = {
        src: confusion([(ex["label"], pred) for ex, pred in results if ex.get("source") == src])
        for src in sorted({ex.get("source", "gate") for ex, _ in results})
    }
    return out
