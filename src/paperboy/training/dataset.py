"""Turn the label store into a QLoRA instruction dataset for the relevance judge.

Pure stdlib (no torch/transformers) so the CLI can import it freely. Each labeled
paper becomes one (prompt -> completion) example: the prompt states the task +
the reader's topics + the paper; the completion is the VERDICT/REASON the judge
should produce. Weak gate labels and gold human ratings both feed in, with a
rating overriding the weak label for the same paper (gold wins).

Note the prompt here deliberately has NO few-shot block (unlike filter.py's
inference prompt): the point of fine-tuning is to internalize the task so the
deployed judge needs only this short prompt. Wiring filter.py to drop the
few-shot when a fine-tuned judge is configured is the deploy-time step (see
docs/finetuning.md).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

JUDGE_TASK = (
    "You are a strict relevance filter for a technical research digest. Decide whether "
    "the paper genuinely matches at least one of the reader's topics. Sharing generic "
    "machine-learning vocabulary is not enough - judge the paper's actual subject."
)


def build_prompt(topics: list[str], title: str, abstract: str) -> str:
    return (
        f"{JUDGE_TASK}\n\n"
        f"Reader's topics: {'; '.join(topics)}.\n"
        f"Title: {title}\nAbstract: {abstract}\n\n"
        "Reply in exactly two lines:\nVERDICT: KEEP or DROP\nREASON: (15 words max)"
    )


def build_completion(verdict: str, reason: str) -> str:
    keep = verdict in ("keep", "good")
    v = "KEEP" if keep else "DROP"
    r = reason.strip() or ("relevant to a topic" if keep else "not relevant to the topics")
    return f"VERDICT: {v}\nREASON: {r}"


def build_judge_examples(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One (prompt, completion) example per labeled paper.

    Features (topics + text) come from the paper's gate event (last one wins); a
    human rating for the same arxiv_id overrides the label (good->KEEP, bad->DROP).
    Papers without topics or text are skipped - the prompt can't be built.
    """
    gate: dict[str, dict] = {}
    rating: dict[str, dict] = {}
    for e in events:
        if e.get("kind") == "gate":
            gate[e["arxiv_id"]] = e
        elif e.get("kind") == "rating":
            rating[e["arxiv_id"]] = e

    examples: list[dict[str, Any]] = []
    for aid, g in gate.items():
        topics = g.get("topics") or []
        title, abstract = g.get("title", ""), g.get("abstract", "")
        if not topics or not (title or abstract):
            continue
        verdict, reason, source = g.get("verdict", ""), g.get("reason", ""), "gate"
        if aid in rating:
            rv = rating[aid]
            verdict = "keep" if rv.get("verdict") == "good" else "drop"
            reason = rv.get("reason", "") or reason
            source = "rating"
        examples.append(
            {
                "arxiv_id": aid,
                "prompt": build_prompt(topics, title, abstract),
                "completion": build_completion(verdict, reason),
                "label": 1 if verdict in ("keep", "good") else 0,
                "source": source,  # "gate" (weak) or "rating" (gold)
                # raw fields so eval can rebuild either judge prompt (few-shot or tuned)
                "topics": list(topics),
                "title": title,
                "abstract": abstract,
            }
        )
    return examples


def split_examples(
    examples: list[dict[str, Any]], val_frac: float = 0.15, seed: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic shuffle + split into (train, val). Tiny sets (< 4) go all to train."""
    if len(examples) < 4:
        return list(examples), []
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, round(len(examples) * val_frac))
    val = [examples[i] for i in idx[:n_val]]
    train = [examples[i] for i in idx[n_val:]]
    return train, val


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
