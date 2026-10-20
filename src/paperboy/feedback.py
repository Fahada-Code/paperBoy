"""Feedback and label capture - the data foundation for training.

Every relevance-gate decision (a weak label) and every human rating (a gold
label) is appended to a JSONL label store. Over time these accumulate the
(paper, topics) -> keep/drop and (paper) -> good/bad examples that a future
learned re-ranker or QLoRA fine-tune will train on. JSONL is append-only,
human-readable, and trivial to stream into a trainer. Labels are user-local
(gitignored), like config.yaml and digests/.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class LabelEvent:
    """One labeling event: a gate verdict (weak label) or a human rating (gold).

    The same schema holds both so the store is a single, joinable stream. A
    trainer selects on `kind`: gate events give (title/abstract/topics -> verdict)
    for the judge, plus (score, matched_topic -> verdict) for a re-ranker; rating
    events give the human's ground truth, joined to gate events by `arxiv_id`.
    """

    kind: str  # "gate" | "rating"
    arxiv_id: str
    title: str = ""
    abstract: str = ""
    url: str = ""
    topics: list[str] = field(default_factory=list)
    matched_topic: str = ""
    score: float = 0.0  # embedding relevance score
    verdict: str = ""  # gate: "keep"|"drop";  rating: "good"|"bad"
    reason: str = ""  # gate reason, or an optional human note
    selected: bool = False  # did it make the final digest
    model: str = ""  # judge model, for gate events
    ts: str = field(default_factory=_now)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def append_events(path: str | Path, events: Iterable[LabelEvent]) -> int:
    """Append events to the JSONL store, creating parent dirs. Returns the count
    written. Each event is one line, so a crash mid-write loses at most one line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as fh:
        for e in events:
            fh.write(e.to_json() + "\n")
            n += 1
    return n


def load_events(path: str | Path) -> list[dict[str, Any]]:
    """Load all events from the store (empty list if missing). Blank or corrupt
    lines are skipped rather than failing the whole read."""
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def summarize_events(events: list[dict[str, Any]]) -> dict[str, int]:
    """Count events by type for the `labels` command."""
    counts = {
        "total": len(events),
        "gate_keep": 0,
        "gate_drop": 0,
        "rating_good": 0,
        "rating_bad": 0,
    }
    for e in events:
        key = f"{e.get('kind')}_{e.get('verdict')}"
        if key in counts:
            counts[key] += 1
    return counts


def write_manifest(path: str | Path, items: list[dict[str, str]]) -> None:
    """Record the papers in the latest digest so `rate` knows what to ask about."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Load the latest-digest manifest (empty list if missing/corrupt)."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []
