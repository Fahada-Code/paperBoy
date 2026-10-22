"""Structured per-run metrics.

Every `paperboy run` appends one JSON line to data_dir/runs.jsonl: how long each
stage took, how many papers survived each step, which models ran, and whether
the run finished. Cheap, append-only, and enough to answer "is the digest getting
slower / emptier / flakier?" without guessing from console scrollback.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunMetrics:
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    status: str = "running"  # ok | empty | failed
    failed_stage: str = ""
    summary_model: str = ""
    judge_model: str = ""
    embed_model: str = ""
    fetched: int = 0
    shortlisted: int = 0
    gate_kept: int = 0
    gate_dropped: int = 0
    selected: int = 0
    top_score: float = 0.0
    low_score: float = 0.0
    output: str = ""
    seconds: dict[str, float] = field(default_factory=dict)  # stage -> wall seconds
    _stage: str = field(default="", repr=False)
    _last: str = field(default="", repr=False)
    _t_stage: float = field(default=0.0, repr=False)
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def mark(self, name: str) -> None:
        """Start timing stage `name`, closing the previous one. Marks keep the
        pipeline code flat (no nested with-blocks) and the last mark tells us
        where a failed run died."""
        now = time.perf_counter()
        if self._stage:
            self.seconds[self._stage] = round(self.seconds.get(self._stage, 0.0) + now - self._t_stage, 2)
        self._stage, self._t_stage = name, now
        if name:
            self._last = name

    def finish(self, status: str) -> None:
        self.mark("")  # close the open stage
        self.status = status
        if status == "failed":
            self.failed_stage = self._last
        self.seconds["total"] = round(time.perf_counter() - self._t0, 2)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}


def append_run(path: str | Path, m: RunMetrics) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(m.to_dict(), ensure_ascii=False) + "\n")


def load_runs(path: str | Path) -> list[dict[str, Any]]:
    """All logged runs, oldest first (empty if missing). Corrupt lines are skipped."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate stats for `paperboy stats`: status counts, mean stage timings
    and mean papers selected, over the given runs."""
    counts: dict[str, int] = {}
    stage_totals: dict[str, list[float]] = {}
    selected = []
    for r in runs:
        counts[r.get("status", "?")] = counts.get(r.get("status", "?"), 0) + 1
        if r.get("status") != "failed":
            selected.append(r.get("selected", 0))
            for k, v in (r.get("seconds") or {}).items():
                stage_totals.setdefault(k, []).append(v)
    return {
        "runs": len(runs),
        "status": counts,
        "mean_seconds": {k: round(sum(v) / len(v), 1) for k, v in stage_totals.items()},
        "mean_selected": round(sum(selected) / len(selected), 1) if selected else 0.0,
    }
