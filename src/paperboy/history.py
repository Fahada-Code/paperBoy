"""Remember which papers earlier digests already delivered.

arXiv's newest-first feed overlaps from day to day (and doesn't move at all on
weekends), so without this a daily digest repeats yesterday's picks. The store
is a small JSON map {base arxiv id: date first delivered} in data_dir, pruned to
a rolling window since old papers fall out of the fetch anyway.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from .sources import Paper

KEEP_DAYS = 30


def base_id(arxiv_id: str) -> str:
    """'2610.12345v2' -> '2610.12345', so a new version doesn't count as new."""
    return arxiv_id.split("v")[0]


def load_seen(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def drop_seen(papers: list[Paper], seen: dict[str, str]) -> list[Paper]:
    return [p for p in papers if base_id(p.arxiv_id) not in seen]


def mark_seen(path: str | Path, papers: list[Paper], today: date | None = None) -> None:
    """Record `papers` as delivered today and prune entries older than KEEP_DAYS."""
    today = today or date.today()
    seen = load_seen(path)
    for p in papers:
        seen.setdefault(base_id(p.arxiv_id), today.isoformat())
    cutoff = (today - timedelta(days=KEEP_DAYS)).isoformat()
    seen = {k: v for k, v in seen.items() if v >= cutoff}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seen, indent=1, sort_keys=True), encoding="utf-8")
