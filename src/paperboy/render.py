"""Render a digest to PDF (or Markdown).

Each digest item is a (Paper, summary) pair. The PDF is deliberately plain and
readable — this is a brief, not a brochure.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from .sources import Paper


@dataclass
class DigestItem:
    paper: Paper
    summary: str


def _esc(text: str) -> str:
    return html.escape(text or "")


def render_markdown(items: list[DigestItem], out_dir: str | Path, day: date | None = None) -> Path:
    day = day or date.today()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"digest-{day.isoformat()}.md"

    lines = [f"# paperBoy digest - {day.isoformat()}", ""]
    if not items:
        lines.append("No papers cleared the relevance threshold today.")
        lines.append("")
    for i, item in enumerate(items, 1):
        p = item.paper
        lines.append(f"## {i}. {p.title}")
        authors = ", ".join(p.authors[:4]) + (" et al." if len(p.authors) > 4 else "")
        meta = f"*{authors}* - {p.primary_category} - [{p.arxiv_id}]({p.url})"
        if p.score:
            meta += f" - relevance {p.score:.2f}"
            if p.matched_topic:
                meta += f" ({p.matched_topic})"
        lines.append(meta)
        lines.append("")
        lines.append(item.summary)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def render_pdf(items: list[DigestItem], out_dir: str | Path, day: date | None = None) -> Path:
    day = day or date.today()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"digest-{day.isoformat()}.pdf"

    styles = getSampleStyleSheet()
    h_title = ParagraphStyle(
        "DigestTitle", parent=styles["Title"], fontSize=20, spaceAfter=2, alignment=TA_LEFT
    )
    h_sub = ParagraphStyle("DigestSub", parent=styles["Normal"], fontSize=9, textColor="#666666")
    h_item = ParagraphStyle(
        "ItemTitle", parent=styles["Heading2"], fontSize=12, spaceBefore=14, spaceAfter=2
    )
    h_meta = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=8, textColor="#666666")
    h_body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14, spaceBefore=4)

    doc = SimpleDocTemplate(
        str(path),
        pagesize=LETTER,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title=f"paperBoy digest {day.isoformat()}",
    )

    flow = [
        Paragraph("paperBoy digest", h_title),
        Paragraph(f"{day.isoformat()} &middot; {len(items)} papers", h_sub),
    ]
    if not items:
        flow.append(Spacer(1, 12))
        flow.append(
            Paragraph("No papers cleared the relevance threshold today.", h_body)
        )
    for i, item in enumerate(items, 1):
        p = item.paper
        authors = ", ".join(p.authors[:4]) + (" et al." if len(p.authors) > 4 else "")
        meta = f"{_esc(authors)} &middot; {_esc(p.primary_category)} &middot; {_esc(p.arxiv_id)}"
        if p.score:
            meta += f" &middot; relevance {p.score:.2f}"
            if p.matched_topic:
                meta += f" ({_esc(p.matched_topic)})"
        flow.append(Paragraph(f"{i}. {_esc(p.title)}", h_item))
        flow.append(Paragraph(meta, h_meta))
        flow.append(Paragraph(_esc(item.summary), h_body))
        flow.append(Spacer(1, 2))

    doc.build(flow)
    return path
