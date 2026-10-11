"""Content sources. arXiv is wired up first; HN / GitHub trending come later."""

from .arxiv import Paper, fetch_arxiv

__all__ = ["Paper", "fetch_arxiv"]
