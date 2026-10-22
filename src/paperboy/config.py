"""Configuration loading for paperBoy.

Reads a YAML config (see config.example.yaml). Everything has a sensible default,
so paperBoy runs even with no config.yaml present — it just uses the Ollama
backend with a small default model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BackendConfig:
    """Which LLM serves requests, and how to reach it."""

    kind: str = "ollama"  # ollama | local | hosted
    model: str = "llama3.2:3b"
    embed_model: str = "nomic-embed-text"  # used for relevance ranking
    host: str = "http://localhost:11434"  # used by ollama/hosted HTTP backends
    timeout: int = 300  # seconds; generous so the first-call cold model load doesn't time out


@dataclass
class ArxivConfig:
    enabled: bool = True
    categories: list[str] = field(default_factory=lambda: ["cs.LG", "cs.CL"])
    max_results: int = 60  # fetched per run, before ranking


@dataclass
class OutputConfig:
    format: str = "pdf"  # pdf | markdown
    dir: str = "./digests"


@dataclass
class Config:
    topics: list[str] = field(default_factory=list)
    top_n: int = 8
    # Relevance floor: papers scoring below this (max cosine to any topic) are
    # dropped. 0.0 keeps everything (off) - calibrate it by running once and
    # reading the per-paper scores paperboy prints, then raise it.
    min_score: float = 0.0
    # After embedding rank, have the LLM judge each shortlisted paper keep/drop.
    # Embeddings are blunt at the margin; this gate catches papers that merely
    # share ML vocabulary. Costs a few extra (short) LLM calls per run.
    llm_filter: bool = True
    # Model for the relevance judge. Empty = use the summary model (backend.model).
    # The judge wants a sharp reasoner; the summarizer can be cheaper. Point this at
    # a larger local model (same Ollama host) to strengthen the gate independently.
    judge_model: str = ""
    # Set true when judge_model is a fine-tuned judge (docs/finetuning.md): the gate
    # then sends the short no-few-shot prompt it was trained on instead of the
    # few-shot prompt the stock models need.
    judge_tuned: bool = False
    # How many LLM requests (gate verdicts, summaries) to send at once. Ollama
    # batches them on the GPU (~2x faster at 2-4, same verdicts). 1 = one at a time.
    llm_workers: int = 4
    # Use the trained learned re-ranker (data_dir/reranker.json) for scoring when
    # it exists. Falls back to raw embedding cosine when off or untrained. Train it
    # with `paperboy train-reranker` once enough labels have accumulated.
    use_reranker: bool = True
    # Where labels/feedback accumulate (JSONL + last-digest manifest). User-local
    # and gitignored, like digests/. This is the training data foundation.
    data_dir: str = "./data"
    backend: BackendConfig = field(default_factory=BackendConfig)
    arxiv: ArxivConfig = field(default_factory=ArxivConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    raw: dict[str, Any] = field(default_factory=dict)  # full parsed YAML, for later phases


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load config from `path`, falling back to defaults for anything missing.

    A missing file is fine (returns all-defaults). This keeps things frictionless:
    the CLI works out of the box against a local Ollama.
    """
    path = Path(path)
    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    backend_data = data.get("backend", {}) or {}
    backend = BackendConfig(
        kind=backend_data.get("kind", BackendConfig.kind),
        model=backend_data.get("model", BackendConfig.model),
        embed_model=backend_data.get("embed_model", BackendConfig.embed_model),
        host=backend_data.get("host", BackendConfig.host),
        timeout=int(backend_data.get("timeout", BackendConfig.timeout)),
    )

    sources = data.get("sources", {}) or {}
    arxiv_data = sources.get("arxiv", {}) or {}
    arxiv_defaults = ArxivConfig()
    arxiv = ArxivConfig(
        enabled=bool(arxiv_data.get("enabled", arxiv_defaults.enabled)),
        categories=list(arxiv_data.get("categories", arxiv_defaults.categories)),
        max_results=int(arxiv_data.get("max_results", arxiv_defaults.max_results)),
    )

    output_data = data.get("output", {}) or {}
    output = OutputConfig(
        format=output_data.get("format", OutputConfig.format),
        dir=output_data.get("dir", OutputConfig.dir),
    )

    return Config(
        topics=list(data.get("topics", []) or []),
        top_n=int(data.get("top_n", 8)),
        min_score=float(data.get("min_score", 0.0)),
        llm_filter=bool(data.get("llm_filter", True)),
        judge_model=str(data.get("judge_model", "") or ""),
        judge_tuned=bool(data.get("judge_tuned", False)),
        llm_workers=max(1, int(data.get("llm_workers", 4))),
        use_reranker=bool(data.get("use_reranker", True)),
        data_dir=str(data.get("data_dir", "./data") or "./data"),
        backend=backend,
        arxiv=arxiv,
        output=output,
        raw=data,
    )
