"""paperBoy command-line interface.

Phase 1: `summarize` is live and runs against the configured LLM backend
(Ollama by default). `run` (the full fetch->rank->summarize->PDF pipeline) is
still a Phase 2 stub.
"""

from __future__ import annotations

import os
import sys

import httpx
import typer

from . import __version__
from .backends import BackendError, get_backend
from .config import load_config
from .rank import rank_papers
from .render import DigestItem, render_markdown, render_pdf
from .sources import fetch_arxiv

app = typer.Typer(
    help="paperBoy - your terminal AI research digest. Fetch, rank, summarize, deliver.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the installed paperBoy version."""
    typer.echo(f"paperBoy {__version__}")


@app.command()
def doctor(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config file."),
) -> None:
    """Check that the configured LLM backend is reachable and working."""
    cfg = load_config(config)
    typer.echo(f"Backend: {cfg.backend.kind}  model: {cfg.backend.model}  host: {cfg.backend.host}")
    try:
        backend = get_backend(cfg.backend)
        reply = backend.generate("Reply with the single word: pong")
    except BackendError as exc:
        typer.secho(f"FAIL: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho(f"OK - backend responded: {reply[:60]!r}", fg=typer.colors.GREEN)


@app.command()
def summarize(
    text: str = typer.Argument(None, help="Text to summarize. Omit to read from stdin."),
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config file."),
    max_words: int = typer.Option(150, "--max-words", "-w", help="Target summary length."),
) -> None:
    """Summarize text with the configured LLM backend."""
    if text is None:
        text = sys.stdin.read()
    text = text.strip()
    if not text:
        typer.secho("No text provided (argument or stdin).", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    cfg = load_config(config)
    try:
        backend = get_backend(cfg.backend)
        summary = backend.summarize(text, max_words=max_words)
    except BackendError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(summary)


@app.command()
def run(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config file."),
    top_n: int = typer.Option(None, "--top", "-n", help="Override how many papers make the digest."),
    fmt: str = typer.Option(None, "--format", "-f", help="Override output format: pdf | markdown."),
) -> None:
    """Run the full pipeline: fetch -> rank -> summarize -> PDF digest."""
    cfg = load_config(config)
    n = top_n or cfg.top_n
    out_format = (fmt or cfg.output.format).lower()

    try:
        backend = get_backend(cfg.backend)
    except BackendError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # 1. fetch
    if not cfg.arxiv.enabled:
        typer.secho("No sources enabled (arxiv.enabled is false).", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Fetching up to {cfg.arxiv.max_results} papers from arXiv {cfg.arxiv.categories}...")
    try:
        papers = fetch_arxiv(cfg.arxiv.categories, cfg.arxiv.max_results)
    except httpx.HTTPError as exc:
        typer.secho(f"arXiv fetch failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    if not papers:
        typer.secho("No papers returned from arXiv.", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"  fetched {len(papers)} papers.")

    # 2. rank
    if cfg.topics:
        typer.echo(f"Ranking against {len(cfg.topics)} topics (embeddings)...")
    else:
        typer.secho("  no topics configured - using most recent papers.", fg=typer.colors.YELLOW)
    try:
        top = rank_papers(papers, cfg.topics, backend.embed, n, cfg.min_score)
    except BackendError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    if cfg.topics and not top:
        typer.secho(
            f"  no papers cleared the relevance floor (min_score={cfg.min_score}); "
            "writing an empty brief.",
            fg=typer.colors.YELLOW,
        )
    else:
        typer.echo(f"  selected top {len(top)}.")

    # 3. summarize
    typer.echo("Summarizing...")
    items: list[DigestItem] = []
    for i, paper in enumerate(top, 1):
        # Show the relevance score so you can calibrate min_score from real runs.
        tag = f"[{paper.score:.2f}] " if cfg.topics else ""
        typer.echo(f"  [{i}/{len(top)}] {tag}{paper.title[:66]}")
        source = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
        try:
            summary = backend.summarize(source, max_words=70, topic=paper.matched_topic)
        except BackendError as exc:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        items.append(DigestItem(paper=paper, summary=summary))

    # 4. render
    if out_format == "markdown":
        path = render_markdown(items, cfg.output.dir)
    else:
        path = render_pdf(items, cfg.output.dir)
    typer.secho(f"Digest written: {path}", fg=typer.colors.GREEN)


@app.command()
def serve(
    model: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model", "-m"),
    quantize: str = typer.Option("4bit", "--quantize", "-q", help="4bit | none"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
    max_batch: int = typer.Option(8, "--max-batch", help="Dynamic batching: max requests per batch."),
    max_wait_ms: int = typer.Option(10, "--max-wait-ms", help="How long to gather a batch."),
) -> None:
    """Start paperBoy's own quantized, dynamically-batched inference engine (Phase 3)."""
    try:
        import uvicorn

        from .serving.server import create_app
    except ImportError:
        typer.secho(
            'Local serving needs extra deps. Install with:  pip install -e ".[local]"',
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    if host not in ("127.0.0.1", "localhost"):
        typer.secho(
            f"WARNING: binding to {host}. The engine has no auth or rate limiting; "
            "only expose it on a trusted network (or behind Tailscale/a reverse proxy).",
            fg=typer.colors.YELLOW,
            err=True,
        )

    os.environ.update(
        PAPERBOY_MODEL=model,
        PAPERBOY_QUANTIZE=quantize,
        PAPERBOY_MAX_BATCH=str(max_batch),
        PAPERBOY_MAX_WAIT_MS=str(max_wait_ms),
    )
    typer.echo(f"Loading {model} ({quantize})... first load downloads + quantizes the model.")
    uvicorn.run(create_app(), host=host, port=port)


@app.command()
def bench(
    model: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model", "-m"),
    quantize: str = typer.Option("4bit", "--quantize", "-q", help="4bit | none"),
    n: int = typer.Option(32, "--requests", "-n", help="Total requests to fire."),
    concurrency: int = typer.Option(16, "--concurrency", "-c"),
    max_new: int = typer.Option(128, "--max-new", help="Tokens to generate per request."),
    batch_sizes: str = typer.Option("1,8", "--batch-sizes", help="Comma-separated, e.g. 1,4,8."),
) -> None:
    """Benchmark the engine across batch sizes (batching off vs on). Prints tok/s, latency, VRAM."""
    try:
        from .serving.benchmark import run_benchmark
        from .serving.engine import InferenceEngine, Request
    except ImportError:
        typer.secho(
            'Benchmarking needs extra deps. Install with:  pip install -e ".[local]"',
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    sizes = [int(s) for s in batch_sizes.split(",") if s.strip()]
    typer.echo(f"Loading {model} ({quantize})...")
    engine = InferenceEngine(model_id=model, quantize=quantize, max_batch_size=max(sizes))
    typer.echo("Warming up...")
    engine.submit(Request(prompt="hello", max_new_tokens=8, temperature=0.0))

    typer.echo(f"\nBenchmark: {n} requests, concurrency {concurrency}, {max_new} new tokens each\n")
    results = []
    for bs in sizes:
        engine.max_batch_size = bs
        label = f"batch={bs}" + (" (off)" if bs == 1 else "")
        res = run_benchmark(engine, label, n_requests=n, concurrency=concurrency, max_new_tokens=max_new)
        typer.echo("  " + res.as_row())
        results.append(res)

    if len(results) >= 2:
        base, best = results[0], results[-1]
        if base.throughput_tok_s > 0:
            speedup = best.throughput_tok_s / base.throughput_tok_s
            typer.secho(
                f"\n  -> {best.label} vs {base.label}: {speedup:.2f}x throughput",
                fg=typer.colors.GREEN,
            )
    engine.close()


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
