"""paperBoy command-line interface.

Phase 1: `summarize` is live and runs against the configured LLM backend
(Ollama by default). `run` (the full fetch->rank->summarize->PDF pipeline) is
still a Phase 2 stub.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import typer

from . import __version__
from .backends import BackendError, get_backend
from .config import load_config
from .feedback import (
    LabelEvent,
    append_events,
    load_events,
    read_manifest,
    summarize_events,
    write_manifest,
)
from .filter import filter_relevant
from .rank import DOC_PREFIX, QUERY_PREFIX, _cosine, rank_papers
from .reranker import (
    Reranker,
    accuracy,
    best_cosine_threshold,
    extract_features,
    train_reranker,
    train_test_split,
)
from .render import DigestItem, render_markdown, render_pdf

LABELS_FILE = "labels.jsonl"
MANIFEST_FILE = "last_digest.json"
RERANKER_FILE = "reranker.json"


def _log_gate_events(cfg, judge_name: str, gate_log: list, selected_ids: set[str]) -> None:
    """Append gate verdicts to the label store as weak training labels (both keeps
    and drops - a classifier needs negatives). Best-effort: never fail over logging."""
    if not gate_log:
        return
    path = Path(cfg.data_dir) / LABELS_FILE
    events = [
        LabelEvent(
            kind="gate",
            arxiv_id=p.arxiv_id,
            title=p.title,
            abstract=p.abstract,
            url=p.url,
            topics=cfg.topics,
            matched_topic=p.matched_topic,
            score=round(p.score, 4),
            verdict="keep" if keep else "drop",
            reason=reason,
            selected=p.arxiv_id in selected_ids,
            model=judge_name,
        )
        for p, keep, reason in gate_log
    ]
    try:
        append_events(path, events)
        typer.echo(f"  logged {len(events)} gate labels to {path}")
    except OSError as exc:
        typer.secho(f"  (could not write labels: {exc})", fg=typer.colors.YELLOW, err=True)
from .sources import fetch_arxiv


def _judge(cfg, backend) -> tuple:
    """(generate fn, model name) for the relevance gate. The judge can run a
    different (e.g. larger or fine-tuned) local model than the summarizer."""
    if not cfg.judge_model or cfg.judge_model == cfg.backend.model:
        return backend.generate, cfg.backend.model
    try:
        return get_backend(replace(cfg.backend, model=cfg.judge_model)).generate, cfg.judge_model
    except BackendError as exc:
        typer.secho(f"Error building judge backend: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


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
    llm_filter: bool = typer.Option(
        None, "--filter/--no-filter", help="LLM relevance gate after embedding rank (overrides config)."
    ),
    reranker_opt: bool = typer.Option(
        None, "--reranker/--no-reranker", help="Use the learned re-ranker if trained (overrides config)."
    ),
) -> None:
    """Run the full pipeline: fetch -> rank -> summarize -> PDF digest."""
    cfg = load_config(config)
    n = top_n or cfg.top_n
    out_format = (fmt or cfg.output.format).lower()
    use_filter = cfg.llm_filter if llm_filter is None else llm_filter
    use_reranker = cfg.use_reranker if reranker_opt is None else reranker_opt

    # Load the learned re-ranker if enabled and trained; otherwise fall back to
    # raw embedding cosine (a bad/corrupt model file must never fail a run).
    reranker: Reranker | None = None
    if use_reranker and cfg.topics:
        rpath = Path(cfg.data_dir) / RERANKER_FILE
        if rpath.exists():
            try:
                reranker = Reranker.load(rpath)
            except (OSError, ValueError, KeyError) as exc:
                typer.secho(f"  (ignoring unreadable re-ranker: {exc})", fg=typer.colors.YELLOW, err=True)

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

    # 2. rank (embeddings). With the LLM gate on, shortlist more candidates than
    # we need, then let the gate trim them back down to the top N.
    if cfg.topics and reranker is not None:
        typer.echo(f"Ranking against {len(cfg.topics)} topics (learned re-ranker, trained on {reranker.n_train})...")
    elif cfg.topics:
        typer.echo(f"Ranking against {len(cfg.topics)} topics (embeddings)...")
    else:
        typer.secho("  no topics configured - using most recent papers.", fg=typer.colors.YELLOW)
    gate_on = use_filter and bool(cfg.topics)
    shortlist_n = min(max(n * 2, 12), cfg.arxiv.max_results) if gate_on else n
    try:
        ranked = rank_papers(papers, cfg.topics, backend.embed, shortlist_n, cfg.min_score, reranker)
    except BackendError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # 2b. LLM relevance gate: keep/drop each shortlisted paper with a reason.
    judge_name = cfg.backend.model
    gate_log: list[tuple[object, bool, str]] = []
    if gate_on and ranked:
        judge_generate, judge_name = _judge(cfg, backend)
        typer.echo(f"Filtering {len(ranked)} candidates with the LLM relevance gate ({judge_name})...")

        def _show(p, keep: bool, reason: str) -> None:
            mark = "keep" if keep else "drop"
            typer.echo(f"  [{mark}] [{p.score:.2f}] {p.title[:55]} - {reason[:45]}")
            gate_log.append((p, keep, reason))

        ranked = filter_relevant(
            ranked, cfg.topics, judge_generate, on_verdict=_show, tuned=cfg.judge_tuned
        )

    top = ranked[:n]

    # 2c. Capture gate verdicts as weak training labels.
    _log_gate_events(cfg, judge_name, gate_log, selected_ids={p.arxiv_id for p in top})

    if cfg.topics and not top:
        gate_note = " and LLM gate" if gate_on else ""
        typer.secho(
            f"  nothing cleared the relevance floor (min_score={cfg.min_score}){gate_note}; "
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

    # Record what made this digest so `paperboy rate` can ask about it later.
    if top:
        try:
            write_manifest(
                Path(cfg.data_dir) / MANIFEST_FILE,
                [{"arxiv_id": p.arxiv_id, "title": p.title, "url": p.url} for p in top],
            )
        except OSError:
            pass  # manifest is a convenience; never fail a run over it


@app.command()
def rate(
    arxiv_id: str = typer.Argument(None, help="Paper id to rate. Omit for interactive mode over the last digest."),
    verdict: str = typer.Argument(None, help="good | bad (required when arxiv_id is given)."),
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config file."),
    note: str = typer.Option("", "--note", help="Optional note stored with the rating."),
) -> None:
    """Record your good/bad judgment on digest papers - the gold labels for training.

    `paperboy rate 2610.12345 good` rates one paper; `paperboy rate` with no args
    walks the most recent digest and asks about each paper.
    """
    cfg = load_config(config)
    labels_path = Path(cfg.data_dir) / LABELS_FILE

    def _record(aid: str, v: str, title: str = "", url: str = "", reason: str = "") -> None:
        append_events(
            labels_path,
            [LabelEvent(kind="rating", arxiv_id=aid, title=title, url=url, verdict=v, reason=reason)],
        )

    # Direct, scriptable form.
    if arxiv_id is not None:
        v = (verdict or "").lower()
        if v not in ("good", "bad"):
            typer.secho("Verdict must be 'good' or 'bad'.", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        _record(arxiv_id, v, reason=note)
        typer.secho(f"Recorded: {arxiv_id} -> {v}", fg=typer.colors.GREEN)
        return

    # Interactive form over the last digest.
    manifest = read_manifest(Path(cfg.data_dir) / MANIFEST_FILE)
    if not manifest:
        typer.secho("No recent digest found. Run `paperboy run` first.", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Rating {len(manifest)} papers from the last digest ([g]ood / [b]ad / [s]kip):")
    rated = 0
    for item in manifest:
        choice = typer.prompt(f"  {item['title'][:70]}", default="s").strip().lower()
        v = {"g": "good", "good": "good", "b": "bad", "bad": "bad"}.get(choice)
        if v is None:
            continue
        _record(item["arxiv_id"], v, title=item.get("title", ""), url=item.get("url", ""))
        rated += 1
    typer.secho(f"Recorded {rated} ratings to {labels_path}", fg=typer.colors.GREEN)


@app.command()
def labels(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config file."),
) -> None:
    """Show how many training labels have accumulated in the label store."""
    cfg = load_config(config)
    labels_path = Path(cfg.data_dir) / LABELS_FILE
    counts = summarize_events(load_events(labels_path))
    if counts["total"] == 0:
        typer.secho(f"No labels yet at {labels_path}. Run `paperboy run` to start logging.", fg=typer.colors.YELLOW)
        return
    typer.echo(f"Label store: {labels_path}")
    typer.echo(f"  total events:   {counts['total']}")
    typer.echo(f"  gate keep/drop: {counts['gate_keep']} / {counts['gate_drop']}  (weak labels)")
    typer.echo(f"  rated good/bad: {counts['rating_good']} / {counts['rating_bad']}  (gold labels)")


@app.command("export-train")
def export_train_cmd(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config file."),
    val_frac: float = typer.Option(0.15, "--val-frac", help="Fraction held out for validation."),
    out_dir: str = typer.Option("", "--out", help="Output dir (default: <data_dir>/train)."),
) -> None:
    """Export the label store as a QLoRA instruction dataset for the judge fine-tune.

    Writes judge.train.jsonl / judge.val.jsonl of (prompt -> completion) examples,
    gold ratings overriding weak gate labels. Feed these to
    `python -m paperboy.training.train_judge`.
    """
    from .training.dataset import build_judge_examples, split_examples, write_jsonl

    cfg = load_config(config)
    events = load_events(Path(cfg.data_dir) / LABELS_FILE)
    if not events:
        typer.secho("No labels yet. Run `paperboy run` / `collect-labels` / `rate` first.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    examples = build_judge_examples(events)
    if not examples:
        typer.secho("No usable examples (need gate labels carrying topics + text).", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    train, val = split_examples(examples, val_frac)
    out = Path(out_dir) if out_dir else Path(cfg.data_dir) / "train"
    write_jsonl(out / "judge.train.jsonl", train)
    write_jsonl(out / "judge.val.jsonl", val)

    pos = sum(e["label"] for e in examples)
    gold = sum(1 for e in examples if e["source"] == "rating")
    typer.secho(f"Exported {len(examples)} examples (relevant {pos} / not {len(examples) - pos}; gold {gold}).", fg=typer.colors.GREEN)
    typer.echo(f"  train: {out / 'judge.train.jsonl'} ({len(train)})")
    typer.echo(f"  val:   {out / 'judge.val.jsonl'} ({len(val)})")
    if len(examples) < 200 or gold < 20:
        typer.secho(
            "  note: this is light for an LLM fine-tune (aim for 200+, with 20+ gold `rate` labels). "
            "Validate the data now with:\n"
            f"    python -m paperboy.training.train_judge --train {out / 'judge.train.jsonl'} --dry-run",
            fg=typer.colors.YELLOW,
        )


@app.command("collect-labels")
def collect_labels_cmd(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config file."),
    pool: int = typer.Option(120, "--pool", help="How many papers to judge for labels."),
) -> None:
    """Run the relevance gate over a large pool purely to generate training labels.

    No digest is produced - it fetches a big batch, ranks by embedding (no floor,
    so the full relevant/irrelevant spectrum is kept), and makes one gate LLM call
    per paper, appending every verdict to the label store. A fast way to bootstrap
    the re-ranker by distilling the gate, instead of waiting for daily runs.
    """
    cfg = load_config(config)
    if not cfg.topics:
        typer.secho("Need topics in config to judge relevance.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    try:
        backend = get_backend(cfg.backend)
    except BackendError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Fetching up to {pool} papers from arXiv {cfg.arxiv.categories}...")
    try:
        papers = fetch_arxiv(cfg.arxiv.categories, pool)
    except httpx.HTTPError as exc:
        typer.secho(f"arXiv fetch failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    try:
        ranked = rank_papers(papers, cfg.topics, backend.embed, pool, 0.0, None)
    except BackendError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    if not ranked:
        typer.secho("No papers to judge.", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)

    judge_generate, judge_name = _judge(cfg, backend)
    typer.echo(f"Judging {len(ranked)} papers with the gate ({judge_name}) - one LLM call each...")

    gate_log: list[tuple[object, bool, str]] = []

    def _show(p, keep: bool, reason: str) -> None:
        typer.echo(f"  [{'keep' if keep else 'drop'}] [{p.score:.2f}] {p.title[:55]}")
        gate_log.append((p, keep, reason))

    filter_relevant(ranked, cfg.topics, judge_generate, on_verdict=_show, tuned=cfg.judge_tuned)
    _log_gate_events(cfg, judge_name, gate_log, selected_ids=set())
    keeps = sum(1 for _, keep, _ in gate_log if keep)
    typer.secho(
        f"Collected {len(gate_log)} labels ({keeps} keep / {len(gate_log) - keeps} drop). "
        "Run `paperboy train-reranker` next.",
        fg=typer.colors.GREEN,
    )


@app.command("train-reranker")
def train_reranker_cmd(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config file."),
    min_labels: int = typer.Option(20, "--min-labels", help="Refuse to train below this many examples."),
    test_frac: float = typer.Option(0.25, "--test-frac", help="Fraction held out for the eval."),
    no_eval: bool = typer.Option(False, "--no-eval", help="Skip the holdout eval vs the cosine baseline."),
) -> None:
    """Train the learned re-ranker on accumulated labels (re-embeds stored papers).

    Builds one example per labeled paper: features are the distribution of its
    per-topic cosine similarities, the label is keep/good (1) vs drop/bad (0). A
    human rating overrides the weak gate label for the same paper (gold wins).
    Saves to data_dir/reranker.json, which `run` then uses automatically.
    """
    cfg = load_config(config)
    events = load_events(Path(cfg.data_dir) / LABELS_FILE)
    if not events:
        typer.secho("No labels yet. Run `paperboy run` (and `paperboy rate`) first.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Gate events carry the features we need (abstract + topics); the last gate
    # event per paper wins, then a human rating overrides its label if present.
    gate_by_id: dict[str, dict] = {}
    rating_by_id: dict[str, dict] = {}
    for e in events:
        if e.get("kind") == "gate":
            gate_by_id[e["arxiv_id"]] = e
        elif e.get("kind") == "rating":
            rating_by_id[e["arxiv_id"]] = e

    labeled: list[tuple[dict, int]] = []
    for aid, g in gate_by_id.items():
        if not g.get("topics") or not (g.get("abstract") or g.get("title")):
            continue  # can't rebuild features without topics + text
        label = 1 if g.get("verdict") == "keep" else 0
        if aid in rating_by_id:
            label = 1 if rating_by_id[aid].get("verdict") == "good" else 0
        labeled.append((g, label))

    pos = sum(lbl for _, lbl in labeled)
    typer.echo(f"Usable labeled papers: {len(labeled)}  (relevant {pos} / not {len(labeled) - pos})")
    if len(labeled) < min_labels:
        typer.secho(
            f"Need at least {min_labels} to train (have {len(labeled)}). Keep running/rating; "
            "lower --min-labels to force.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)
    if pos == 0 or pos == len(labeled):
        typer.secho("Labels are all one class - can't learn a boundary yet.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        backend = get_backend(cfg.backend)
    except BackendError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo("Re-embedding labeled papers to build features...")
    topic_cache: dict[str, list[float]] = {}

    def _topic_vec(topic: str) -> list[float]:
        if topic not in topic_cache:
            topic_cache[topic] = backend.embed(topic, prefix=QUERY_PREFIX)
        return topic_cache[topic]

    examples: list[tuple[list[float], int]] = []
    try:
        for g, label in labeled:
            doc_vec = backend.embed(f"{g.get('title', '')}\n\n{g.get('abstract', '')}", prefix=DOC_PREFIX)
            sims = [_cosine(doc_vec, _topic_vec(t)) for t in g["topics"]]
            examples.append((extract_features(sims), label))
    except BackendError as exc:
        typer.secho(f"Error embedding: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Holdout eval: train on a split, compare to the tuned cosine baseline on the
    # unseen rest, then retrain on everything for the shipped model.
    if not no_eval and len(examples) >= 8:
        tr, te = train_test_split(examples, test_frac)
        ev = train_reranker(tr)
        thr = best_cosine_threshold(tr)
        r_acc = accuracy(lambda f: ev.score(f) >= 0.5, te)
        c_acc = accuracy(lambda f, t=thr: f[0] >= t, te)
        verdict = "beats" if r_acc > c_acc else ("ties" if r_acc == c_acc else "trails")
        typer.echo(
            f"Holdout eval (n_test={len(te)}): re-ranker {r_acc:.0%}  vs  cosine baseline "
            f"{c_acc:.0%} (thr {thr:.2f})  -> re-ranker {verdict} cosine"
        )

    model = train_reranker(examples)
    correct = sum(1 for feats, lbl in examples if (model.score(feats) >= 0.5) == bool(lbl))
    acc = correct / len(examples)
    out = Path(cfg.data_dir) / RERANKER_FILE
    model.save(out)

    typer.secho(f"Trained re-ranker on {model.n_train} examples; train accuracy {acc:.0%}.", fg=typer.colors.GREEN)
    weights = ", ".join(f"{name}={w:+.2f}" for name, w in zip(("max", "mean", "min", "std", "gap"), model.weights))
    typer.echo(f"  weights: {weights}")
    typer.echo(f"  saved: {out}  (run will use it automatically)")
    if model.n_train < 60:
        typer.secho("  note: small training set - expect overfitting; keep rating to improve it.", fg=typer.colors.YELLOW)


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
