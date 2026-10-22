# CLAUDE.md — paperBoy

Guidance for Claude Code when working in this repo. Keep it current: when a decision is made or
a convention changes, update this file in the same turn.

## What this is

**paperBoy** is a terminal-first, open-source AI research digest. On a schedule it:
**fetch** new research (arXiv first; HN / GitHub trending later) → **rank** it against the user's
configured topics via embeddings → **summarize** the top picks with an LLM → emit a **PDF brief**.

Python + Typer CLI. **No frontend — ever.** It's a tool, not a product.

## Who it's for (and who it's NOT for)

Built by Fahad for himself, to keep up with a fast-moving field, and dogfooded daily. It's open
source so that **developers like him** can `pip install` and run it too. We do **NOT** bend the
design to court users who don't need it (no web UI, no hosted SaaS, no onboarding funnels). Real
mass-user acquisition is a *different, future project's* job. Optimize here for: genuinely useful,
trivially installable, well-documented, honest.

## Non-negotiable principles

- **Terminal-first.** No web frontend. CLI + config file + PDF output.
- **Pluggable LLM backend.** Must work with **Ollama** (free, no GPU — the default), a **local GPU**
  model (quantized, the performance showcase), and a **hosted API**. Low adoption friction is a
  feature: someone with no GPU must be able to run it.
- **The LLM summarizes and explains — it never predicts, advises, or recommends actions.** This is
  a research digest, not an advisor. (Avoids the naive-AI red flag and keeps claims honest.)
- **Honesty in the repo.** Don't fake capabilities. Stubbed commands should say they're stubbed.
  Don't put benchmark numbers in the README until they're measured on real hardware.

## Architecture (the important constraint)

Free-tier cloud (AWS/Supabase) **cannot run an LLM** (no GPU, tiny RAM). So the system SPLITS:
- **Heavy inference** → local (Ollama, or the RTX 4050 quantized path).
- **Orchestration / scheduling / storage** → free tier (GitHub Actions cron + S3 or Supabase).

Don't propose "deploy the model to AWS free tier" — it won't fit.

## Phased roadmap

- [x] **Phase 0** — repo, Typer CLI skeleton (`version`/`summarize`/`run` stubs), config, MIT, gitignore.
- [x] **Phase 1** — pluggable LLM backend (`backends/`: Backend ABC + OllamaBackend + `get_backend`
      factory), config loader, live `summarize` (arg or stdin), `doctor` health check. Verified on
      Ollama `llama3.2:3b` on the RTX 4050 (~4s warm summary; ~75s first-call cold load).
- [x] **Phase 2** — `run`: fetch (arXiv Atom API) → embed-rank vs topics (nomic-embed-text, needs
      search_query:/search_document: prefixes) → dedup → summarize top N → PDF (reportlab) or
      Markdown. Modules: sources/arxiv.py, rank.py, render.py. Verified live.
      QUALITY (2026-10-15): ranking scores each paper by its BEST match across topics (max cosine
      over per-topic embeddings), NOT cosine to the averaged topic centroid - the centroid blurred
      distinct interests and surfaced "random" papers. Added a relevance floor `min_score` (config,
      default 0.0 = off): papers below it are dropped so a quiet day yields a short/empty brief, not
      filler; `run` prints each paper's score for calibration; render.py emits an honest empty-state.
      The ranker also records WHICH topic matched best (Paper.matched_topic); the summarizer is
      grounded in that single topic (base.summarize(topic=...)) and told to admit a weak/tangential
      link rather than invent one - fixes summaries fabricating a catch-all topic connection. The
      matched topic is shown in the digest meta line. Verified live on qwen2.5:7b (Fahad's config,
      min_score 0.60): borderline papers now self-report "connection not explicit / tangential".
      LLM RELEVANCE GATE (2026-10-15, filter.py): after embedding rank, shortlist ~2x top_n
      candidates and have the LLM judge each KEEP/DROP + one-line reason (fail-open on backend
      error; config `llm_filter` default true; `run --filter/--no-filter` override). Embeddings
      are blunt at the margin (on-topic FOLD and off-topic neural-PDE papers both scored 0.64);
      the gate reliably drops the vocabulary-only matches. Guard: a small model sometimes emits
      VERDICT: KEEP with a reason that says "irrelevant" - when they contradict, trust the reason
      and drop. Verified live: gate cut 16 candidates to 6 genuinely-relevant papers. Known: a 7B
      judge is noisy at the margin (occasionally drops a defensible paper like inference-time
      scaling); a larger/hosted judge would be steadier. Modules now: sources/, rank.py, filter.py,
      render.py. Tests: test_filter.py (verdict parsing, contradiction guard, fail-open, noop).
      JUDGE UPGRADE (2026-10-20): (1) decoupled the judge model from the summarizer - config
      `judge_model` (empty = reuse backend.model); cli builds a second backend via
      dataclasses.replace(cfg.backend, model=judge_model) so the gate can run a larger local
      model than summaries. (2) Sharpened the 7B judge with a 3-example few-shot block + a
      THOUGHT reasoning line before the verdict (filter.py _FEWSHOT): Example A keep, B drops a
      vocabulary-only match (neural-PDE), C drops an ADJACENT-but-unlisted subfield (bilevel
      optimization / meta-learning). Verified live: C fixed the judge's invented-adjacency
      over-keeps (bilevel-opt and ARC-AGI papers now correctly dropped). Remaining 7B borderline
      calls (e.g. PEFT-for-brain-encoding kept under "model compression") are the small-model
      ceiling - raise it by pulling a bigger model and setting judge_model, now that it's wired.
      DECISION (2026-10-20): judge stays LOCAL, not a hosted API. This project is a resume/portfolio
      showcase; a self-hosted inference pipeline on Fahad's own GPU is the more impressive (and
      principle-consistent) story than calling a paid API. "hosted" backend remains unimplemented
      by choice. Don't add a hosted/paid dependency unless Fahad asks.
- [x] **Phase 3** — local GPU serving in `serving/` (engine.py = 4-bit bnb NF4 load + dynamic
      batching worker; server.py = FastAPI; benchmark.py = concurrent-load harness). `serve` + `bench`
      CLI cmds, `local` backend. bitsandbytes 4-bit works on Windows NATIVE (no WSL2 needed) with
      torch 2.6.0+cu124. Representative bench (Qwen2.5-1.5B 4-bit, RTX 4050, 24 req @ conc 12, 128 tok):
      batch=1 → 7.4 tok/s / p99 96.7s; batch=8 → 75.9 tok/s / p99 6.7s = ~10x throughput, +50MB VRAM.
      Note: batch=1 slowness is partly bnb per-op overhead at bs=1. Optional follow-up: fp16 comparison.
- [~] **Phase 4** — autonomy (IN PROGRESS). CI done (.github/workflows/ci.yml: pytest 3.11-3.13).
      Scheduled digest done (.github/workflows/digest.yml): cron 13:00 UTC, installs Ollama on the
      runner (CPU), pulls llama3.2:3b + nomic-embed-text, runs `paperboy run`, uploads PDF as an
      artifact. DECISION (2026-10-10): chose GitHub Actions cron + Ollama-on-CPU (free, always-on,
      no API key, no laptop) over hosted-API or local-scheduler. Scheduled digest CONFIRMED GREEN
      (multiple successful cron runs Oct 11-15). HARDENING (2026-10-15): fetch_arxiv now retries
      transient failures (timeouts, conn errors, 5xx) with exponential backoff - a single arXiv
      read-timeout had killed the Oct 14 run. Ollama model blobs now cached in the workflow
      (actions/cache on ~/.ollama/models). Structured run metrics DONE (2026-10-22, see Phase 6 notes: metrics.py, `paperboy stats`).
- [~] **Phase 6** — training data foundation (feedback loop). DECISION (2026-10-20): before any
      fine-tune/re-ranker, capture labels - training's real bottleneck is data, not compute. New
      module feedback.py + `data_dir` config (default ./data, gitignored). `run` now logs every
      relevance-gate verdict (keep AND drop) to data/labels.jsonl as a weak LabelEvent carrying
      title/abstract/topics/matched_topic/score/verdict/reason/selected/model/ts - enough to train
      a re-ranker (score,matched_topic -> verdict) OR a judge fine-tune (title+abstract+topics ->
      keep/drop). `paperboy rate <id> good|bad` (or interactive over the last digest via a
      last_digest.json manifest) adds GOLD human labels; `paperboy labels` prints running counts.
      Verified live: a run logged 16 gate labels (8/8), `rate` added gold labels, `labels` shows
      them. Tests: test_feedback.py (roundtrip, corrupt-line skip, summarize, manifest).
      LEARNED RE-RANKER (2026-10-20, reranker.py): pure-Python logistic regression (no numpy/
      sklearn/GPU, same aesthetic as rank.py) over a fixed 5-dim feature vector of a paper's
      per-topic cosine distribution (max, mean, min, std, gap=max-2nd). Rationale: raw max-cosine
      can't tell a sharp single-topic match from broad vocabulary overlap; the distribution can.
      `paperboy train-reranker` rebuilds features by re-embedding the logged title+abstract vs the
      logged topics (so train/inference features match exactly), labels = keep/good(1) vs
      drop/bad(0) with gold ratings overriding the weak gate label per arxiv_id; saves
      data/reranker.json. rank_papers takes an optional reranker (scores become probabilities, so
      min_score then applies to a 0-1 prob); `run` auto-loads it when use_reranker (config, default
      true) and the file exists, else falls back to cosine. `--reranker/--no-reranker` override.
      Tests: test_reranker.py (feature stats, toy-separation, stable sigmoid, save/load). STATUS:
      infra complete + verified end-to-end, but DO NOT trust it until ~50+ labels (ideally with
      gold ratings) - demo-trained on 16 it overfit and ranked worse than cosine, so the demo
      model was deleted; it auto-activates once properly trained.
      DATA + EVAL (2026-10-20): to unblock training without waiting weeks for daily labels, two
      additions. (1) `paperboy collect-labels --pool N`: fetch N papers, embed-rank (no floor, full
      spectrum), run the gate over all N, log every verdict - bulk-generates weak labels in one
      sitting to DISTILL the gate into the re-ranker. No digest produced. Honest ceiling: distilling
      the 7B gate makes the re-ranker FAST (replaces ~16 LLM calls/run with a microsecond
      classifier), not smarter than the gate - gold `rate` labels are what beat the gate. (2)
      train-reranker now does a deterministic holdout eval (reranker.py train_test_split /
      best_cosine_threshold / accuracy): trains on a split, compares re-ranker@0.5 vs a TUNED
      cosine-threshold baseline on the unseen rest, prints "beats/ties/trails", then retrains on all
      data for the shipped model. Gate-event logging factored into cli._log_gate_events (shared by
      run + collect-labels). Tests: 36 total.
      RESULT (2026-10-20, NEGATIVE - honest): collected 120 gate labels via collect-labels, trained
      the re-ranker, holdout eval (n_test=30): re-ranker 57% vs TUNED cosine baseline 63% (thr 0.61)
      -> re-ranker TRAILS cosine; train acc 66% ~= majority-class baseline (75/120 drop = 62.5%);
      learned weights went backwards (max=-0.49, mean=+0.99). Conclusion: distilling the gate into
      the 5 cosine-distribution features (max/mean/min/std/gap) does NOT beat plain max-cosine - the
      gate's signal lives in the ABSTRACT TEXT it reads, not in the shape of the embedding
      similarities, so these features can't reproduce it. The 5-feature re-ranker as a cosine-only
      model is a dead end. Underperforming model deleted; digests stay on cosine + LLM gate (which
      already produce clean digests). The eval harness + collect-labels + 141-event label store all
      stand and are the real artifacts. NEXT (revised): to actually beat cosine a re-ranker needs
      richer features - the full doc embedding (768-d), which needs ~1000+ labels to not overfit -
      OR skip the re-ranker and invest training effort in a QLoRA fine-tune of the judge/summarizer
      on gold `rate` labels (Phase 3 bitsandbytes stack -> GGUF -> Ollama), with before/after eval.
      Keep collecting gold labels via daily use + `rate`. Keep it LOCAL (see Phase 2 decision).
      QLORA PIPELINE (2026-10-20): built the judge fine-tune infra (did NOT run a real train - 120
      mostly-weak labels would overfit like the re-ranker; gated on more/gold data). New:
      training/dataset.py (pure stdlib: label store -> (prompt->completion) judge instruction
      examples; gold ratings override weak gate labels; no few-shot in the train prompt since the
      model internalizes the task); `paperboy export-train` writes judge.train/val.jsonl with a
      readiness warning; training/train_judge.py = QLoRA script (4-bit NF4 + LoRA + grad
      checkpointing + paged_adamw_8bit, 1.5B base fits 6GB, 7B tight), heavy imports lazy so
      `--dry-run` validates data with no ML deps; `[train]` extra in pyproject (peft/datasets/trl,
      reuses [local] torch/transformers/bnb); docs/finetuning.md (collect -> export -> train ->
      merge -> GGUF -> Ollama Modelfile -> judge_model). Verified: 42 tests pass; export-train gave
      102 train/18 val from current labels; dry-run prints a correct example. DEPLOY GAP (documented):
      filter.py always sends its few-shot prompt; to use a fine-tuned judge with the lighter
      no-few-shot prompt, add a branch that drops few-shot when judge_model is a tuned model.
      NEXT: accumulate 200+ labels incl 20+ gold via daily use + collect-labels + rate, then run the
      QLoRA train and compare tuned-vs-stock judge on the val split; ship only if it clearly wins.
      DEPLOY GAP CLOSED + JUDGE EVAL (2026-10-22): config `judge_tuned` (default false) makes
      filter.py send the exact no-few-shot training prompt (dataset.build_prompt, no system) for a
      fine-tuned judge. `paperboy eval-judge [-m model] [--tuned] [--split val|all] [--gold-only]`
      (training/evaluate.py) runs the real gate over the same val split export-train holds out and
      prints acc/prec/rec/f1 vs majority-class, split weak vs gold. Baseline (n=38 val, 391 events,
      only 2 gold): qwen2.5:7b 92% acc (prec 100%, rec 83%) - inflated, weak labels are mostly its
      own verdicts; llama3.2:3b 74% (rec 56%, drops real papers). Gold labels are the bottleneck.
      GOLD LABEL QUEUE (2026-10-22): `paperboy rate -r N` (feedback.review_queue) spends scarce
      human ratings on the margin: unrated papers ordered by (1) gate runs that disagreed, (2) gate
      vs embedding-score conflicts around the best keep/drop score cut (~0.61 on current data),
      (3) nearest that cut. Shown blind (abstract + url), judge verdict revealed after answering;
      prints judge-vs-you agreement. Next: Fahad rates ~20-30 via `rate -r 30`, then re-run
      `eval-judge --gold-only` for an honest judge baseline.
      RUN METRICS + FAST RANK (2026-10-22): metrics.py logs one line per `run` to data/runs.jsonl
      (status ok|empty|failed + failed_stage, per-stage seconds, fetched/shortlisted/gate kept/
      dropped/selected, score range, models); `paperboy stats` prints recent runs + means. First
      measurement exposed ranking at 158s of a 235s run: Ollama on this laptop costs ~2.3s PER
      embedding request regardless of size. Fix: Backend.embed_many (default loops) + Ollama
      override using batched /api/embed (32/request, identical vectors - cosine 1.0 vs legacy,
      404 -> falls back to per-item). rank_papers takes embed_many. Rank 158s -> 8s, run 235s -> 74s.
      The gate (~4s/paper, 12 papers) is now the largest stage.
      PARALLEL LLM CALLS (2026-10-22): config `llm_workers` (default 4) sends gate verdicts and
      summaries concurrently (ThreadPoolExecutor, ordered map so progress + digest order are
      unchanged). Ollama batches them on the GPU: gate on 12 papers 49s -> 22s, identical verdicts
      at temperature 0. Full 8-paper digest now ~63s (was ~235s for 3 papers before batching).
      eval-judge uses it too. Observed quality gap (not fixed): qwen2.5:7b summaries still overstate
      tangential topic links (e.g. a KAN paper "enhancing model compression") despite the prompt.
      NO REPEATS (2026-10-22): history.py keeps data/seen.json {base arxiv id: first delivered
      date}, 30-day window. `run` drops already-delivered papers after fetch (metric skipped_seen;
      `--include-seen` to disable) and marks the digest's picks after render. arXiv doesn't move on
      weekends, so without this the cron repeated Friday. digest.yml now caches data/ between runs
      (key per run_id, restore by prefix) - UNVERIFIED until the next cron run on GitHub.
      Also: tests/test_cli.py end-to-end CLI tests (fake backend + fake arXiv) for run/stats/
      rate -r/eval-judge/doctor; `doctor` now checks summary, embedding and judge models separately.
      README brought current (gate, feedback/eval commands, hosted marked not implemented).
- [ ] **Phase 5** — PyPI release, README demo (asciinema), CONTRIBUTING, examples.
      DECISION (2026-10-15): Fahad wants paperBoy great FOR HIM first; PyPI/pip release DEFERRED to
      the end (prioritize digest quality over distribution). README/CONTRIBUTING/examples drafted
      (uncommitted in tree). asciinema demo = manual step for Fahad (needs his terminal+GPU+models);
      script at docs/demo.md. Next quality steps after ranking: tune summarization prompt to explain
      relevance; add a feedback log (mark good/bad) - the data that would later justify any fine-tune.

Work one phase at a time. Each phase must end in a runnable, demoable win. Guard scope ruthlessly.

## Project layout

```
paperBoy/
├── pyproject.toml          # installable package; `paperboy` console entry point
├── src/paperboy/
│   ├── __init__.py         # __version__
│   └── cli.py              # Typer app: version / summarize / run
├── config.example.yaml     # topics, sources, backend, output (copy to config.yaml)
├── CLAUDE.md               # this file
├── README.md               # public pitch + roadmap
└── LICENSE                 # MIT
```

`config.yaml` is gitignored (user-local). Add new modules under `src/paperboy/` (e.g. `backends/`,
`sources/`, `rank.py`, `render.py`) as phases land.

## Dev environment & commands

- **OS:** Windows 11. Shell examples below are PowerShell.
- **Python:** 3.13. **No `uv`** installed — use the stdlib `venv` + `pip`.
- **Location caveat:** repo lives under OneDrive (`...\OneDrive\Desktop\GithubDesktop\paperBoy`).
  `.venv` is gitignored and disposable — recreate it rather than moving it. If git throws file-lock
  errors, pause OneDrive sync during the operation.
- **GPU (Phase 3):** RTX 4050 Laptop, ~6 GB VRAM, CUDA-capable (Windows WMI misreports as 4 GB).

```powershell
# one-time setup (or after moving the repo)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .

# run the CLI
.\.venv\Scripts\paperboy.exe --help
.\.venv\Scripts\paperboy.exe version
```

## Conventions

- Keep dependencies lean; justify each new one. Prefer stdlib + well-known libraries.
- Type-hint public functions; keep the CLI thin and push logic into importable modules (testable).
- Avoid non-ASCII in strings printed to the console (Windows code-page renders em-dashes as `?`).
- Commits: small and frequent, one logical change each. Messages super short, simple, human
  sounding, no em dashes, no long bodies. NO Co-Authored-By / Claude attribution - commits must
  appear solely from Fahad (his git identity).
- Only commit/push when Fahad asks. Remote: github.com/Fahada-Code/paperBoy (main tracks origin/main).
