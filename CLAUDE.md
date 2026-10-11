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
- [x] **Phase 2** — `run`: fetch (arXiv Atom API) → embed-rank vs topics (nomic-embed-text, cosine
      to topic centroid, needs search_query:/search_document: prefixes) → dedup → summarize top N →
      PDF (reportlab) or Markdown. Modules: sources/arxiv.py, rank.py, render.py. Verified live.
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
      no API key, no laptop) over hosted-API or local-scheduler. PENDING: confirm first green run
      (trigger via Actions tab); optional: cache Ollama models, structured run metrics/logging.
- [ ] **Phase 5** — PyPI release, README demo (asciinema), CONTRIBUTING, examples.

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
- Commit messages: imperative subject, note the phase. End with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Only commit/push when Fahad asks. Not yet pushed to a GitHub remote.
