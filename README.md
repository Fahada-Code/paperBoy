# paperBoy 🗞️

[![CI](https://github.com/Fahada-Code/paperBoy/actions/workflows/ci.yml/badge.svg)](https://github.com/Fahada-Code/paperBoy/actions/workflows/ci.yml)

**A terminal-first AI research digest.** On a schedule it **fetches** new research,
**ranks** it against the topics you actually care about, **summarizes** the top picks
with an LLM, and hands you a clean **PDF brief** — no feeds to scroll, no frontend to babysit.

## What it does

paperBoy is a small Python/Typer CLI that runs one pipeline:

```
fetch (arXiv)  ->  rank (embeddings)  ->  summarize (LLM)  ->  PDF / Markdown brief
```

1. **Fetch** — pulls recent papers from the arXiv Atom API for the categories you choose
   (HN / GitHub trending are planned).
2. **Rank** — embeds each paper and scores it by cosine similarity to the centroid of your
   configured topics, then dedups.
3. **Summarize** — sends the top N abstracts to a pluggable LLM backend for short,
   plain-language summaries.
4. **Deliver** — writes a dated PDF (or Markdown) digest to a local folder.

## Why it exists

Staying current on ML/systems research is a part-time job. paperBoy turns it into a
two-minute read. It is deliberately **terminal-first** and **self-hostable**: run it on a
free Ollama model with no GPU, or point it at a local quantized model on your own GPU for
the fast path. It was built by its author to dogfood daily, and open-sourced so other devs
can run the same thing.

## Design guardrails

These are non-negotiable, not aspirations:

- **Terminal-first. No web UI, ever.** CLI + a config file + a PDF. It is a tool, not a
  product, and not a hosted SaaS.
- **The LLM summarizes and explains — it never predicts, advises, or recommends actions.**
  This is a research digest, not an advisor. Honest summaries over confident guesses.
- **Pluggable backends with low adoption friction.** If you have no GPU, you can still run it.

## Install

paperBoy is not on PyPI yet (`pip install paperboy` is **coming soon**). For now, install
from source into a virtual environment.

Windows (PowerShell):

```powershell
git clone https://github.com/Fahada-Code/paperBoy.git
cd paperBoy
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\paperboy.exe --help
```

macOS / Linux:

```bash
git clone https://github.com/Fahada-Code/paperBoy.git
cd paperBoy
python -m venv .venv
.venv/bin/pip install -e .
paperboy --help
```

Requires Python 3.11+. The commands below use the plain `paperboy` form (for an activated
venv or installed package); on Windows you can substitute `.\.venv\Scripts\paperboy.exe`.

## Quickstart

The default backend is [Ollama](https://ollama.com/) — free and runs on CPU, no GPU required.

```bash
# 1. install Ollama, then pull the default models:
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 2. copy the config and edit your topics:
cp config.example.yaml config.yaml

# 3. check the backend is reachable:
paperboy doctor

# 4. generate today's digest -> ./digests/*.pdf
paperboy run
```

More:

```bash
paperboy run --top 5 --format markdown   # fewer picks, Markdown instead of PDF
paperboy summarize "long text here..."   # summarize ad-hoc text (argument or stdin)
paperboy version
```

Everything is driven by `config.yaml` — topics, arXiv categories, how many picks make the
digest, which backend, and the output format. See [`config.example.yaml`](config.example.yaml).

## Backends

Low adoption friction is a feature. paperBoy talks to any of three backends; pick one in
`config.yaml` under `backend.kind`:

| Backend  | `kind`   | Needs a GPU? | Use it for |
|----------|----------|--------------|------------|
| Ollama   | `ollama` | No (CPU OK)  | The default. Easiest install, runs anywhere. |
| Local GPU| `local`  | Yes          | paperBoy's own 4-bit quantized, dynamically-batched engine — the performance path. |
| Hosted   | `hosted` | No           | A cloud API, when you'd rather not run a model yourself. |

### Local GPU serving

The `local` backend ships its own inference engine: a 4-bit quantized model (bitsandbytes
NF4) behind a FastAPI server with a **dynamic batching** scheduler that groups concurrent
requests into a single GPU pass. Install the extras and run it:

```bash
pip install -e ".[local]"            # torch (CUDA), transformers, bitsandbytes, ...
paperboy serve                       # starts the engine on 127.0.0.1:8000
paperboy bench --batch-sizes 1,8     # measure batching's effect
```

**Benchmark** — Qwen2.5-1.5B-Instruct (4-bit NF4) on an **RTX 4050 Laptop (6 GB)**,
24-request burst at concurrency 12, 128 new tokens each (one representative run):

| config                 | throughput    | p99 latency | VRAM    |
|------------------------|--------------:|------------:|--------:|
| batching off (batch=1) | 7.4 tok/s     | 96.7 s      | ~1.1 GB |
| batching on (batch=8)  | **75.9 tok/s**| **6.7 s**   | ~1.1 GB (+50 MB) |

That is roughly **10x throughput** for about +50 MB VRAM. Reproduce with `paperboy bench`.
(Part of batch=1's slowness is bitsandbytes' per-op overhead at batch size 1, which batching
amortizes.) These are the only measured numbers we publish; anything not listed here hasn't
been benchmarked.

## How scheduling works

Free-tier cloud (AWS / Supabase) can't run an LLM — no GPU, tiny RAM — so paperBoy splits
responsibilities:

- **Heavy inference stays local.** On the scheduled path it runs on the CI runner's CPU via
  Ollama; on your own machine it can use the local GPU engine.
- **Orchestration / scheduling runs on the free tier.** A GitHub Actions cron workflow
  (`.github/workflows/digest.yml`) installs Ollama on the runner, pulls the models, runs
  `paperboy run`, and uploads the resulting PDF as a build artifact.

The result: a daily digest with no always-on server, no API key, and no laptop that has to
stay awake. (Don't try to "deploy the model to a free-tier VM" — it won't fit.)

## Roadmap

- [x] **Phase 0** — repo, Typer CLI skeleton, config
- [x] **Phase 1** — pluggable LLM backend (Ollama now; local GPU / hosted), `summarize`, `doctor`
- [x] **Phase 2** — fetch (arXiv) -> embed-rank -> summarize -> PDF/Markdown (`run`)
- [x] **Phase 3** — local GPU serving: 4-bit quantization + dynamic batching + benchmark (`serve`, `bench`)
- [~] **Phase 4** — autonomy: CI (pytest 3.11-3.13) + scheduled digest via GitHub Actions cron
- [ ] **Phase 5** — PyPI release, demo cast, docs, polish

## License

MIT — see [LICENSE](LICENSE).
