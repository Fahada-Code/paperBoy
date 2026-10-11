# paperBoy 🗞️


A terminal AI research digest. Every morning it **fetches** new research (arXiv,
and later HN / GitHub trending), **ranks** it against the topics you actually
care about, **summarizes** the top picks with an LLM, and hands you a clean
**PDF brief** — no feeds to scroll, no frontend to babysit.

Built for myself to keep up with a field that moves too fast to read in full.
If you're a dev who wants the same thing, it's a `pip install` away.

## Why it exists

Staying current on ML/systems research is a part-time job. paperBoy makes it a
two-minute read. It's deliberately **terminal-first** and **self-hostable**: run
it on a free Ollama model with no GPU, or point it at a local quantized model on
your own GPU for the fast path.

## Status

🚧 **Phase 2 — the pipeline runs.** `paperboy run` fetches recent arXiv papers,
ranks them against your topics with embeddings, summarizes the top picks with a
local LLM, and writes a PDF digest.

```bash
# prerequisites: install Ollama, then:
#   ollama pull llama3.2:3b
#   ollama pull nomic-embed-text
cp config.example.yaml config.yaml   # edit your topics
paperboy doctor                      # check the backend is reachable
paperboy run                         # today's digest -> ./digests/*.pdf
paperboy run --top 5 --format markdown
paperboy summarize "long text..."    # or summarize ad-hoc text (arg or stdin)
```

## Local GPU serving (Phase 3)

paperBoy ships its own inference engine — a 4-bit quantized model (bitsandbytes
NF4) behind a FastAPI server with a **dynamic batching** scheduler that groups
concurrent requests into one GPU pass. Install the extras and run it:

```bash
pip install -e ".[local]"              # torch (CUDA), transformers, bitsandbytes, ...
paperboy serve                         # starts the engine on :8000
paperboy bench --batch-sizes 1,8       # measure batching's effect
```

**Benchmark** — Qwen2.5-1.5B-Instruct (4-bit NF4) on an RTX 4050 Laptop (6 GB),
24-request burst at concurrency 12, 128 new tokens each (representative run):

| config | throughput | p50 latency | p99 latency | VRAM |
|---|---:|---:|---:|---:|
| batching off (batch=1) | 7.4 tok/s | 74.0 s | 96.7 s | 1116 MB |
| batching on (batch=8) | **75.9 tok/s** | 3.9 s | 6.7 s | 1166 MB |

→ **~10x throughput and ~14x lower p99 latency for +50 MB VRAM.** Reproduce with
`paperboy bench`. (Part of batch=1's slowness is bitsandbytes' per-op overhead at
batch size 1, which batching amortizes — a batching × quantization effect.)

## Roadmap

- [x] **Phase 0** — repo, CLI skeleton, config
- [x] **Phase 1** — pluggable LLM backend (Ollama now; local GPU / hosted later), `summarize`, `doctor`
- [x] **Phase 2** — fetch (arXiv) → embed-rank → summarize → PDF/Markdown (`run`)
- [x] **Phase 3** — local GPU serving: 4-bit quantization + dynamic batching + benchmark (`serve`, `bench`)
- [ ] **Phase 4** — scheduled runs + cloud storage + metrics (autonomy)
- [ ] **Phase 5** — PyPI release, docs, polish

## Design principles

- **Terminal-first.** No web UI; it's a tool, not a product.
- **Pluggable backends.** Works with no GPU (Ollama) or screams on one (local quantized).
- **Honest.** The LLM *summarizes and explains* — it never predicts or advises.

## License

MIT — see [LICENSE](LICENSE).
