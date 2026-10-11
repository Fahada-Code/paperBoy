"""Benchmark the inference engine: dynamic batching vs none.

Fires a burst of concurrent requests at the engine and measures aggregate
throughput (tokens/sec), per-request latency (p50/p99), and VRAM. Running it
with batching off (max_batch_size=1) and on (e.g. 8) shows the real speedup --
these are the numbers that back the resume bullet.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from statistics import median
import time

from .engine import InferenceEngine, Request

DEFAULT_PROMPT = (
    "Summarize in two sentences why dynamic batching improves LLM inference "
    "throughput on a single GPU."
)


@dataclass
class BenchResult:
    label: str
    n_requests: int
    concurrency: int
    wall_s: float
    total_gen_tokens: int
    throughput_tok_s: float
    requests_per_s: float
    latency_p50_s: float
    latency_p99_s: float
    peak_vram_mb: float

    def as_row(self) -> str:
        return (
            f"{self.label:<22} "
            f"{self.throughput_tok_s:>8.1f} tok/s  "
            f"{self.requests_per_s:>6.2f} req/s  "
            f"p50 {self.latency_p50_s:>6.2f}s  "
            f"p99 {self.latency_p99_s:>6.2f}s  "
            f"VRAM {self.peak_vram_mb:>6.0f}MB"
        )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = min(len(ordered) - 1, int(round(pct / 100.0 * (len(ordered) - 1))))
    return ordered[k]


def run_benchmark(
    engine: InferenceEngine,
    label: str,
    n_requests: int = 32,
    concurrency: int = 16,
    max_new_tokens: int = 128,
    prompt: str = DEFAULT_PROMPT,
) -> BenchResult:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def one() -> Request:
        return engine.submit(
            Request(prompt=prompt, max_new_tokens=max_new_tokens, temperature=0.0)
        )

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda _: one(), range(n_requests)))
    wall = time.perf_counter() - start

    latencies = [r.latency for r in results]
    total_gen = sum(r.n_gen_tokens for r in results)
    peak_vram = (
        torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    )

    return BenchResult(
        label=label,
        n_requests=n_requests,
        concurrency=concurrency,
        wall_s=wall,
        total_gen_tokens=total_gen,
        throughput_tok_s=total_gen / wall if wall else 0.0,
        requests_per_s=n_requests / wall if wall else 0.0,
        latency_p50_s=median(latencies) if latencies else 0.0,
        latency_p99_s=_percentile(latencies, 99),
        peak_vram_mb=peak_vram,
    )
