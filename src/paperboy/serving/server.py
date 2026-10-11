"""FastAPI wrapper around the InferenceEngine.

Exposes a tiny HTTP API so paperBoy's `local` backend (and the benchmark) can
talk to the engine exactly like it talks to Ollama. Endpoints:

  GET  /health    -> {status, model, vram_mb}
  POST /generate  -> {response, n_prompt_tokens, n_gen_tokens, latency_s}
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from .engine import InferenceEngine, Request


class GenerateIn(BaseModel):
    prompt: str
    system: str | None = None
    max_new_tokens: int = 128
    temperature: float = 0.2


class GenerateOut(BaseModel):
    response: str
    n_prompt_tokens: int
    n_gen_tokens: int
    latency_s: float


def create_app() -> FastAPI:
    app = FastAPI(title="paperBoy inference engine")
    engine = InferenceEngine(
        model_id=os.environ.get("PAPERBOY_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        quantize=os.environ.get("PAPERBOY_QUANTIZE", "4bit"),
        max_batch_size=int(os.environ.get("PAPERBOY_MAX_BATCH", "8")),
        max_wait_ms=int(os.environ.get("PAPERBOY_MAX_WAIT_MS", "10")),
    )
    app.state.engine = engine

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": engine.model_id, "vram_mb": round(engine.vram_mb(), 1)}

    @app.post("/generate", response_model=GenerateOut)
    def generate(body: GenerateIn) -> GenerateOut:
        req = engine.submit(
            Request(
                prompt=body.prompt,
                system=body.system,
                max_new_tokens=body.max_new_tokens,
                temperature=body.temperature,
            )
        )
        return GenerateOut(
            response=req.result,
            n_prompt_tokens=req.n_prompt_tokens,
            n_gen_tokens=req.n_gen_tokens,
            latency_s=round(req.latency, 4),
        )

    return app
