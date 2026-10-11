"""Local backend — talks to paperBoy's own inference server (Phase 3).

Same HTTP contract as the Ollama backend, so paperBoy doesn't care which is
serving the model. Point it at a running `paperboy serve` (the quantized,
dynamically-batched engine) via `kind: local` in config.
"""

from __future__ import annotations

import httpx

from .base import Backend, BackendError


class LocalBackend(Backend):
    def __init__(self, host: str = "http://localhost:8000", timeout: int = 300):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, *, system: str | None = None, temperature: float | None = None) -> str:
        payload = {
            "prompt": prompt,
            "system": system,
            "temperature": 0.2 if temperature is None else temperature,
        }
        try:
            resp = httpx.post(f"{self.host}/generate", json=payload, timeout=self.timeout)
        except httpx.ConnectError as exc:
            raise BackendError(
                f"Can't reach the paperBoy engine at {self.host}. "
                "Start it with:  paperboy serve"
            ) from exc
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise BackendError(f"Engine returned {resp.status_code}: {resp.text[:200]}") from exc
        return str(resp.json().get("response", "")).strip()

    # The local engine does generation only; ranking still uses Ollama embeddings.
    # (embed() inherits the base's "not supported" behavior.)
