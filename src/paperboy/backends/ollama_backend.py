"""Ollama backend — the default, no-GPU-required path.

Talks to a locally running Ollama server over its HTTP API. This is the
frictionless default: anyone can `ollama pull <model>` and run paperBoy with no
GPU. The RTX 4050 quantized path (Phase 3) will implement the same `generate()`
contract behind a different backend.
"""

from __future__ import annotations

import httpx

from .base import Backend, BackendError


class OllamaBackend(Backend):
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: int = 300,
        embed_model: str = "nomic-embed-text",
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.embed_model = embed_model

    def generate(self, prompt: str, *, system: str | None = None, temperature: float | None = None) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["options"] = {"temperature": temperature}

        try:
            resp = httpx.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
        except httpx.ConnectError as exc:
            raise BackendError(
                f"Can't reach Ollama at {self.host}. Is it running? "
                "Install from https://ollama.com and start it, then retry."
            ) from exc
        except httpx.TimeoutException as exc:
            raise BackendError(
                f"Ollama timed out after {self.timeout}s generating with '{self.model}'."
            ) from exc

        if resp.status_code == 404:
            raise BackendError(
                f"Model '{self.model}' isn't available in Ollama. "
                f"Pull it first:  ollama pull {self.model}"
            )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise BackendError(f"Ollama returned {resp.status_code}: {resp.text[:200]}") from exc

        data = resp.json()
        return str(data.get("response", "")).strip()

    def embed(self, text: str, *, prefix: str = "") -> list[float]:
        try:
            resp = httpx.post(
                f"{self.host}/api/embeddings",
                json={"model": self.embed_model, "prompt": f"{prefix}{text}"},
                timeout=self.timeout,
            )
        except httpx.ConnectError as exc:
            raise BackendError(f"Can't reach Ollama at {self.host}. Is it running?") from exc

        if resp.status_code == 404:
            raise BackendError(
                f"Embedding model '{self.embed_model}' isn't available. "
                f"Pull it first:  ollama pull {self.embed_model}"
            )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise BackendError(f"Ollama embeddings returned {resp.status_code}: {resp.text[:200]}") from exc

        return [float(x) for x in resp.json().get("embedding", [])]
