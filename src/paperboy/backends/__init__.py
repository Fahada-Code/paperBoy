"""Backend registry: pick an LLM backend from config.

The factory is the single place that maps a config `kind` to a concrete backend,
so the rest of the app depends only on the `Backend` contract.
"""

from __future__ import annotations

from ..config import BackendConfig
from .base import Backend, BackendError
from .local_backend import LocalBackend
from .ollama_backend import OllamaBackend

__all__ = ["Backend", "BackendError", "get_backend"]


def get_backend(cfg: BackendConfig) -> Backend:
    kind = cfg.kind.lower()
    if kind == "ollama":
        return OllamaBackend(
            model=cfg.model,
            host=cfg.host,
            timeout=cfg.timeout,
            embed_model=cfg.embed_model,
        )
    if kind == "local":
        # Talks to `paperboy serve` (default :8000). Set host accordingly in config.
        host = cfg.host if ":8000" in cfg.host else "http://localhost:8000"
        return LocalBackend(host=host, timeout=cfg.timeout)
    if kind == "hosted":
        raise BackendError("The 'hosted' backend isn't implemented yet. Use kind: ollama for now.")
    raise BackendError(f"Unknown backend kind: {cfg.kind!r}. Expected one of: ollama, local, hosted.")
